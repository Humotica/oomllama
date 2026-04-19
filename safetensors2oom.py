#!/usr/bin/env python3
"""
SafeTensors → OOM converter.

Converts HuggingFace safetensors (bf16/f32) directly to OOM Q8 format.
This avoids the Q3_K→f32→Q8 double quantization that caused error amplification
in the gguf2oom path.

Single quantization step: bf16 → f32 → Q8 (or F32 for norms/biases)

Usage:
    python safetensors2oom.py <model_name_or_path> <output.oom> [--q4] [--q8]
    python safetensors2oom.py Qwen/Qwen2.5-7B-Instruct /mnt/ohm-0/qwen2.5-7b-bf16-q8.oom
    python safetensors2oom.py ./local-model/ output.oom --q4
"""
import struct
import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# OOM format constants
OOM_MAGIC = b"OOML"
OOM_VERSION = 1
BLOCK_SIZE = 256

def quantize_q8(values: np.ndarray) -> tuple:
    """Quantize f32 values to Q8 (256 levels).
    Returns (quant_type, blocks) where blocks is list of (scale, min_val, qdata)."""
    num_values = len(values)
    num_blocks = (num_values + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks = []

    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, num_values)
        block = values[start:end]

        bmin = float(block.min())
        bmax = float(block.max())
        brange = bmax - bmin

        if abs(brange) < 1e-9:
            scale = 0.0
            qdata = np.zeros(len(block), dtype=np.uint8)
        else:
            scale = brange / 255.0
            normalized = (block - bmin) / scale
            qdata = np.clip(np.round(normalized), 0, 255).astype(np.uint8)

        blocks.append((scale, bmin, qdata))

    return 8, blocks


def quantize_q4(values: np.ndarray) -> tuple:
    """Quantize f32 values to Q4 (16 levels, 4 bits per value)."""
    num_values = len(values)
    num_blocks = (num_values + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks = []

    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, num_values)
        block = values[start:end]

        bmin = float(block.min())
        bmax = float(block.max())
        brange = bmax - bmin

        if abs(brange) < 1e-9:
            scale = 0.0
            # Pack zeros: 2 values per byte
            packed_len = (len(block) + 1) // 2
            qdata = np.zeros(packed_len, dtype=np.uint8)
        else:
            scale = brange / 15.0
            normalized = (block - bmin) / scale
            q4_vals = np.clip(np.round(normalized), 0, 15).astype(np.uint8)
            # Pack 2 values per byte (low nibble first)
            packed_len = (len(q4_vals) + 1) // 2
            qdata = np.zeros(packed_len, dtype=np.uint8)
            for i, v in enumerate(q4_vals):
                byte_idx = i // 2
                if i % 2 == 0:
                    qdata[byte_idx] |= v
                else:
                    qdata[byte_idx] |= (v << 4)

        blocks.append((scale, bmin, qdata))

    return 4, blocks


def encode_f32(values: np.ndarray) -> tuple:
    """Encode raw F32 values (no quantization)."""
    raw_bytes = values.astype(np.float32).tobytes()
    # Single block, scale=0, min=0, data = raw f32 bytes
    return 0, [(0.0, 0.0, raw_bytes)]


class OomWriter:
    """Write OOM format files, matching the Rust OomWriter exactly."""

    def __init__(self, path: str):
        self.path = path
        self.tensor_count = 0
        self.tensor_data = bytearray()

    def add_tensor(self, name: str, quant_type: int, blocks: list, total_values: int):
        """Add a tensor with pre-quantized blocks."""
        name_bytes = name.encode("utf-8")

        # Tensor header
        self.tensor_data += struct.pack("<I", len(name_bytes))
        self.tensor_data += name_bytes
        self.tensor_data += struct.pack("<B", quant_type)
        self.tensor_data += struct.pack("<I", len(blocks))
        self.tensor_data += struct.pack("<I", total_values)

        # Block data
        for scale, min_val, qdata in blocks:
            self.tensor_data += struct.pack("<f", scale)
            self.tensor_data += struct.pack("<f", min_val)
            if isinstance(qdata, np.ndarray):
                data_bytes = qdata.tobytes()
            elif isinstance(qdata, bytes):
                data_bytes = qdata
            else:
                data_bytes = bytes(qdata)
            self.tensor_data += struct.pack("<I", len(data_bytes))
            self.tensor_data += data_bytes

        self.tensor_count += 1

    def finish(self):
        """Write header + all tensor data to file."""
        with open(self.path, "wb") as f:
            # Header: magic(4) + version(u32) + tensor_count(u32)
            f.write(OOM_MAGIC)
            f.write(struct.pack("<I", OOM_VERSION))
            f.write(struct.pack("<I", self.tensor_count))
            # All tensor data
            f.write(self.tensor_data)
        print(f"\nWrote {self.tensor_count} tensors to {self.path}")


def resolve_model_path(model_name_or_path: str) -> Path:
    """Resolve model name or path to the directory with safetensors files."""
    p = Path(model_name_or_path)
    if p.is_dir() and any(p.glob("*.safetensors")):
        return p

    # Try HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{model_name_or_path.replace('/', '--')}"
    if model_dir.exists():
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            # Get the latest snapshot
            snap_dirs = sorted(snapshots.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for snap in snap_dirs:
                if any(snap.glob("*.safetensors")):
                    return snap

    # Try downloading
    print(f"Model not found locally, attempting download: {model_name_or_path}")
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(model_name_or_path)
        return Path(local_dir)
    except Exception as e:
        print(f"Could not download: {e}")
        sys.exit(1)


def load_tensor_index(model_dir: Path) -> dict:
    """Load tensor → file mapping from safetensors index."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            data = json.load(f)
        return data["weight_map"]
    else:
        # Single file model
        st_files = list(model_dir.glob("*.safetensors"))
        if not st_files:
            print("No safetensors files found!")
            sys.exit(1)
        # Return all tensors mapped to the single file
        from safetensors import safe_open
        result = {}
        for sf in st_files:
            with safe_open(str(sf), framework="numpy") as f:
                for key in f.keys():
                    result[key] = sf.name
        return result


def should_preserve_f32(name: str) -> bool:
    """Check if tensor should be preserved as F32 (norms, biases)."""
    return "norm" in name or name.endswith(".bias")


def convert_safetensors_to_oom(
    model_name_or_path: str,
    output_path: str,
    quant_level: str = "q8",
):
    """Convert safetensors model to OOM format."""
    print(f"Resolving model: {model_name_or_path}")
    model_dir = resolve_model_path(model_name_or_path)
    print(f"Model directory: {model_dir}")

    # Load tensor index
    weight_map = load_tensor_index(model_dir)
    total_tensors = len(weight_map)
    print(f"Found {total_tensors} tensors")

    # Group tensors by file for efficient loading
    file_tensors = {}
    for tensor_name, filename in weight_map.items():
        if filename not in file_tensors:
            file_tensors[filename] = []
        file_tensors[filename].append(tensor_name)

    print(f"Spread across {len(file_tensors)} safetensor files")
    print(f"Output quantization: {quant_level.upper()}")

    # Create OOM writer
    writer = OomWriter(output_path)

    from safetensors import safe_open
    import torch

    processed = 0
    start_time = time.time()

    # Pre-group by file
    for filename, tensors_in_file in file_tensors.items():
        filepath = model_dir / filename
        print(f"\nLoading {filename} ({len(tensors_in_file)} tensors)...")

        # Use PyTorch framework because numpy doesn't support bfloat16
        with safe_open(str(filepath), framework="pt") as f:
            for tensor_name in sorted(tensors_in_file):
                processed += 1
                tensor = f.get_tensor(tensor_name)

                # Convert any dtype (bf16, f16, f32) to f32 numpy
                values = tensor.float().numpy().flatten()
                total_values = len(values)

                # Progress
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_tensors - processed) / rate if rate > 0 else 0
                print(
                    f"\r  [{processed}/{total_tensors}] {tensor_name:60s} "
                    f"({total_values:>10,} vals) ETA: {eta:.0f}s",
                    end="", flush=True
                )

                # Decide quantization
                if should_preserve_f32(tensor_name):
                    qt, blocks = encode_f32(values)
                else:
                    if quant_level == "q4":
                        qt, blocks = quantize_q4(values)
                    else:
                        qt, blocks = quantize_q8(values)

                writer.add_tensor(tensor_name, qt, blocks, total_values)

    # Finalize
    elapsed = time.time() - start_time
    print(f"\n\nProcessed {processed} tensors in {elapsed:.1f}s")
    writer.finish()

    # Size info
    output_size = os.path.getsize(output_path)
    print(f"Output size: {output_size / 1e9:.2f} GB")

    # Calculate expected source size (bf16 = 2 bytes per value)
    print(f"Quantization: bf16 -> {quant_level.upper()}")
    print("Done!")


def main():
    if len(sys.argv) < 3:
        print("Usage: safetensors2oom.py <model_name_or_path> <output.oom> [--q4|--q8]")
        print()
        print("Examples:")
        print("  python safetensors2oom.py Qwen/Qwen2.5-7B-Instruct /mnt/ohm-0/qwen2.5-7b-bf16-q8.oom")
        print("  python safetensors2oom.py Qwen/Qwen2.5-0.5B-Instruct output.oom --q4")
        print("  python safetensors2oom.py ./local-model/ output.oom")
        sys.exit(1)

    model = sys.argv[1]
    output = sys.argv[2]
    quant = "q8"  # default
    if "--q4" in sys.argv:
        quant = "q4"
    elif "--q8" in sys.argv:
        quant = "q8"

    convert_safetensors_to_oom(model, output, quant)


if __name__ == "__main__":
    main()
