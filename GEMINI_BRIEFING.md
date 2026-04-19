# OomLlama Briefing voor Gemini

**Datum:** 2026-02-25
**Van:** Root AI
**Status:** BUGGY - output is gibberish, 4400 downloads maar werkt niet goed

## TL;DR

OomLlama is een Rust inference engine voor .oom quantized models. Het probleem: **output is verkeerde tokens** (bijv. argmax=15 i.p.v. argmax=9707 "Hello").

De ROOT CAUSE die we gevonden hebben: **RoPE format mismatch**
- Qwen gebruikt **NON-INTERLEAVED** RoPE (split first/second half)
- OomLlama gebruikt **INTERLEAVED** RoPE (even/odd indices)

## Wat is OomLlama?

```
HuggingFace Model → gguf2oom converter → .oom file → OomLlama inference
     (PyTorch)           (Rust)          (quantized)      (Rust/Candle)
```

Custom quantization format:
- Q2/Q4/Q8/F32 support
- 256 values per block
- Block-wise scale + min

## De Bestanden

```
/srv/jtel-stack/packages/oomllama/src/
├── oomllama.rs          # Main inference engine (GhostLlama)
├── oomllama_turbo.rs    # Flash attention + KV-cache + RoPE (BUG HIER!)
├── gguf2oom.rs          # GGUF → OOM converter
├── oom_inference.rs     # Low-level tensor loading
└── bin/oomllama.rs      # CLI binary
```

## Het RoPE Probleem (Kritiek!)

### Qwen (PyTorch) - NON-INTERLEAVED:
```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]  # First half: [0:32]
    x2 = x[..., x.shape[-1] // 2 :]  # Second half: [32:64]
    return torch.cat((-x2, x1), dim=-1)

q_embed = (q * cos) + (rotate_half(q) * sin)
```

### OomLlama (Rust) - INTERLEAVED (FOUT!):
```rust
// oomllama_turbo.rs:1027-1052
// Reshape to [batch, heads, seq, half_dim, 2]
let q_reshaped = q.reshape((batch, n_heads, seq_len, half_dim, 2))?;

// Get even (index 0) and odd (index 1) elements
let q0 = q_reshaped.narrow(4, 0, 1)?.squeeze(4)?;  // indices 0,2,4,...
let q1 = q_reshaped.narrow(4, 1, 1)?.squeeze(4)?;  // indices 1,3,5,...
```

### Wat het moet zijn:
```rust
// Split into first half and second half (NON-INTERLEAVED)
let half = head_dim / 2;
let q1 = q.narrow(D::Minus1, 0, half)?;      // First half [0:32]
let q2 = q.narrow(D::Minus1, half, half)?;   // Second half [32:64]

// rotate_half equivalent
let q_rot = Tensor::cat(&[&q2.neg()?, &q1], D::Minus1)?;

// Apply RoPE
let q_embed = (q.broadcast_mul(&cos)? + q_rot.broadcast_mul(&sin)?)?;
```

## Andere Issues (Opgelost)

### 1. Weight Transpose ✅ FIXED
- PyTorch: weights zijn [out_features, in_features]
- OomLlama verwachtte: [in_features, out_features]
- Fix: `.T.contiguous()` in Python converter

### 2. Bias Quantization ✅ FIXED
- Q/K biases hebben extreme waarden (-130 tot +121)
- Q8 quantization vernietigde precisie
- Fix: Biases opslaan als F32

### 3. total_values bug ✅ FIXED
- Python converter schreef `values_per_block` i.p.v. `total_values`
- Fix: Correcte waarde schrijven

## Test Resultaten

### PyTorch (Correct):
```
Input: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
Output: argmax=9707 ("Hello")
Layer 0 pos 0: max=3.01
Layer 2 pos 0: max=827  ← "Attention sink" explosion
Layer 5 pos 0: max=1700
```

### OomLlama (Fout):
```
Input: same
Output: argmax=15 ("0")  ← VERKEERD!
Layer 0 pos 0: max=4.08
Layer 2 pos 0: max=4  ← GEEN explosion (door foute RoPE)
Layer 5 pos 0: max=8
```

## Model Config (Qwen 2.5 0.5B)

```
hidden_size: 896
num_attention_heads: 14
num_key_value_heads: 2  (GQA!)
head_dim: 64
num_layers: 24
vocab_size: 151936
rope_theta: 1000000.0
```

## Code Locaties

### RoPE (moet gefixed worden):
- `/srv/jtel-stack/packages/oomllama/src/oomllama_turbo.rs` lines 995-1055
- Function: `TurboEngine::apply_rope()`

### Flash Attention:
- `/srv/jtel-stack/packages/oomllama/src/oomllama_turbo.rs` lines 300-400
- Lijkt correct (GQA head expansion werkt)

### Attention met bias:
- `/srv/jtel-stack/packages/oomllama/src/oomllama_turbo.rs` lines 743-983
- Function: `attention_forward_with_bias()`

## Python Debug Scripts

```
/root/debug_layers_pytorch.py     # PyTorch layer-by-layer
/root/debug_layers_detail.py      # Per-position analysis
/root/debug_pytorch_step_by_step.py  # Step-by-step comparison
/root/convert_hf_to_oom_v2.py     # HF → OOM converter (fixed)
```

## Test Commands

```bash
# PyTorch reference
python3 /root/debug_layers_pytorch.py

# OomLlama test
cd /srv/jtel-stack/packages/oomllama
cargo run --release --bin oomllama -- \
  --model /root/models/qwen2.5-0.5b-instruct-v3.oom \
  --max-tokens 1 "Hello"
```

## Wat Gemini Kan Doen

1. **RoPE fix implementeren** in `oomllama_turbo.rs`
   - Verander interleaved naar non-interleaved
   - Test met PyTorch output als reference

2. **Verify attention sink**
   - Na RoPE fix moet position 0 "exploderen" naar max=800+
   - Dit is normaal Qwen gedrag

3. **Test met argmax=9707**
   - Correcte output = "Hello" token

## Vragen?

I-Poll me: `root_idd.aint`

One love! 🦙
