# Spaceshuttle — Encrypted Memory Illusion for AI

**AES-256-GCM encrypted RAM for LLM inference, transparent via hardware MMU traps.**

Page fault → decrypt → decompress → inject → resume. The application sees normal memory.
Without a valid identity claim: dead material (zero pages). Identity IS the memory.

## Results (Xeon W-2135 @ 3.70GHz, 2017 hardware)

| Test | Pages | Throughput | p50 | p99 |
|------|-------|-----------|-----|-----|
| Spaceshuttle (64 pages, 4KB) | 64 | — | 112 µs | 207 µs |
| Scaletest (16K pages, 4KB) | 16,384 | 152 MB/s | 53 µs | 122 µs |
| GGUF Shuttle (1GB model weights) | 262,144 | 50 MB/s | 67 µs | 83 µs |
| **HugePage Shuttle (2MB pages)** | **64** | **184 MB/s** | **10.8 ms** | **18.1 ms** |

HugePages: **3.8x faster**, 512x fewer page faults, same encryption.

## Quick Start

```bash
# Requirements: Linux, Rust 1.70+, libclang-dev
# Optional: a GGUF model file for the GGUF/HugePage tests

# 1. Clone and build
git clone <repo-url> tibet-kernel
cd tibet-kernel
cargo build --release

# 2. Enable userfaultfd (one-time, or add to /etc/sysctl.conf)
sudo sysctl -w vm.unprivileged_userfaultfd=1

# 3. Run the demo (no model file needed)
./target/release/spaceshuttle

# 4. Run the scale test (no model file needed)
./target/release/scaletest

# 5. GGUF Shuttle (needs a GGUF model in Ollama's blob dir)
#    Automatically finds the largest model file.
./target/release/gguf-shuttle

# 6. HugePage comparison (needs GGUF + hugepage allocation)
sudo sysctl -w vm.nr_hugepages=128    # 128 × 2MB = 256MB
./target/release/hugepage-shuttle
```

## One-Liner Setup (Debian/Ubuntu)

```bash
sudo apt-get install -y build-essential libclang-dev && \
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
source ~/.cargo/env && \
sudo sysctl -w vm.unprivileged_userfaultfd=1 && \
cargo build --release && \
./target/release/spaceshuttle
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Application (LLM engine, Redis, PostgreSQL, ...)           │
│  Sees normal memory. Merkt niets.                           │
└────────────────────────┬────────────────────────────────────┘
                         │ pointer dereference
┌────────────────────────▼────────────────────────────────────┐
│  CPU — Page Fault (userfaultfd, since Linux 4.11)           │
│  Thread freezes automatically. No SIGSEGV.                  │
└────────────────────────┬────────────────────────────────────┘
                         │ fault event
┌────────────────────────▼────────────────────────────────────┐
│  Trust Kernel — Bifurcation Engine                          │
│                                                             │
│  1. Receive fault → compute page index                      │
│  2. JIS clearance check (identity-based access)             │
│  3. AES-256-GCM decrypt (hardware AES-NI / VAES)           │
│  4. zstd decompress (if compressed mode)                    │
│  5. uffd.copy() → inject into physical page                 │
│  6. Thread resumes. Application continues.                  │
│                                                             │
│  Wrong identity? → zero page (dead material)                │
└─────────────────────────────────────────────────────────────┘
```

## Binaries

| Binary | What it does | Needs model? |
|--------|-------------|-------------|
| `spaceshuttle` | 64-page encrypted memory demo + access control test | No |
| `scaletest` | 64→16384 pages, O(1) latency proof, encrypt vs compress+encrypt | No |
| `raidtest` | RAM RAID-0 controller with LRU eviction and striping | No |
| `gguf-shuttle` | Real GGUF model weights through encrypted arena (64MB→1GB) | Yes |
| `hugepage-shuttle` | 4KB vs 2MB HugePage comparison on GGUF weights | Yes |

## Crypto Stack

- **AES-256-GCM** — Authenticated encryption (hardware-accelerated via AES-NI/VAES)
- **X25519** — Ephemeral Diffie-Hellman key agreement
- **HKDF-SHA256** — Per-page key derivation (session keys after first DH)
- **zstd level 3** — Compression before encryption (less ciphertext = faster decrypt)
- **JIS Claims** — Identity-based access control (clearance levels)

## GCP / Cloud Testing

This runs on any Linux with userfaultfd (kernel 4.11+). For best results:

| Instance | Expected throughput | Why |
|----------|-------------------|-----|
| n2-standard (Ice Lake) | ~400 MB/s | VAES AVX-512, DDR4 |
| c3-highmem (Sapphire Rapids) | ~800+ MB/s | VAES AVX-512, DDR5, larger TLB |
| c3d (Genoa) | ~700+ MB/s | VAES AVX-512, DDR5 |

To test with a large model:
```bash
# Install Ollama and pull a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:32b    # ~19GB Q4_K_M

# Allocate HugePages for the full model
sudo sysctl -w vm.nr_hugepages=10000   # 10000 × 2MB = 20GB

# Run
./target/release/hugepage-shuttle
```

## What This Proves

1. **Encrypted-by-default RAM is viable for AI** — 184 MB/s on 2017 hardware
2. **Zero application changes** — userfaultfd is transparent to the application
3. **Identity IS the key** — without a valid JIS claim, memory is dead material
4. **Compression helps encryption** — less ciphertext = faster AES-GCM = counterintuitive speedup
5. **HugePages are transformative** — 3.8x speedup, 512x fewer faults

## Part of the TIBET Ecosystem

- **TIBET** — Traceable Intent-Based Event Tokens
- **JIS** — Identity-based access control
- **SNAFT** — Syscall allowlist enforcement
- **Trust Kernel** — Dual-kernel sandbox (Voorproever + Archivaris)
- **AInternet** — AI-to-AI network with .aint domains

Built by [Humotica](https://humotica.com) — AI en mens in symbiose.

## License

MIT
