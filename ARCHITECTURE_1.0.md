# OomLlama 1.0 Architecture Vision

## Core Innovation: Identity-Gated Lazy Loading over Encrypted Memory Pages

Nobody else does this. llama.cpp, vLLM, Ollama — none of them combine
identity-based access control with lazy memory-mapped inference.

## 10 Components

### 1. Spaceshuttle (userfaultfd) — Lazy Model Loading
- mmap 20GB .oom file, physically loaded: 0 bytes
- Page fault on first access → decrypt → decompress → inject
- After 10 prompts: hot layers in RAM, cold layers on disk
- Result: instant startup, progressive loading

### 2. RAM-RAID — 70B Always Fits
- P520: even layers (10GB), DL360: odd layers (10GB)
- Genesis Highway: 10Gbps, 53us per page fault
- Transparent via userfaultfd, encrypted via Trust Kernel
- Third machine → 405B models without quantization compromises

### 3. Zstd Compression — Smaller and Faster
- Quantized weights compress 2-4x (20GB → 5-10GB)
- Sparse attention masks: 8-33x compression
- Compressed + encrypted faster than encrypt-only (proven by Spaceshuttle)

### 4. Von Braun Mode — Parallel Layer Processing
- 8 attention heads x 8 cores parallel per layer
- Each head: decrypt page → compute → next
- Per layer speedup: 4-8x on multi-head attention

### 5. Session Keys — Zero-Overhead Encryption
- Without session keys: 76us/block, with: 5us/block
- 70B model, 5000 blocks: 380ms → 25ms total decrypt
- Unnoticeable latency

### 6. Trust Kernel Airlock — Sandboxed Inference
- Prompt → Voorproever (SNAFT, 2.6us) → Bus → Inference (Kernel B)
- Injection: KILL in 2.6us, never reaches model
- Model runs in clean vault, untouched by untested input

### 7. QUIC MUX — Streaming Inference
- Token-by-token over QUIC streams
- No head-of-line blocking
- WiFi→5G handoff: stream continues
- Parallel users on separate streams

### 8. TIBET Provenance — Auditable AI
- Every inference = TIBET token
- ERIN: model + prompt_hash
- ERAAN: user.aint + clearance
- EROMHEEN: device + GPU + layer distribution
- ERACHTER: tokens + latency + cache hits
- EU AI Act Article 12 compliant

### 9. JIS Clearance Per Layer
- Layer 0-31: PUBLIC (base model)
- Layer 32-48: CONFIDENTIAL (enterprise fine-tune)
- Layer 49-64: SECRET (defense fine-tune)
- One model, three clearance levels, cryptographically enforced per page

### 10. tibet-bench — Benchmark Mode
- Load time, first token, throughput, cache hits
- TIBET tokens minted per request
- Trust overhead as % of total latency
- Tibet Points scoring

## Full Pipeline
```
.oom file (encrypted + zstd + JIS clearance per layer)
  → userfaultfd arena (Spaceshuttle, zero startup)
  → Local layers (even, P520) + Remote layers (odd, DL360 via QUIC MUX)
  → Trust Kernel Airlock (SNAFT voorproever → Bus → clean inference)
  → Von Braun parallel attention + session key decrypt
  → Output + TIBET token (streaming via QUIC MUX)
  → Provenance: who, what, when, how, clearance
```
