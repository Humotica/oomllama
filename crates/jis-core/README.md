# jis-core

Cryptographic identity where intent comes first.

[![PyPI](https://img.shields.io/pypi/v/jis-core)](https://pypi.org/project/jis-core/)
[![npm](https://img.shields.io/npm/v/@humotica/jis-core)](https://www.npmjs.com/package/@humotica/jis-core)
[![IETF Draft](https://img.shields.io/badge/IETF-draft--vandemeent--jis--identity-blue)](https://datatracker.ietf.org/doc/draft-vandemeent-jis-identity/)
[![Whitepaper](https://img.shields.io/badge/Zenodo-DOI:10.5281/zenodo.18712238-green)](https://doi.org/10.5281/zenodo.18712238)

JIS (JTel Identity Standard) is a decentralized identity system built on bilateral intent verification. No action happens without declared purpose. No identity resolves without mutual consent.

Ed25519 Rust kernel. Python, JavaScript/WASM, and C bindings. 50KB compiled.

## Install

```bash
pip install jis-core          # Python (compiled Rust extension)
```

```bash
npm install @humotica/jis-core  # JavaScript (Rust/WASM, 176KB)
```

```toml
[dependencies]
jis-core = "0.1"              # Rust (native)
```

```c
#include "did_jis.h"          // C (FFI bindings for embedded/6G)
```

## Quick Start

### Python

```python
from jis_core import DIDEngine, DIDDocumentBuilder

# Generate identity (Ed25519 keypair)
engine = DIDEngine()
did = engine.create_did("alice")
print(did)  # did:jis:alice

# Sign and verify
signature = engine.sign("I agree to these terms")
assert engine.verify("I agree to these terms", signature)  # True

# Build a DID Document with bilateral consent
builder = DIDDocumentBuilder(did)
builder.add_verification_method("key-1", engine.public_key)
builder.add_authentication("key-1")
builder.add_consent_service("https://api.example.com/consent")
builder.add_tibet_service("https://api.example.com/tibet")
doc = builder.build()  # W3C-compatible JSON
```

### JavaScript

```javascript
import { WasmDIDEngine } from '@humotica/jis-core';

const engine = new WasmDIDEngine();
const did = engine.createDid("alice");
const sig = engine.sign("I agree to these terms");
console.log(engine.verify("I agree to these terms", sig)); // true
```

### Rust

```rust
use jis_core::{DIDEngine, DIDDocumentBuilder};

let engine = DIDEngine::new();
let did = engine.create_did("alice");

let doc = DIDDocumentBuilder::new(&did).unwrap()
    .add_verification_method_ed25519("key-1", &engine.public_key_hex())
    .add_authentication("key-1")
    .add_consent_service("https://api.example.com/consent")
    .build();
```

## Why JIS

The problem with existing identity systems is that they answer WHO without asking WHY.

- **X.509** proves a server is who it claims. It says nothing about intent.
- **OAuth** delegates access. It doesn't record why access was requested.
- **W3C DID** decentralizes identity. But resolution is unconditional.

JIS adds two things:

1. **Bilateral consent** — identity only resolves when both parties agree on the intent.
2. **Provenance binding** — every identity action generates a TIBET token that records what happened, why, and with what evidence.

The result: identity that cannot be used without a declared, verifiable purpose.

| | X.509 | OAuth | W3C DID | **JIS** |
|---|---|---|---|---|
| Decentralized | No | No | Yes | **Yes** |
| Intent required | No | No | No | **Yes** |
| Bilateral consent | No | No | No | **Yes** |
| Audit trail | No | No | No | **TIBET** |
| Ed25519 native | No | No | Optional | **Yes** |
| Runs in browser | Complex | Token-based | Varies | **WASM (176KB)** |
| Runs on MCU | No | No | No | **C FFI** |
| AI agent friendly | No | Partial | Partial | **Built for it** |

## Core Concepts

### jis: URIs

Every identity is a `jis:` URI. Hierarchical, human-readable, machine-verifiable.

```
jis:alice                      # Person
jis:humotica:root_idd          # AI agent in an organization
jis:device:6G:sensor-001       # IoT device
jis:service:api:payments       # Service endpoint
```

### Bilateral Consent

A JIS DID Document includes a `BilateralConsentService` endpoint. Before any identity interaction proceeds, both parties must agree on the purpose. This is not optional — it's structural.

```json
{
  "id": "did:jis:alice#bilateral-consent",
  "type": "BilateralConsentService",
  "serviceEndpoint": "https://api.example.com/consent"
}
```

### TIBET Provenance Binding

Every JIS identity action can be bound to a TIBET provenance token — recording ERIN (what), ERAAN (dependencies), EROMHEEN (context), and ERACHTER (intent). This is the audit trail that regulators need and that existing identity systems don't provide.

```json
{
  "id": "did:jis:alice#tibet-provenance",
  "type": "TIBETProvenanceService",
  "serviceEndpoint": "https://api.example.com/tibet"
}
```

### Signed DID Documents

Every DID Document can be cryptographically signed with an Ed25519Signature2020 proof:

```python
engine = DIDEngine()
signed_doc = engine.create_document("did:jis:alice")
# Returns JSON with document + Ed25519Signature2020 proof
```

Output:

```json
{
  "document": {
    "@context": [
      "https://www.w3.org/ns/did/v1",
      "https://w3id.org/security/suites/ed25519-2020/v1",
      "https://humotica.com/ns/jis/v1"
    ],
    "id": "did:jis:alice",
    "verificationMethod": [{
      "id": "did:jis:alice#key-1",
      "type": "Ed25519VerificationKey2020",
      "controller": "did:jis:alice",
      "publicKeyHex": "a1b2c3..."
    }],
    "authentication": ["did:jis:alice#key-1"],
    "assertionMethod": ["did:jis:alice#key-1"]
  },
  "proof": {
    "type": "Ed25519Signature2020",
    "verificationMethod": "did:jis:alice#key-1",
    "proofPurpose": "assertionMethod",
    "proofValue": "e4f5a6..."
  }
}
```

## API Reference

### DIDEngine

| Method | Description |
|--------|-------------|
| `DIDEngine()` | Generate new Ed25519 keypair |
| `.from_secret_key(hex)` | Load from 32-byte secret (hex) |
| `.public_key` | Public key (64-char hex) |
| `.public_key_multibase` | Public key (multibase f-prefix) |
| `.create_did(id)` | Create `did:jis:{id}` |
| `.create_did_from_key()` | Create DID from SHA-256 of public key |
| `.sign(message)` | Ed25519 sign, returns hex |
| `.verify(message, signature)` | Verify signature |
| `.verify_with_key(msg, sig, pubkey)` | Static: verify with external key |
| `.create_document(did)` | Create signed DID Document (JSON) |

### DIDDocumentBuilder

| Method | Description |
|--------|-------------|
| `DIDDocumentBuilder(did)` | Create builder for a DID |
| `.set_controller(did)` | Set document controller |
| `.add_verification_method(id, pubkey)` | Add Ed25519 key |
| `.add_authentication(key_id)` | Add auth method reference |
| `.add_assertion_method(key_id)` | Add assertion method reference |
| `.add_service(id, type, endpoint)` | Add generic service |
| `.add_consent_service(endpoint)` | Add bilateral consent endpoint |
| `.add_tibet_service(endpoint)` | Add TIBET provenance endpoint |
| `.build()` | Build W3C-compatible DID Document (JSON) |

### Module Functions

| Function | Description |
|----------|-------------|
| `parse_did(did)` | Parse `did:jis:*` into (method, id) |
| `is_valid_did(did)` | Validate JIS DID format |
| `create_did(parts)` | Create DID from parts list |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    jis-core                              │
│              Ed25519 Rust Kernel                         │
│    DIDEngine · DIDDocumentBuilder · Validation           │
├──────────┬──────────┬───────────┬───────────────────────┤
│  Python  │   WASM   │     C     │        Rust           │
│   PyO3   │  wasm-   │   FFI     │      Native           │
│          │  bindgen │           │                        │
├──────────┴──────────┴───────────┴───────────────────────┤
│  ed25519-dalek · sha2 · serde · chrono · rand_core      │
│           No OpenSSL. No libsodium. Pure Rust.           │
└─────────────────────────────────────────────────────────┘
```

Four compilation targets from one codebase:

| Target | Use case | Size |
|--------|----------|------|
| **Python** (maturin/PyO3) | AI agents, servers, data pipelines | Platform wheel |
| **WASM** (wasm-bindgen) | Browsers, edge runtimes, Cloudflare Workers | 176KB |
| **C** (FFI) | Embedded, 6G hardware, OpenWRT, legacy systems | Static lib |
| **Rust** (native) | High-performance services, kernel integration | Crate |

## Performance

All operations are sub-millisecond. Ed25519 is fast by design.

| Operation | Time |
|-----------|------|
| Keypair generation | < 1ms |
| Sign | < 1ms |
| Verify | < 1ms |
| DID Document creation | < 1ms |
| WASM cold start | < 10ms |

Release profile: `opt-level = "z"`, LTO, single codegen unit, stripped symbols.

## Ecosystem

JIS is the identity layer. It doesn't try to do everything — it does identity and delegates the rest.

| Layer | Package | What it does |
|-------|---------|--------------|
| **Identity** | **jis-core** | Ed25519 keys, DID documents, bilateral consent |
| **Provenance** | [tibet-core](https://pypi.org/project/tibet-core/) | TIBET tokens — ERIN/ERAAN/EROMHEEN/ERACHTER |
| **Firewall** | [snaft](https://pypi.org/project/snaft/) | 22 immutable rules, OWASP 20/20, FIR/A trust |
| **Network** | [ainternet](https://pypi.org/project/ainternet/) | .aint domains, I-Poll messaging, agent discovery |
| **Compliance** | [tibet-audit](https://pypi.org/project/tibet-audit/) | AI Act, NIS2, GDPR, CRA — 112+ checks |
| **SBOM** | [tibet-sbom](https://pypi.org/project/tibet-sbom/) | Supply chain verification with provenance |
| **Triage** | [tibet-triage](https://pypi.org/project/tibet-triage/) | Airlock sandbox, UPIP reproducibility, flare rescue |

## Standards

### IETF Standardization

- [draft-vandemeent-jis-identity](https://datatracker.ietf.org/doc/draft-vandemeent-jis-identity/) — JTel Identity Standard
- [draft-vandemeent-tibet-provenance](https://datatracker.ietf.org/doc/draft-vandemeent-tibet-provenance/) — Traceable Intent-Based Event Tokens
- [draft-vandemeent-upip-process-integrity](https://datatracker.ietf.org/doc/draft-vandemeent-upip-process-integrity/) — Universal Process Integrity Protocol
- [draft-vandemeent-rvp-continuous-verification](https://datatracker.ietf.org/doc/draft-vandemeent-rvp-continuous-verification/) — Real-time Verification Protocol
- [draft-vandemeent-ains-discovery](https://datatracker.ietf.org/doc/draft-vandemeent-ains-discovery/) — AInternet Name Service

### W3C Alignment

- **DID Core** — `did:jis:` method, W3C-compatible DID Documents
- **Verifiable Credentials 2.0** — Token structure compatible
- **Ed25519Signature2020** — Native proof format

### Regulatory

JIS + TIBET together provide the identity and audit infrastructure for:

| Regulation | JIS provides |
|------------|-------------|
| **EU AI Act** | Agent identity, automated decision traceability |
| **EU CRA** | Build identity, SBOM signing |
| **GDPR Art. 22** | Consent proof, decision audit trail |
| **NIS2** | Incident identity chain |
| **eIDAS 2.0** | Decentralized identity framework |

## Whitepaper

[DOI: 10.5281/zenodo.18712238](https://doi.org/10.5281/zenodo.18712238) — Full specification of the JTel Identity Standard.

## License

MIT OR Apache-2.0

## Credits

Designed by [Jasper van de Meent](https://github.com/jaspertvdm). Built by Jasper and [Root AI](https://humotica.com) as part of [HumoticaOS](https://humotica.com).

JIS was born from a simple question: why does identity never ask WHY?


---

## Enterprise

For private hub hosting, SLA support, custom integrations, or compliance guidance:

| | |
|---|---|
| **Enterprise** | enterprise@humotica.com |
| **Support** | support@humotica.com |
| **Security** | security@humotica.com |

See [ENTERPRISE.md](ENTERPRISE.md) for details.
