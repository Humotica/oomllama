//! Pipeline — Sovereign Inference Pipeline
//!
//! The full OomLlama v1.0 pipeline:
//!
//! ```text
//! Prompt + JIS Identity
//!   → Airlock: Voorproever SNAFT check (2.6µs kill)
//!   → Clearance: JIS layer access policy
//!   → Inference: GhostLlama / TurboEngine
//!   → Provenance: TIBET token (ERIN/ERAAN/EROMHEEN/ERACHTER)
//!   → Output + audit trail
//! ```
//!
//! Nobody else does this: identity-gated inference with cryptographic
//! provenance on every single call. Not a wrapper — built into the engine.

pub mod airlock;
pub mod sovereign;
