//! Bridge Layer — External crate integration
//!
//! Connects OomLlama's internal modules to the real ecosystem crates:
//! - `tibet-trust-kernel`: Voorproever/Bus/Archivaris pipeline, SessionKey, Bifurcation
//! - `tibet-store-mmu`: userfaultfd MmuArena for lazy model loading
//! - `jis-core`: Ed25519 identity, JISClearance, parse_jis()
//!
//! The bridge does NOT replace internal modules — it augments them
//! with the production crypto and identity primitives.

pub mod trust_kernel;
pub mod tibet;
pub mod jis;
