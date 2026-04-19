//! Transport — Distributed inference over QUIC MUX + RAM-RAID
//!
//! Two components:
//!
//! ## RAM-RAID
//! Distributed model memory across machines. 70B always fits:
//! ```text
//! P520:  even layers (10GB) — local GPU
//! DL360: odd layers  (10GB) — remote via Genesis Highway
//! ```
//! Transparent via userfaultfd: page fault on odd layer → network fetch → inject.
//!
//! ## QUIC MUX (planned)
//! Token-by-token streaming over QUIC streams:
//! - No head-of-line blocking (unlike TCP)
//! - WiFi→5G handoff: stream continues
//! - Parallel users on separate streams
//!
//! Currently uses TCP MUX from trust-kernel (Phase 1).
//! QUIC upgrade planned when quinn is integrated (Phase 2).

pub mod ram_raid;
pub mod quic_mux;
