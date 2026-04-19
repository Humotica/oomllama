//! # OomLlama — Sovereign AI Inference Engine
//!
//! The auto where trust-kernel is the motor, tibet-* are the parts, JIS is the key.
//!
//! ## Core Architecture
//!
//! - **OomLlama**: .oom quantization engine (Q2/Q4/Q8), GhostLlama lazy loading
//! - **Trust Kernel**: AES-256-GCM bifurcation, SessionKey (5µs decrypt), Voorproever/Bus/Archivaris
//! - **Spaceshuttle**: userfaultfd → decrypt → decompress → inject (zero-startup model loading)
//! - **JIS Identity**: Ed25519, clearance per layer (PUBLIC→SECRET), parse_jis()
//! - **TIBET Provenance**: Every inference = TIBET token (ERIN/ERAAN/EROMHEEN/ERACHTER)
//! - **QUIC MUX**: Token-by-token streaming, no head-of-line blocking
//! - **RAM-RAID**: Distributed memory — 70B always fits across machines
//!
//! ## The Sovereign Guarantee
//!
//! ```text
//! Your model. Your identity. Your machine. Your inference.
//! Encrypted at rest, verified at runtime, auditable forever.
//! ```
//!
//! One love, one fAmIly! 💙

// Bridge layer — external crate integration (trust-kernel, tibet, jis)
pub mod bridge;

// Pipeline — Sovereign inference flow (Airlock → Inference → Provenance)
pub mod pipeline;

// Spaceshuttle — userfaultfd lazy encrypted model loading
pub mod spaceshuttle;

// Transport — RAM-RAID distributed memory + QUIC MUX streaming
pub mod transport;

pub mod types;
pub mod intent;
pub mod trust;
pub mod timeslot;
pub mod router;
pub mod tibet;
pub mod error;
pub mod space;
pub mod sema;
pub mod betti;
pub mod sentinel;
pub mod gfx;
pub mod snaft;
pub mod memory;
pub mod anchor;
pub mod refinery;
pub mod aindex;
pub mod chronos;
pub mod autonomy;
pub mod vision;
pub mod vault;
pub mod machtig;
pub mod negotiation;
pub mod tasks;
pub mod shield;
pub mod ingest;
pub mod discovery;
pub mod scanner;
pub mod briefing;
pub mod batch;
pub mod report;
pub mod vector;
pub mod kernel;
pub mod embedding;
pub mod oomllama;
pub mod oomllama_turbo;
pub mod quant;
pub mod gguf2oom;
pub mod safetensors2oom;
pub mod multi_gpu;

// Von Braun — Parallel multi-head attention processing
pub mod von_braun;

// tibet-bench — Sovereign inference benchmarking
pub mod tibet_bench;

// turbo_mem — Non-temporal stores, prefetch, madvise (Doom/Quake memory tricks)
pub mod turbo_mem;

pub use types::IDD;
pub use intent::{Intent, Action, IntentVerification};
pub use trust::*;
pub use timeslot::*;
pub use router::*;
pub use tibet::*;
pub use error::*;
pub use space::*;
pub use sema::{SemaRegistry, SemanticAddress, SemaResolution};
pub use betti::{BettiManager, ResourcePool, ResourceType, AllocationRequest, BettiStats, Allocation};
pub use sentinel::{SentinelClassifier, SentinelOutput};
pub use gfx::{GfxMonitor, GfxStatus, GpuInfo};
pub use snaft::{SnaftValidator, ThreatResult, ThreatType, SnaftStats};
pub use memory::{ConversationMemory, Conversation, ConversationMessage, MemoryStats};
pub use anchor::Anchor;
pub use refinery::{Refinery, PurityLevel, RefineResult};
pub use aindex::{AIndex, AIndexRecord};
pub use chronos::{ChronosManager, TimeCapsule, UnlockCondition, CapsuleState};
pub use autonomy::AutonomyDaemon;
pub use vision::{VisionRouter, VisionProvider, VisionPreference};
pub use vault::TibetVault;
pub use machtig::{Machtiging, Constraint};
pub use negotiation::{Offer, Agreement, NegotiationContext};
pub use tasks::{Task, TaskStatus, TaskType};
pub use discovery::{Discovery, DiscoveryRadar, DiscoveryType};
pub use scanner::ChimeraScanner;
pub use briefing::{MorningBriefing, BriefingEngine};
pub use shield::{LiabilityShield, SettlementAction};
pub use ingest::{IngestManager, IngestJob, IngestStatus};
pub use batch::{BatchProcessor, BatchConfig};
pub use report::{BatchReport, BatchItemReport};
pub use vector::{VectorRecord, VectorMeta};
pub use kernel::{SovereignKernel, SovereignIdentity, NeuralCoreInfo};
pub use embedding::EmbeddingEngine;
pub use oomllama::OomLlama;
pub use oomllama_turbo::{TurboEngine, TurboConfig, ModelKVCache, LayerPin, PinStrategy};
pub use multi_gpu::{MultiGPUConfig, MultiGPUManager, LayerStrategy};
pub use gguf2oom::{OomQuantLevel, convert_gguf_to_oom, convert_gguf_to_oom_with_quant};

// Sovereign pipeline exports
pub use pipeline::sovereign::{SovereignPipeline, SovereignOutput, SovereignError};
pub use pipeline::airlock::{PipelineAirlock, AirlockVerdict};
pub use bridge::jis::{LayerPolicy, resolve_identity, inference_claim};
pub use bridge::trust_kernel::{InferenceAirlock, SessionCrypto};
pub use bridge::tibet::{InferenceProvenance, DeviceContext, LayerDistribution};
pub use spaceshuttle::{Spaceshuttle, SpaceshuttleStats, SealedLayerStats};
pub use transport::ram_raid::{ModelRaid, DistributionStrategy, MemoryEstimate};
pub use transport::quic_mux::{StreamMux, InferenceStream, StreamEvent, StreamReceiver};
pub use von_braun::{VonBraunEngine, VonBraunConfig, VonBraunStats};
pub use tibet_bench::{BenchReport, BenchRunner, BenchMeasurement};
pub use turbo_mem::{nt_memcpy, nt_memzero, nt_page_inject, prefetch_read, prefetch_nta, prefetch_page, optimize_arena, ArenaOptResult};


/// OomLlama version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The heart of HumoticaOS - Jasper's heartbeat interval
pub const HEARTBEAT_INTERVAL_MS: u64 = 1000;
// Triggering GPU indexing at za 10 jan 2026 14:15:51 CET
pub mod oom_inference;

// Python bindings (only when feature enabled)
#[cfg(feature = "python")]
pub mod python;
