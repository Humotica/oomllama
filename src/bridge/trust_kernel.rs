//! Trust Kernel Bridge — Production crypto for OomLlama
//!
//! Wraps `tibet-trust-kernel` types for use in the inference pipeline:
//! - Voorproever: SNAFT validation (2.6µs kill)
//! - Bus: one-way signed payload transport
//! - Archivaris: JIS clearance + TIBET token minting
//! - Bifurcation: AES-256-GCM encrypt/decrypt with SessionKey
//!
//! Session keys: first DH = 76µs, then HKDF = 5µs per block.

use std::sync::Arc;

use tibet_trust_kernel::bifurcation::{
    AirlockBifurcation, BifurcationResult, ClearanceLevel, EncryptedBlock, JisClaim,
};
use tibet_trust_kernel::voorproever::{Voorproever, VoorproeverVerdict};
use tibet_trust_kernel::bus::{BusPayload, VirtualBus};
use tibet_trust_kernel::archivaris::Archivaris;
pub use tibet_trust_kernel::archivaris::ArchivarisResult;
use tibet_trust_kernel::config::TrustKernelConfig;
use tibet_trust_kernel::mux::TibetMuxFrame;
use tibet_trust_kernel::watchdog::Watchdog;

/// The inference airlock — Voorproever → Bus → Archivaris
///
/// Every prompt passes through this pipeline before reaching the model.
/// If SNAFT kills the input, it never touches inference.
pub struct InferenceAirlock {
    voorproever: Voorproever,
    bus: Arc<VirtualBus>,
    archivaris: Archivaris,
}

impl InferenceAirlock {
    /// Create a new airlock with balanced config
    pub fn new() -> Self {
        let config = TrustKernelConfig::balanced();
        let bus = VirtualBus::new(4 * 1024 * 1024); // 4MB bus
        let watchdog = Watchdog::new(
            config.watchdog.timeout_ms,
            config.watchdog.heartbeat_interval_ms,
            config.watchdog.max_missed_heartbeats,
        );
        Self {
            voorproever: Voorproever::new(config.clone(), bus.clone(), watchdog),
            bus,
            archivaris: Archivaris::new(config, VirtualBus::new(4 * 1024 * 1024)),
        }
    }

    /// Create an airlock with specific security profile
    pub fn with_profile(profile: &str) -> Self {
        let config = TrustKernelConfig::from_name(profile);
        let bus = VirtualBus::new(config.bus.max_payload_bytes);
        let watchdog = Watchdog::new(
            config.watchdog.timeout_ms,
            config.watchdog.heartbeat_interval_ms,
            config.watchdog.max_missed_heartbeats,
        );
        Self {
            voorproever: Voorproever::new(config.clone(), bus.clone(), watchdog),
            bus,
            archivaris: Archivaris::new(config, VirtualBus::new(4 * 1024 * 1024)),
        }
    }

    /// Validate a prompt through the full pipeline
    ///
    /// Returns the ArchivarisResult if the prompt passes SNAFT + JIS,
    /// or None if killed/rejected.
    pub fn validate_prompt(
        &mut self,
        prompt: &str,
        from_aint: &str,
        claim: &JisClaim,
    ) -> Option<ArchivarisResult> {
        // Wrap prompt as a MUX frame
        let frame = TibetMuxFrame {
            channel_id: 0,
            intent: "inference".to_string(),
            from_aint: from_aint.to_string(),
            payload: prompt.to_string(),
        };

        // Step 1: Voorproever evaluates the frame
        match self.voorproever.evaluate(&frame) {
            VoorproeverVerdict::Pass { bus_payload, .. } => {
                // Step 2: Archivaris processes the signed payload
                let result = self.archivaris.process(&bus_payload, &frame);
                match result {
                    ArchivarisResult::Success { .. } => Some(result),
                    _ => None,
                }
            }
            VoorproeverVerdict::Kill { .. } | VoorproeverVerdict::Reject { .. } => None,
        }
    }
}

/// Session-key backed encryption for model pages
///
/// First seal = DH key exchange (76µs), subsequent = HKDF (5µs).
/// Used by Spaceshuttle for per-page decrypt during inference.
pub struct SessionCrypto {
    engine: AirlockBifurcation,
}

impl SessionCrypto {
    pub fn new() -> Self {
        Self {
            engine: AirlockBifurcation::new(),
        }
    }

    /// Seal a model page for storage
    pub fn seal_page(
        &mut self,
        data: &[u8],
        page_index: usize,
        clearance: ClearanceLevel,
        source: &str,
    ) -> Option<EncryptedBlock> {
        match self.engine.seal_session(data, page_index, clearance, source) {
            BifurcationResult::Sealed { block, .. } => Some(block),
            _ => None,
        }
    }

    /// Open an encrypted model page
    pub fn open_page(
        &mut self,
        block: &EncryptedBlock,
        claim: &JisClaim,
    ) -> Option<Vec<u8>> {
        match self.engine.open(block, claim) {
            BifurcationResult::Opened { plaintext, .. } => Some(plaintext),
            _ => None,
        }
    }
}

// Re-export commonly used types
pub use tibet_trust_kernel::bifurcation::{
    ClearanceLevel as TrustClearance,
    EncryptedBlock as TrustBlock,
    JisClaim as TrustClaim,
};
// TibetMuxFrame already imported at top of file
