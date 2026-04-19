//! Sovereign Pipeline — The complete inference flow
//!
//! This is the top-level entry point for OomLlama v1.0:
//! identity in, verified output + TIBET token out.
//!
//! ```text
//! SovereignPipeline::infer(prompt, identity)
//!   → Airlock (SNAFT + JIS)
//!   → GhostLlama inference (with layer access limits)
//!   → TIBET provenance token
//!   → SovereignOutput { text, tibet_token, latency, layers_used }
//! ```

use std::sync::Arc;
use std::time::Instant;

use crate::bridge::jis::{self, Clearance, LayerPolicy};
use crate::bridge::tibet::{InferenceProvenance, DeviceContext, LayerDistribution};
use crate::pipeline::airlock::{PipelineAirlock, AirlockVerdict};
use crate::tibet::TibetToken;
use tibet_trust_kernel::bifurcation::JisClaim;

/// Output of a sovereign inference call
#[derive(Debug)]
pub struct SovereignOutput {
    /// The generated text
    pub text: String,
    /// TIBET provenance token for this inference
    pub tibet_token: TibetToken,
    /// End-to-end latency in milliseconds
    pub latency_ms: f64,
    /// How many layers were accessible to this caller
    pub layers_accessible: u32,
    /// Total layers in the model
    pub total_layers: u32,
    /// The caller's .aint identity
    pub caller_aint: String,
}

/// Error types for sovereign inference
#[derive(Debug)]
pub enum SovereignError {
    /// SNAFT killed the prompt
    SnaftKill { reason: String },
    /// Caller lacks clearance
    ClearanceDenied { required: Clearance, provided: Clearance },
    /// Inference engine error
    InferenceError { reason: String },
    /// Identity resolution failed
    IdentityError { reason: String },
}

impl std::fmt::Display for SovereignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SnaftKill { reason } => write!(f, "SNAFT KILL: {}", reason),
            Self::ClearanceDenied { required, provided } =>
                write!(f, "Clearance denied: need {:?}, have {:?}", required, provided),
            Self::InferenceError { reason } => write!(f, "Inference error: {}", reason),
            Self::IdentityError { reason } => write!(f, "Identity error: {}", reason),
        }
    }
}

impl std::error::Error for SovereignError {}

/// The Sovereign Inference Pipeline
///
/// Wraps GhostLlama with identity gating, SNAFT security,
/// and TIBET provenance. This is the production entry point.
pub struct SovereignPipeline {
    /// Two-phase security gate
    airlock: PipelineAirlock,
    /// TIBET provenance minter
    provenance: InferenceProvenance,
    /// Model name for logging
    model_name: String,
    /// Number of layers in the model
    total_layers: u32,
}

impl SovereignPipeline {
    /// Create a new sovereign pipeline
    ///
    /// - `model_name`: Name of the loaded model (e.g., "humotica-32b")
    /// - `total_layers`: Number of transformer layers
    /// - `device_context`: Hardware info for TIBET EROMHEEN
    pub fn new(
        model_name: &str,
        total_layers: u32,
        device_context: DeviceContext,
    ) -> Self {
        Self {
            airlock: PipelineAirlock::new(total_layers),
            provenance: InferenceProvenance::new(model_name, device_context),
            model_name: model_name.to_string(),
            total_layers,
        }
    }

    /// Create pipeline with specific security profile
    pub fn with_profile(
        model_name: &str,
        total_layers: u32,
        device_context: DeviceContext,
        security_profile: &str,
    ) -> Self {
        Self {
            airlock: PipelineAirlock::with_profile(total_layers, security_profile),
            provenance: InferenceProvenance::new(model_name, device_context),
            model_name: model_name.to_string(),
            total_layers,
        }
    }

    /// Run sovereign inference
    ///
    /// This is the main entry point. It:
    /// 1. Resolves the caller's identity (jis:, did:jis:, or .aint)
    /// 2. Runs SNAFT validation on the prompt
    /// 3. Checks JIS clearance against layer policy
    /// 4. Calls the inference function
    /// 5. Mints a TIBET provenance token
    ///
    /// The `infer_fn` closure receives (prompt, max_layers) and returns the generated text.
    /// This decouples the pipeline from GhostLlama's internals.
    pub fn infer<F>(
        &mut self,
        prompt: &str,
        caller_identity: &str,
        caller_clearance: Clearance,
        max_tokens: usize,
        infer_fn: F,
    ) -> Result<SovereignOutput, SovereignError>
    where
        F: FnOnce(&str, u32) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
    {
        let t0 = Instant::now();

        // Step 1: Resolve identity
        let identity = jis::resolve_identity(caller_identity)
            .ok_or_else(|| SovereignError::IdentityError {
                reason: format!("Cannot resolve identity: {}", caller_identity),
            })?;

        let from_aint = identity.aint_domain
            .as_deref()
            .unwrap_or(&identity.id);

        // Step 2: Create JIS claim
        let claim = jis::inference_claim(caller_identity, caller_clearance.clone());

        // Step 3: Airlock check (SNAFT + JIS)
        let max_layer = match self.airlock.check(
            prompt,
            from_aint,
            &self.to_trust_claim(&claim),
            &caller_clearance,
        ) {
            AirlockVerdict::Pass { max_layer, .. } => max_layer,
            AirlockVerdict::SnaftKill { reason } => {
                return Err(SovereignError::SnaftKill { reason });
            }
            AirlockVerdict::ClearanceDenied { required, provided } => {
                return Err(SovereignError::ClearanceDenied { required, provided });
            }
        };

        // Step 4: Inference (with layer limit)
        let text = infer_fn(prompt, max_layer)
            .map_err(|e| SovereignError::InferenceError {
                reason: e.to_string(),
            })?;

        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Step 5: Mint TIBET provenance token
        let tibet_token = self.provenance.mint_inference_token(
            prompt,
            from_aint,
            &caller_clearance.to_string(),
            text.split_whitespace().count(), // approximate token count
            latency_ms,
            0, // cache hits — will be filled by Spaceshuttle in Fase 4
        );

        Ok(SovereignOutput {
            text,
            tibet_token,
            latency_ms,
            layers_accessible: max_layer + 1,
            total_layers: self.total_layers,
            caller_aint: from_aint.to_string(),
        })
    }

    /// Convert jis-core JISClaim to trust-kernel JisClaim
    fn to_trust_claim(&self, claim: &jis_core::JISClaim) -> JisClaim {
        JisClaim {
            identity: claim.identity.clone(),
            ed25519_pub: claim.ed25519_pub.clone(),
            clearance: LayerPolicy::to_trust_clearance(&claim.clearance),
            role: claim.role.clone(),
            dept: claim.dept.clone(),
            claimed_at: claim.claimed_at.clone(),
            signature: claim.signature.clone(),
        }
    }
}

impl std::fmt::Display for SovereignOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[Sovereign] {} | {}/{} layers | {:.1}ms | TIBET: {}",
            self.caller_aint,
            self.layers_accessible,
            self.total_layers,
            self.latency_ms,
            self.tibet_token.id,
        )
    }
}
