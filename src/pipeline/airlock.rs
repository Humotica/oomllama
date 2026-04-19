//! Airlock — SNAFT + JIS gate before inference
//!
//! Two-phase security gate:
//! 1. **SNAFT phase**: Voorproever scans prompt for injection, returns PASS/KILL
//! 2. **JIS phase**: Check if caller's clearance covers the model's layer range
//!
//! If either phase fails, the prompt never reaches the model.
//! The Airlock is stateless per request — crash-safe by design.

use crate::bridge::trust_kernel::{InferenceAirlock, ArchivarisResult};
use crate::bridge::jis::{LayerPolicy, Clearance};
use tibet_trust_kernel::bifurcation::JisClaim;

/// Result of the airlock check
#[derive(Debug)]
pub enum AirlockVerdict {
    /// Prompt is safe and caller has sufficient clearance
    Pass {
        /// Max accessible layer for this caller
        max_layer: u32,
        /// The archivaris result with TIBET token
        archivaris_result: ArchivarisResult,
    },
    /// SNAFT killed the prompt (injection detected)
    SnaftKill {
        reason: String,
    },
    /// Caller lacks clearance for this model
    ClearanceDenied {
        required: Clearance,
        provided: Clearance,
    },
}

/// The two-phase airlock gate
pub struct PipelineAirlock {
    airlock: InferenceAirlock,
    layer_policy: LayerPolicy,
}

impl PipelineAirlock {
    /// Create airlock for a model with N layers
    pub fn new(total_layers: u32) -> Self {
        Self {
            airlock: InferenceAirlock::new(),
            layer_policy: LayerPolicy::default_for_layers(total_layers),
        }
    }

    /// Create airlock with specific security profile
    pub fn with_profile(total_layers: u32, profile: &str) -> Self {
        Self {
            airlock: InferenceAirlock::with_profile(profile),
            layer_policy: LayerPolicy::default_for_layers(total_layers),
        }
    }

    /// Run a prompt through both SNAFT and JIS gates
    pub fn check(
        &mut self,
        prompt: &str,
        from_aint: &str,
        claim: &JisClaim,
        caller_clearance: &Clearance,
    ) -> AirlockVerdict {
        // Phase 1: SNAFT — is the prompt safe?
        let archivaris_result = match self.airlock.validate_prompt(prompt, from_aint, claim) {
            Some(result) => result,
            None => {
                return AirlockVerdict::SnaftKill {
                    reason: format!("SNAFT rejected prompt from {}", from_aint),
                };
            }
        };

        // Phase 2: JIS — does the caller have clearance?
        // Find the highest layer this caller can access
        let mut max_layer = 0u32;
        for layer in 0..self.layer_policy.total_layers {
            if self.layer_policy.can_access_layer(caller_clearance, layer) {
                max_layer = layer;
            } else {
                break;
            }
        }

        // Check if caller can access at least the base model
        if !self.layer_policy.can_access_layer(caller_clearance, 0) {
            return AirlockVerdict::ClearanceDenied {
                required: self.layer_policy.required_clearance(0),
                provided: caller_clearance.clone(),
            };
        }

        AirlockVerdict::Pass {
            max_layer,
            archivaris_result,
        }
    }

    /// Get the layer policy (for inspection / debugging)
    pub fn policy(&self) -> &LayerPolicy {
        &self.layer_policy
    }
}
