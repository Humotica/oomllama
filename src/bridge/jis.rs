//! JIS Bridge — Identity-gated inference
//!
//! Maps jis-core identity primitives to OomLlama's layer clearance model:
//!
//! ```text
//! Layer  0-31: Unclassified  (base model, anyone)
//! Layer 32-48: Confidential  (enterprise fine-tune)
//! Layer 49-64: Secret        (defense fine-tune)
//! Layer  65+: TopSecret      (sovereign fine-tune)
//! ```
//!
//! Your JIS clearance determines which layers you can decrypt.
//! No clearance = dead material (zero page via Spaceshuttle).

use jis_core::{JISIdentity, JISClearance, JISClaim, parse_jis, DIDEngine};
use tibet_trust_kernel::bifurcation::ClearanceLevel;

/// Layer clearance policy — maps JIS clearance to accessible layer ranges
pub struct LayerPolicy {
    /// Total number of layers in the model
    pub total_layers: u32,
    /// Layer ranges per clearance level (inclusive)
    pub boundaries: Vec<(JISClearance, u32, u32)>,
}

impl LayerPolicy {
    /// Default policy for a model with N layers
    ///
    /// Splits into roughly:
    /// - 50% Unclassified (base model)
    /// - 25% Confidential (enterprise)
    /// - 20% Secret (defense)
    /// - 5% TopSecret (sovereign)
    pub fn default_for_layers(total: u32) -> Self {
        let base = (total as f32 * 0.50) as u32;
        let conf = (total as f32 * 0.75) as u32;
        let secr = (total as f32 * 0.95) as u32;

        Self {
            total_layers: total,
            boundaries: vec![
                (JISClearance::Unclassified, 0, base.saturating_sub(1)),
                (JISClearance::Confidential, base, conf.saturating_sub(1)),
                (JISClearance::Secret, conf, secr.saturating_sub(1)),
                (JISClearance::TopSecret, secr, total.saturating_sub(1)),
            ],
        }
    }

    /// Check if a clearance level can access a specific layer
    pub fn can_access_layer(&self, clearance: &JISClearance, layer: u32) -> bool {
        let clearance_rank = Self::rank(clearance);
        for (boundary_clearance, start, end) in &self.boundaries {
            if layer >= *start && layer <= *end {
                return clearance_rank >= Self::rank(boundary_clearance);
            }
        }
        false
    }

    /// Get the clearance required for a specific layer
    pub fn required_clearance(&self, layer: u32) -> JISClearance {
        for (clearance, start, end) in &self.boundaries {
            if layer >= *start && layer <= *end {
                return clearance.clone();
            }
        }
        JISClearance::TopSecret // Unknown layers require highest clearance
    }

    /// Map JIS clearance to trust-kernel ClearanceLevel
    pub fn to_trust_clearance(clearance: &JISClearance) -> ClearanceLevel {
        match clearance {
            JISClearance::Unclassified => ClearanceLevel::Unclassified,
            JISClearance::Restricted => ClearanceLevel::Restricted,
            JISClearance::Confidential => ClearanceLevel::Confidential,
            JISClearance::Secret => ClearanceLevel::Secret,
            JISClearance::TopSecret => ClearanceLevel::TopSecret,
        }
    }

    fn rank(clearance: &JISClearance) -> u8 {
        match clearance {
            JISClearance::Unclassified => 0,
            JISClearance::Restricted => 1,
            JISClearance::Confidential => 2,
            JISClearance::Secret => 3,
            JISClearance::TopSecret => 4,
        }
    }
}

/// Resolve any identity format to a JISIdentity
///
/// Accepts: "jis:alice", "did:jis:alice", "alice.aint"
pub fn resolve_identity(input: &str) -> Option<JISIdentity> {
    parse_jis(input)
}

/// Create a JIS claim for inference access
pub fn inference_claim(
    identity: &str,
    clearance: JISClearance,
) -> JISClaim {
    let engine = DIDEngine::new();
    let claim_data = format!("inference:{}", identity);
    JISClaim {
        identity: identity.to_string(),
        ed25519_pub: engine.public_key_hex(),
        clearance,
        role: "inference".to_string(),
        dept: "oomllama".to_string(),
        claimed_at: chrono::Utc::now().to_rfc3339(),
        signature: engine.sign(claim_data.as_bytes()),
    }
}

// Re-export core types
pub use jis_core::{
    JISIdentity as Identity,
    JISClearance as Clearance,
    JISClaim as Claim,
    parse_jis as parse,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_policy_64_layers() {
        let policy = LayerPolicy::default_for_layers(64);

        // Base model layers: anyone
        assert!(policy.can_access_layer(&JISClearance::Unclassified, 0));
        assert!(policy.can_access_layer(&JISClearance::Unclassified, 31));

        // Enterprise layers: need Confidential+
        assert!(!policy.can_access_layer(&JISClearance::Unclassified, 32));
        assert!(policy.can_access_layer(&JISClearance::Confidential, 32));

        // Defense layers: need Secret+
        assert!(!policy.can_access_layer(&JISClearance::Confidential, 49));
        assert!(policy.can_access_layer(&JISClearance::Secret, 49));

        // TopSecret has access to everything
        assert!(policy.can_access_layer(&JISClearance::TopSecret, 0));
        assert!(policy.can_access_layer(&JISClearance::TopSecret, 63));
    }

    #[test]
    fn test_resolve_identity_formats() {
        // All three formats should parse
        assert!(resolve_identity("jis:root_idd").is_some());
        assert!(resolve_identity("did:jis:root_idd").is_some());
        assert!(resolve_identity("root_idd.aint").is_some());
    }
}
