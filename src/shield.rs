//! Liability Shield & Settlement Layer
//!
//! "fail2flag4intent" - Automated consequences for trust violations.
//! Providing a legal and financial shield for businesses using AETHER.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::tibet::{TibetToken, TibetFactory};
use crate::trust::TrustRegistry;

/// Automated settlement action for violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SettlementAction {
    /// Lock the IDD's account/lane
    LockLane(Uuid),
    /// Slash the trust score by a percentage
    SlashTrust(f64),
    /// Log a formal liability claim in TIBET
    LiabilityClaim { amount: f64, reason: String },
    /// Flag for human (Sheriff) review
    FlagForReview(String),
}

pub struct LiabilityShield {
    pub node_id: Uuid,
    pub protection_level: f64, // 0.0 - 1.0
}

impl LiabilityShield {
    pub fn new(node_id: Uuid) -> Self {
        Self {
            node_id,
            protection_level: 1.0, // Default to full protection
        }
    }

    /// Process a violation and determine settlement
    pub fn handle_violation(
        &self, 
        idd_id: Uuid, 
        trust: &TrustRegistry, 
        reason: &str,
        factory: &TibetFactory
    ) -> Vec<SettlementAction> {
        let mut actions = Vec::new();
        let score = trust.get_trust(idd_id).score();

        // fail2flag4intent logic
        if score < 0.5 {
            actions.push(SettlementAction::SlashTrust(0.1));
            actions.push(SettlementAction::FlagForReview(reason.to_string()));
        }

        if score < 0.2 {
            actions.push(SettlementAction::LockLane(idd_id));
        }

        // TIBET Audit of the shield action
        let _token = factory.action(
            "LiabilityShieldTrigger",
            "SHIELD-SYSTEM",
            serde_json::json!({
                "idd": idd_id,
                "score": score,
                "actions": actions,
                "reason": reason,
            }),
        );

        actions
    }
}
