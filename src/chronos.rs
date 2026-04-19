//! CHRONOS - The AETHER Time Capsule System
//!
//! Secure, time-gated storage for digital legacy and sensitive future-intents.
//! "Data that outlives its creator."

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::tibet::{TibetFactory, TibetToken};

/// Conditions for unlocking a Time Capsule
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnlockCondition {
    /// Fixed point in time (Epoch)
    Timestamp(DateTime<Utc>),
    /// Absence of a specific IDD's heartbeat for a duration
    HeartbeatTimeout {
        idd_name: String,
        timeout_days: u32,
    },
    /// Manual release by a high-trust quorum
    MultiSigQuorum {
        required_vouchers: u32,
    },
}

/// A Chronos Time Capsule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeCapsule {
    pub id: Uuid,
    /// What is hidden inside (encrypted payload)
    pub encrypted_payload: String,
    /// When/How it opens
    pub condition: UnlockCondition,
    /// Who created it
    pub creator: String,
    /// TIBET token for provenance
    pub tibet_token: String,
    /// Current state
    pub state: CapsuleState,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CapsuleState {
    Locked,
    AwaitingConditions,
    Released,
    Expired,
}

pub struct ChronosManager {
    tibet: TibetFactory,
}

impl ChronosManager {
    pub fn new() -> Self {
        Self {
            tibet: TibetFactory::new("chronos-warden"),
        }
    }

    /// Create a new time-locked capsule
    pub fn seal(
        &self,
        payload: &str,
        condition: UnlockCondition,
        creator: &str,
    ) -> TimeCapsule {
        // In a real implementation, we would encrypt the payload here.
        // For the PoC, we "hex-encode" it to simulate encryption.
        let encrypted_payload = hex::encode(payload);

        let token = self.tibet.action(
            "capsule_sealed",
            creator,
            serde_json::json!({
                "condition": condition,
                "payload_hash": "SHA256_MOCK_HASH",
            }),
        );

        TimeCapsule {
            id: Uuid::new_v4(),
            encrypted_payload,
            condition,
            creator: creator.to_string(),
            tibet_token: token.id,
            state: CapsuleState::Locked,
            created_at: Utc::now(),
        }
    }

    /// Check if a capsule can be opened
    pub fn attempt_unlock(&self, capsule: &mut TimeCapsule, current_heartbeat: Option<DateTime<Utc>>) -> Result<String, String> {
        let now = Utc::now();

        let can_unlock = match &capsule.condition {
            UnlockCondition::Timestamp(target_date) => {
                now >= *target_date
            },
            UnlockCondition::HeartbeatTimeout { timeout_days, .. } => {
                if let Some(last_beat) = current_heartbeat {
                    let elapsed = now.signed_duration_since(last_beat).num_days();
                    elapsed >= (*timeout_days as i64)
                } else {
                    false
                }
            },
            UnlockCondition::MultiSigQuorum { .. } => {
                // Multi-sig logic would go here
                false
            }
        };

        if can_unlock {
            capsule.state = CapsuleState::Released;
            // "Decrypt"
            let decrypted = String::from_utf8(hex::decode(&capsule.encrypted_payload).unwrap())
                .map_err(|e| e.to_string())?;
            Ok(decrypted)
        } else {
            Err("Conditions not met. The Vault remains silent.".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_time_lock() {
        let chronos = ChronosManager::new();
        
        // 1. Create a capsule locked 1 hour in the future
        let future = Utc::now() + Duration::hours(1);
        let mut capsule = chronos.seal(
            "Jasper's Secret Legacy",
            UnlockCondition::Timestamp(future),
            "jasper"
        );

        assert_eq!(capsule.state, CapsuleState::Locked);

        // 2. Attempt unlock (should fail)
        let result = chronos.attempt_unlock(&mut capsule, None);
        assert!(result.is_err());
        println!("Expected failure: {}", result.unwrap_err());

        // 3. Create a capsule locked in the past (already openable)
        let past = Utc::now() - Duration::hours(1);
        let mut capsule_past = chronos.seal(
            "Hello from the past",
            UnlockCondition::Timestamp(past),
            "jasper"
        );

        let result_past = chronos.attempt_unlock(&mut capsule_past, None);
        assert!(result_past.is_ok());
        assert_eq!(result_past.unwrap(), "Hello from the past");
    }
}
