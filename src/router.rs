//! JIS Router - The Borrow Checker for Identity
//!
//! The router verifies intents and opens lanes in AETHER.
//! If an intent passes JIS, it WILL execute safely.

use chrono::Utc;
use dashmap::DashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{JisError, JisResult};
use crate::intent::{Action, Intent, IntentCategory, IntentVerification, VerificationFailure};
use crate::machtig::{Constraint, Machtiging};
use crate::tibet::TibetFactory;
use crate::timeslot::TimeGate;
use crate::trust::TrustRegistry;
use crate::types::{IDD, Lane, LaneState, LaneType};

/// JIS Router - the heart of AETHER routing
#[derive(Clone)]
pub struct JisRouter {
    /// Trust registry
    trust: Arc<TrustRegistry>,
    /// Active lanes
    lanes: DashMap<Uuid, Lane>,
    /// Registered IDDs
    idds: DashMap<Uuid, IDD>,
    /// Registered Machtigingen (Authorities)
    machtigingen: DashMap<Uuid, Machtiging>,
    /// Time gates
    time_gates: DashMap<Uuid, TimeGate>,
    /// TIBET token factory
    tibet: TibetFactory,
    /// Pending verifications
    pending: DashMap<Uuid, IntentVerification>,
}

impl JisRouter {
    /// Create a new JIS Router
    pub fn new() -> Self {
        Self {
            trust: Arc::new(TrustRegistry::new()),
            lanes: DashMap::new(),
            idds: DashMap::new(),
            machtigingen: DashMap::new(),
            time_gates: DashMap::new(),
            tibet: TibetFactory::default(),
            pending: DashMap::new(),
        }
    }

    /// Set the hardware anchor (ASP) for this router
    pub fn with_hwid(mut self, hwid: impl Into<String>) -> Self {
        self.tibet = self.tibet.with_hwid(hwid);
        self
    }

    /// Register an IDD with the router
    pub fn register_idd(&self, idd: IDD, initial_trust: Option<f64>) {
        let id = idd.id;
        self.idds.insert(id, idd);
        if let Some(trust) = initial_trust {
            self.trust.set_trust(id, trust, "Initial registration");
        }
    }

    /// Get an IDD by ID
    pub fn get_idd(&self, id: Uuid) -> Option<IDD> {
        self.idds.get(&id).map(|r| r.clone())
    }

    /// Register a Machtiging (Authority delegation)
    pub fn add_machtiging(&self, machtiging: Machtiging) {
        self.machtigingen.insert(machtiging.id, machtiging);
    }

    /// Get a Machtiging by ID
    pub fn get_machtiging(&self, id: Uuid) -> Option<Machtiging> {
        self.machtigingen.get(&id).map(|r| r.clone())
    }

    /// Verify an intent - the core JIS function
    ///
    /// This is like Rust's borrow checker but for identity and trust.
    /// If verification passes, the intent is safe to execute.
    pub fn verify_intent(&self, intent: &Intent) -> IntentVerification {
        let mut failures = Vec::new();

        // === Check 1: Actor exists ===
        if self.idds.get(&intent.actor.id).is_none() {
            failures.push(VerificationFailure::InvalidActor);
        }

        // === Check 2: Target exists (if specified) ===
        if let Some(ref target) = intent.target {
            if self.idds.get(&target.id).is_none() {
                failures.push(VerificationFailure::InvalidTarget);
            }
        }

        // === Check 3: Trust threshold ===
        let actor_trust = self.trust.get_trust(intent.actor.id).score();
        if actor_trust < intent.min_trust {
            failures.push(VerificationFailure::InsufficientTrust {
                required: intent.min_trust,
                actual: actor_trust,
            });
        }

        // === Check 4: Capabilities ===
        if let Some(ref idd) = self.idds.get(&intent.actor.id) {
            for cap in &intent.required_capabilities {
                if !idd.capabilities.contains(cap) {
                    failures.push(VerificationFailure::MissingCapability(cap.clone()));
                }
            }
        }

        // === Check 5: Timeslot ===
        if !intent.is_within_timeslot() {
            failures.push(VerificationFailure::OutsideTimeslot);
        }

        // === Check 6: Expiration ===
        if !intent.is_valid() {
            failures.push(VerificationFailure::Expired);
        }

        // === Check 7: Signature (if provided) ===
        if intent.signature.is_some() && !intent.verified {
            failures.push(VerificationFailure::InvalidSignature);
        }

        // === Check 8: Authority (Machtiging) ===
        // Certain categories or actions require explicit delegation
        if intent.category == IntentCategory::Resource || intent.category == IntentCategory::Admin {
            let mut authorized = false;
            let action_str = format!("{:?}", intent.action);
            let amount = intent.payload.get("amount").and_then(|v| v.as_f64());
            
            // Check all machtigingen for this actor
            for entry in self.machtigingen.iter() {
                let m = entry.value();
                if m.grantee == intent.actor.name && m.verify(&action_str, amount, "AETHER") {
                    authorized = true;
                    break;
                }
            }

            if !authorized {
                failures.push(VerificationFailure::MachtigingRequired(
                    format!("No valid authority for action {} (category {:?})", action_str, intent.category)
                ));
            }
        }

        // === Create verification result ===
        let passed = failures.is_empty();
        let verification = IntentVerification {
            intent_id: intent.id,
            passed,
            actor_trust,
            failures,
            verified_at: Utc::now(),
            tibet_token_id: None,
        };

        // === Create TIBET token ===
        let reason = if passed { "Verification passed" } else { "Verification failed" };
        let _token = self.tibet.intent_verified(intent.id, passed, reason);

        // Boost trust on successful verification
        if passed {
            self.trust.boost(intent.actor.id, 0.01, "Successful intent verification");
        }

        verification
    }

    /// Route an intent - verify and execute
    pub fn route(&self, intent: Intent) -> JisResult<Lane> {
        // Verify first
        let verification = self.verify_intent(&intent);
        if !verification.passed {
            return Err(JisError::VerificationFailed(verification.failures));
        }

        // Get target (or use actor as self-reference)
        let target = intent.target.clone().unwrap_or_else(|| intent.actor.clone());

        // Check for existing lane
        let lane_key = lane_key(intent.actor.id, target.id);
        if self.lanes.contains_key(&lane_key) {
            return Err(JisError::LaneAlreadyExists {
                from: intent.actor.id,
                to: target.id,
            });
        }

        // Create lane
        let lane = Lane {
            id: Uuid::new_v4(),
            from: intent.actor.clone(),
            to: target,
            lane_type: LaneType::Data, // Default to data lane
            state: LaneState::Active,
            opened_at: Utc::now(),
            expires_at: intent.timeslot.as_ref().map(|s| s.end),
            tibet_token_id: None,
        };

        // Store lane
        self.lanes.insert(lane_key, lane.clone());

        // Create TIBET token for lane opening
        let _token = self.tibet.lane_opened(lane.id, &intent.actor.name, &lane.to.name);

        Ok(lane)
    }

    /// Close a lane
    pub fn close_lane(&self, lane_id: Uuid) -> JisResult<()> {
        let mut found = None;
        for mut entry in self.lanes.iter_mut() {
            if entry.id == lane_id {
                entry.state = LaneState::Closed;
                found = Some(entry.key().clone());
                break;
            }
        }

        match found {
            Some(key) => {
                self.lanes.remove(&key);
                Ok(())
            }
            None => Err(JisError::LaneNotFound(lane_id)),
        }
    }

    /// Get active lanes for an IDD
    pub fn get_lanes(&self, idd_id: Uuid) -> Vec<Lane> {
        self.lanes
            .iter()
            .filter(|entry| entry.from.id == idd_id || entry.to.id == idd_id)
            .filter(|entry| entry.state == LaneState::Active)
            .map(|entry| entry.clone())
            .collect()
    }

    /// Get trust score for an IDD
    pub fn get_trust(&self, idd_id: Uuid) -> f64 {
        self.trust.get_trust(idd_id).score()
    }

    /// Find an IDD by name
    pub fn find_by_name(&self, name: &str) -> Option<IDD> {
        self.idds.iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.clone())
    }

    /// Add a time gate
    pub fn add_time_gate(&self, gate: TimeGate) {
        self.time_gates.insert(gate.slot.id, gate);
    }

    /// Get router statistics
    pub fn stats(&self) -> RouterStats {
        // Memory leak fix: cleanup stale entries on stats call
        self.cleanup_stale();

        RouterStats {
            registered_idds: self.idds.len(),
            active_lanes: self.lanes.iter().filter(|e| e.state == LaneState::Active).count(),
            time_gates: self.time_gates.len(),
            pending_verifications: self.pending.len(),
            active_negotiations: 0, // Placeholder
            active_machtigingen: self.machtigingen.len(),
        }
    }

    /// Memory leak fix: cleanup closed lanes and stale pending verifications
    pub fn cleanup_stale(&self) {
        // Remove closed/sealed lanes (keep only active ones for max 1000 entries)
        let lane_count = self.lanes.len();
        if lane_count > 1000 {
            let mut to_remove: Vec<Uuid> = self.lanes
                .iter()
                .filter(|e| e.state != LaneState::Active)
                .map(|e| *e.key())
                .collect();
            // Remove oldest non-active lanes
            for id in to_remove.drain(..) {
                self.lanes.remove(&id);
            }
        }

        // Remove stale pending verifications (keep max 500)
        let pending_count = self.pending.len();
        if pending_count > 500 {
            let mut to_remove: Vec<Uuid> = Vec::new();
            let mut count = 0;
            for entry in self.pending.iter() {
                count += 1;
                if count > 500 {
                    to_remove.push(*entry.key());
                }
            }
            for id in to_remove {
                self.pending.remove(&id);
            }
        }

        // Remove expired time gates
        let now = Utc::now();
        self.time_gates.retain(|_, gate| gate.slot.end > now);
    }
}

impl Default for JisRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a deterministic key for a lane between two IDDs
fn lane_key(from: Uuid, to: Uuid) -> Uuid {
    // Use XOR to make key symmetric (A->B same as B->A)
    // For directional lanes, use different logic
    let from_bytes = from.as_bytes();
    let to_bytes = to.as_bytes();
    let mut key_bytes = [0u8; 16];
    for i in 0..16 {
        key_bytes[i] = from_bytes[i] ^ to_bytes[i];
    }
    Uuid::from_bytes(key_bytes)
}

/// Router statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct RouterStats {
    pub registered_idds: usize,
    pub active_lanes: usize,
    pub time_gates: usize,
    pub pending_verifications: usize,
    pub active_negotiations: usize,
    pub active_machtigingen: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::Action;

    #[test]
    fn test_idd_registration() {
        let router = JisRouter::new();
        let idd = IDD::new("test_ai").with_domain("test.aint");

        router.register_idd(idd.clone(), Some(0.8));

        assert!(router.get_idd(idd.id).is_some());
        assert!((router.get_trust(idd.id) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_intent_verification() {
        let router = JisRouter::new();
        let actor = IDD::new("test_actor");

        router.register_idd(actor.clone(), Some(0.7));

        let intent = Intent::new(actor, Action::Send, "Test intent")
            .with_min_trust(0.5);

        let verification = router.verify_intent(&intent);
        assert!(verification.passed);
    }

    #[test]
    fn test_insufficient_trust() {
        let router = JisRouter::new();
        let actor = IDD::new("low_trust_actor");

        router.register_idd(actor.clone(), Some(0.2));

        let intent = Intent::new(actor, Action::Send, "High trust intent")
            .with_min_trust(0.8);

        let verification = router.verify_intent(&intent);
        assert!(!verification.passed);
        assert!(verification.failures.iter().any(|f| matches!(f, VerificationFailure::InsufficientTrust { .. })));
    }

    #[test]
    fn test_lane_creation() {
        let router = JisRouter::new();
        let from = IDD::new("from_ai");
        let to = IDD::new("to_ai");

        router.register_idd(from.clone(), Some(0.8));
        router.register_idd(to.clone(), Some(0.8));

        let intent = Intent::new(from.clone(), Action::OpenLane, "Open lane")
            .with_target(to.clone())
            .with_min_trust(0.5);

        let lane = router.route(intent).unwrap();
        assert_eq!(lane.state, LaneState::Active);
        assert_eq!(lane.from.id, from.id);
        assert_eq!(lane.to.id, to.id);
    }
}
