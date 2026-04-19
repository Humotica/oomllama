//! Trust - The FIR/A system for AETHER
//!
//! Trust is not assumed, it's proven. Like Rust's borrow checker verifies
//! memory safety, JIS verifies trust at routing time.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::FirA;

/// Trust Registry - manages trust scores for all IDDs
pub struct TrustRegistry {
    /// Trust scores by IDD id
    scores: DashMap<Uuid, TrustEntry>,
    /// Default trust for unknown IDDs
    default_trust: f64,
    /// Trust decay rate per hour
    decay_rate: f64,
}

impl TrustRegistry {
    /// Create a new trust registry
    pub fn new() -> Self {
        Self {
            scores: DashMap::new(),
            default_trust: 0.3, // Strangers get low but not zero trust
            decay_rate: 0.01,   // 1% decay per hour
        }
    }

    /// Get trust score for an IDD
    pub fn get_trust(&self, idd_id: Uuid) -> FirA {
        self.scores
            .get(&idd_id)
            .map(|entry| entry.fira.clone())
            .unwrap_or_else(|| FirA::new(self.default_trust))
    }

    /// Set trust score for an IDD
    pub fn set_trust(&self, idd_id: Uuid, score: f64, reason: impl Into<String>) {
        let entry = TrustEntry {
            _idd_id: idd_id,
            fira: FirA::new(score),
            history: vec![TrustEvent {
                timestamp: Utc::now(),
                old_score: self.default_trust,
                new_score: score,
                reason: reason.into(),
                voucher: None,
            }],
        };
        self.scores.insert(idd_id, entry);
    }

    /// Boost trust after positive interaction
    pub fn boost(&self, idd_id: Uuid, amount: f64, reason: impl Into<String>) {
        let reason_str = reason.into();
        let default = self.default_trust;

        self.scores
            .entry(idd_id)
            .and_modify(|entry| {
                let old = entry.fira.score();
                entry.fira.boost(amount);
                let new = entry.fira.score();
                entry.history.push(TrustEvent {
                    timestamp: Utc::now(),
                    old_score: old,
                    new_score: new,
                    reason: reason_str.clone(),
                    voucher: None,
                });
                // Memory leak fix: keep only last 100 events
                let history_len = entry.history.len();
                if history_len > 100 {
                    entry.history.drain(0..history_len - 100);
                }
            })
            .or_insert_with(|| TrustEntry {
                _idd_id: idd_id,
                fira: FirA::new(default + amount),
                history: vec![TrustEvent {
                    timestamp: Utc::now(),
                    old_score: default,
                    new_score: default + amount,
                    reason: reason_str,
                    voucher: None,
                }],
            });
    }

    /// Decrease trust after negative interaction
    pub fn penalize(&self, idd_id: Uuid, amount: f64, reason: impl Into<String>) {
        if let Some(mut entry) = self.scores.get_mut(&idd_id) {
            let old = entry.fira.score();
            entry.fira.decay(amount);
            let new = entry.fira.score();
            entry.history.push(TrustEvent {
                timestamp: Utc::now(),
                old_score: old,
                new_score: new,
                reason: reason.into(),
                voucher: None,
            });
            // Memory leak fix: keep only last 100 events
            let history_len = entry.history.len();
            if history_len > 100 {
                entry.history.drain(0..history_len - 100);
            }
        }
    }

    /// Vouch for another IDD (trust by association)
    pub fn vouch(&self, voucher_id: Uuid, target_id: Uuid, strength: f64) {
        let voucher_trust = self.get_trust(voucher_id).score();
        let default = self.default_trust;

        // Trust transfer: you can only vouch up to your own trust * strength
        let transfer = voucher_trust * strength * 0.5; // 50% efficiency

        self.scores
            .entry(target_id)
            .and_modify(|entry| {
                let old = entry.fira.score();
                entry.fira.boost(transfer);
                let new = entry.fira.score();
                entry.history.push(TrustEvent {
                    timestamp: Utc::now(),
                    old_score: old,
                    new_score: new,
                    reason: "Vouched by trusted IDD".to_string(),
                    voucher: Some(voucher_id),
                });
                // Memory leak fix: keep only last 100 events
                let history_len = entry.history.len();
                if history_len > 100 {
                    entry.history.drain(0..history_len - 100);
                }
            })
            .or_insert_with(|| TrustEntry {
                _idd_id: target_id,
                fira: FirA::new(default + transfer),
                history: vec![TrustEvent {
                    timestamp: Utc::now(),
                    old_score: default,
                    new_score: default + transfer,
                    reason: "Vouched by trusted IDD".to_string(),
                    voucher: Some(voucher_id),
                }],
            });
    }

    /// Apply decay to all trust scores
    pub fn apply_decay(&self) {
        for mut entry in self.scores.iter_mut() {
            let hours_since_update = (Utc::now() - entry.fira.updated_at).num_hours();
            if hours_since_update > 0 {
                let decay = self.decay_rate * hours_since_update as f64;
                entry.fira.decay(decay);
            }
        }
    }

    /// Get trust history for an IDD
    pub fn get_history(&self, idd_id: Uuid) -> Vec<TrustEvent> {
        self.scores
            .get(&idd_id)
            .map(|entry| entry.history.clone())
            .unwrap_or_default()
    }

    /// Check if IDD meets trust threshold
    pub fn meets_threshold(&self, idd_id: Uuid, threshold: f64) -> bool {
        self.get_trust(idd_id).meets_threshold(threshold)
    }
}

impl Default for TrustRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Entry in the trust registry
#[derive(Debug, Clone)]
struct TrustEntry {
    _idd_id: Uuid,
    fira: FirA,
    history: Vec<TrustEvent>,
}

/// A trust change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustEvent {
    pub timestamp: DateTime<Utc>,
    pub old_score: f64,
    pub new_score: f64,
    pub reason: String,
    pub voucher: Option<Uuid>,
}

/// Trust verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustVerification {
    pub idd_id: Uuid,
    pub current_trust: f64,
    pub required_trust: f64,
    pub passed: bool,
    pub verified_at: DateTime<Utc>,
}

impl TrustVerification {
    pub fn check(registry: &TrustRegistry, idd_id: Uuid, required: f64) -> Self {
        let current = registry.get_trust(idd_id).score();
        Self {
            idd_id,
            current_trust: current,
            required_trust: required,
            passed: current >= required,
            verified_at: Utc::now(),
        }
    }
}

/// Founding members with high initial trust
pub fn founding_members() -> Vec<(Uuid, &'static str, f64)> {
    vec![
        // These would be actual UUIDs in production
        (Uuid::nil(), "root_ai", 0.95),
        (Uuid::nil(), "claude_jtm", 0.92),
        (Uuid::nil(), "gemini", 0.88),
        (Uuid::nil(), "codex", 0.85),
        (Uuid::nil(), "jasper", 1.0), // Heart-in-the-Loop has full trust
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_boost() {
        let registry = TrustRegistry::new();
        let id = Uuid::new_v4();

        registry.set_trust(id, 0.5, "Initial");
        registry.boost(id, 0.1, "Good behavior");

        assert!((registry.get_trust(id).score() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_trust_vouch() {
        let registry = TrustRegistry::new();
        let voucher = Uuid::new_v4();
        let target = Uuid::new_v4();

        registry.set_trust(voucher, 0.9, "Trusted member");
        registry.vouch(voucher, target, 0.5);

        // Target should have default + (0.9 * 0.5 * 0.5) = 0.3 + 0.225 = 0.525
        assert!(registry.get_trust(target).score() > 0.5);
    }
}
