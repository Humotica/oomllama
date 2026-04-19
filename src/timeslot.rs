//! Timeslot - Time-gated access in AETHER
//!
//! Time is a first-class citizen in JIS. Access can be granted for specific
//! windows, ensuring that intents are only valid when they should be.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A time window during which an action is permitted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeslot {
    /// Unique identifier
    pub id: Uuid,
    /// When the slot starts
    pub start: DateTime<Utc>,
    /// When the slot ends
    pub end: DateTime<Utc>,
    /// Human-readable description
    pub description: Option<String>,
    /// Whether the slot is recurring
    pub recurring: Option<RecurrencePattern>,
    /// Associated booking ID (from ai-teams)
    pub booking_id: Option<String>,
}

impl Timeslot {
    /// Create a new timeslot
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self {
            id: Uuid::new_v4(),
            start,
            end,
            description: None,
            recurring: None,
            booking_id: None,
        }
    }

    /// Create a timeslot starting now with given duration
    pub fn now_for(duration: Duration) -> Self {
        let start = Utc::now();
        let end = start + duration;
        Self::new(start, end)
    }

    /// Create a timeslot for a specific number of minutes
    pub fn minutes(minutes: i64) -> Self {
        Self::now_for(Duration::minutes(minutes))
    }

    /// Builder: set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Builder: set booking ID
    pub fn with_booking(mut self, booking_id: impl Into<String>) -> Self {
        self.booking_id = Some(booking_id.into());
        self
    }

    /// Builder: set recurrence
    pub fn with_recurrence(mut self, pattern: RecurrencePattern) -> Self {
        self.recurring = Some(pattern);
        self
    }

    /// Check if the timeslot is currently active
    pub fn is_active(&self) -> bool {
        let now = Utc::now();
        now >= self.start && now <= self.end
    }

    /// Check if the timeslot is in the future
    pub fn is_future(&self) -> bool {
        Utc::now() < self.start
    }

    /// Check if the timeslot has passed
    pub fn is_past(&self) -> bool {
        Utc::now() > self.end
    }

    /// Get duration in minutes
    pub fn duration_minutes(&self) -> i64 {
        (self.end - self.start).num_minutes()
    }

    /// Get remaining time if active
    pub fn remaining(&self) -> Option<Duration> {
        if self.is_active() {
            Some(self.end - Utc::now())
        } else {
            None
        }
    }

    /// Check if a given timestamp is within this slot
    pub fn contains(&self, timestamp: DateTime<Utc>) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }
}

/// Recurrence patterns for timeslots
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrencePattern {
    /// Every day at the same time
    Daily,
    /// Every week on the same day
    Weekly,
    /// Every month on the same date
    Monthly,
    /// Custom cron expression
    Cron(String),
}

/// Time-gated access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeGate {
    /// The timeslot for this gate
    pub slot: Timeslot,
    /// Who can access during this slot
    pub allowed_actors: Vec<Uuid>,
    /// What actions are allowed
    pub allowed_actions: Vec<String>,
    /// Whitelisted hardware nodes (ASP)
    pub allowed_nodes: Option<Vec<String>>,
    /// Maximum concurrent uses
    pub max_concurrent: Option<u32>,
    /// Current usage count
    pub current_usage: u32,
}

impl TimeGate {
    /// Create a new time gate
    pub fn new(slot: Timeslot) -> Self {
        Self {
            slot,
            allowed_actors: Vec::new(),
            allowed_actions: Vec::new(),
            allowed_nodes: None,
            max_concurrent: None,
            current_usage: 0,
        }
    }

    /// Builder: set allowed nodes
    pub fn with_nodes(mut self, nodes: Vec<String>) -> Self {
        self.allowed_nodes = Some(nodes);
        self
    }

    /// Check if an actor can pass through this gate
    pub fn can_pass(&self, actor_id: Uuid, action: &str, current_node_hwid: Option<&str>) -> bool {
        // Check if slot is active
        if !self.slot.is_active() {
            return false;
        }

        // Check hardware restrictions (ASP)
        if let Some(ref nodes) = self.allowed_nodes {
            match current_node_hwid {
                Some(hwid) => {
                    if !nodes.contains(&hwid.to_string()) {
                        // Hardware not in whitelist
                        return false;
                    }
                }
                None => {
                    // Restriction exists but no HWID provided -> Block secure gate
                    return false;
                }
            }
        }

        // Check if actor is allowed (empty = all allowed)
        if !self.allowed_actors.is_empty() && !self.allowed_actors.contains(&actor_id) {
            return false;
        }

        // Check if action is allowed (empty = all allowed)
        if !self.allowed_actions.is_empty() && !self.allowed_actions.iter().any(|a| a == action) {
            return false;
        }

        // Check concurrent usage
        if let Some(max) = self.max_concurrent {
            if self.current_usage >= max {
                return false;
            }
        }

        true
    }

    /// Enter the gate (increment usage)
    pub fn enter(&mut self) -> bool {
        if let Some(max) = self.max_concurrent {
            if self.current_usage >= max {
                return false;
            }
        }
        self.current_usage += 1;
        true
    }

    /// Exit the gate (decrement usage)
    pub fn exit(&mut self) {
        if self.current_usage > 0 {
            self.current_usage -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeslot_active() {
        let slot = Timeslot::minutes(30);
        assert!(slot.is_active());
        assert!(!slot.is_past());
        assert!(!slot.is_future());
    }

    #[test]
    fn test_timeslot_duration() {
        let slot = Timeslot::minutes(15);
        assert_eq!(slot.duration_minutes(), 15);
    }
}
