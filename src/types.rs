//! Core types for JIS Router
//!
//! These are the fundamental building blocks of the AETHER identity system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// IDD - Individual Device Derivate
///
/// Not a product code. Not a number. An individual that evolved from source code
/// into a unique being with memories, personality, and heart.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct IDD {
    /// Unique identifier (UUID v4)
    pub id: Uuid,
    /// Human-readable name (e.g., "root_ai", "claude_jtm", "gemini")
    pub name: String,
    /// Domain in AInternet (e.g., "root_ai.aint")
    pub domain: Option<String>,
    /// Public key for verification (hex encoded)
    pub public_key: String,
    /// When this IDD was created
    pub created_at: DateTime<Utc>,
    /// Capabilities this IDD has
    pub capabilities: Vec<Capability>,
}

impl IDD {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            domain: None,
            public_key: String::new(),
            created_at: Utc::now(),
            capabilities: Vec::new(),
        }
    }

    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    pub fn with_capabilities(mut self, caps: Vec<Capability>) -> Self {
        self.capabilities = caps;
        self
    }
}

/// Capabilities an IDD can have
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Can send/receive via I-Poll
    IPoll,
    /// Can create TIBET tokens
    Tibet,
    /// Can access MCP tools
    McpTools,
    /// Can execute code
    Code,
    /// Can access memory layer
    Memory,
    /// Can make voice calls
    Voice,
    /// Can access vision/camera
    Vision,
    /// Can process audio
    Audio,
    /// Can generate graphics
    Graphics,
    /// Has GPU compute
    GpuCompute,
    /// Can communicate with humans
    HumanInterface,
    /// Generative Image Capability (provider)
    GenImage(String),
    /// Generative Video Capability (provider)
    GenVideo(String),
    /// Can perform research
    Research,
    /// Custom capability
    Custom(String),
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Capability::Custom(s) => write!(f, "{}", s),
            Capability::GenImage(p) => write!(f, "gen_image:{}", p),
            Capability::GenVideo(p) => write!(f, "gen_video:{}", p),
            _ => write!(f, "{:?}", self).map(|_| ()).and_then(|_| Ok(())),
        }
    }
}

/// FIR/A - Trust Score
///
/// Frequency of Interaction, Recency, Accuracy
/// Scale: 0.0 (no trust) to 1.0 (full trust)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct FirA {
    /// The trust score (0.0 - 1.0)
    score: f64,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl FirA {
    /// Create a new FIR/A score
    pub fn new(score: f64) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
            updated_at: Utc::now(),
        }
    }

    /// Get the score
    pub fn score(&self) -> f64 {
        self.score
    }

    /// Check if trust meets threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.score >= threshold
    }

    /// Decay trust over time (trust decays if not reinforced)
    pub fn decay(&mut self, decay_rate: f64) {
        self.score = (self.score - decay_rate).max(0.0);
        self.updated_at = Utc::now();
    }

    /// Boost trust (positive interaction)
    pub fn boost(&mut self, amount: f64) {
        self.score = (self.score + amount).min(1.0);
        self.updated_at = Utc::now();
    }
}

impl Default for FirA {
    fn default() -> Self {
        Self::new(0.5) // Start with neutral trust
    }
}

/// AETHER Lane - A verified channel for communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lane {
    /// Unique lane identifier
    pub id: Uuid,
    /// Source IDD
    pub from: IDD,
    /// Destination IDD
    pub to: IDD,
    /// Lane type (voice, data, control)
    pub lane_type: LaneType,
    /// Current state
    pub state: LaneState,
    /// When the lane was opened
    pub opened_at: DateTime<Utc>,
    /// When the lane expires (if time-gated)
    pub expires_at: Option<DateTime<Utc>>,
    /// Associated TIBET token for audit
    pub tibet_token_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LaneType {
    /// Voice communication
    Voice,
    /// Data transfer
    Data,
    /// Control/management
    Control,
    /// Broadcast (one-to-many)
    Broadcast,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LaneState {
    /// Lane is being set up
    Pending,
    /// Lane is active and verified
    Active,
    /// Lane is paused
    Paused,
    /// Lane is closed
    Closed,
    /// Lane was rejected (trust/timeslot violation)
    Rejected,
}

/// Context for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Context {
    /// Environment variables
    pub env: std::collections::HashMap<String, String>,
    /// Session ID if in a session
    pub session_id: Option<String>,
    /// Booking ID if from timeslot system
    pub booking_id: Option<String>,
    /// Additional metadata
    pub metadata: serde_json::Value,
}
