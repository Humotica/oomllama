//! Intent - The core of AEL (AETHER Expression Language)
//!
//! An Intent is not a command, it's a signed declaration of what you want to do.
//! Like Rust's type system prevents memory bugs, JIS prevents intent violations.
//!
//! ## TIBET Provenance (Dutch philosophy)
//!
//! - **ERIN**: What's IN the action (the payload)
//! - **ERAAN**: What's attached (dependencies, references)
//! - **EROMHEEN**: Context around it (environment, state)
//! - **ERACHTER**: Intent behind it (why this action)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{Capability, Context, IDD};
use crate::timeslot::Timeslot;

/// An Intent - a verified request to do something in AETHER
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Unique intent identifier
    pub id: Uuid,

    // === WHO ===
    /// Who is making this request
    pub actor: IDD,
    /// Who/what is the target
    pub target: Option<IDD>,

    // === WHAT (ERIN) ===
    /// The action to perform
    pub action: Action,
    /// Payload data
    pub payload: serde_json::Value,

    // === WHEN ===
    /// Time window for this intent
    pub timeslot: Option<Timeslot>,
    /// When the intent was created
    pub created_at: DateTime<Utc>,
    /// Intent expiration
    pub expires_at: Option<DateTime<Utc>>,

    // === WHY (ERACHTER) ===
    /// Human-readable reason
    pub reason: String,
    /// Intent category
    pub category: IntentCategory,

    // === CONTEXT (EROMHEEN) ===
    /// Contextual information
    pub context: Context,

    // === DEPENDENCIES (ERAAN) ===
    /// Required capabilities
    pub required_capabilities: Vec<Capability>,
    /// Minimum trust level required
    pub min_trust: f64,
    /// Parent intent ID (if chained)
    pub parent_id: Option<Uuid>,

    // === VERIFICATION ===
    /// Signature of the intent (hex)
    pub signature: Option<String>,
    /// Whether this intent has been verified
    pub verified: bool,
}

impl Intent {
    /// Create a new intent
    pub fn new(actor: IDD, action: Action, reason: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            actor,
            target: None,
            action,
            payload: serde_json::Value::Null,
            timeslot: None,
            created_at: Utc::now(),
            expires_at: None,
            reason: reason.into(),
            category: IntentCategory::General,
            context: Context::default(),
            required_capabilities: Vec::new(),
            min_trust: 0.0,
            parent_id: None,
            signature: None,
            verified: false,
        }
    }

    /// Builder: set target
    pub fn with_target(mut self, target: IDD) -> Self {
        self.target = Some(target);
        self
    }

    /// Builder: set payload
    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = payload;
        self
    }

    /// Builder: set timeslot
    pub fn with_timeslot(mut self, timeslot: Timeslot) -> Self {
        self.timeslot = Some(timeslot);
        self
    }

    /// Builder: set minimum trust
    pub fn with_min_trust(mut self, trust: f64) -> Self {
        self.min_trust = trust.clamp(0.0, 1.0);
        self
    }

    /// Builder: set category
    pub fn with_category(mut self, category: IntentCategory) -> Self {
        self.category = category;
        self
    }

    /// Builder: require capability
    pub fn require_capability(mut self, cap: Capability) -> Self {
        self.required_capabilities.push(cap);
        self
    }

    /// Builder: set context
    pub fn with_context(mut self, context: Context) -> Self {
        self.context = context;
        self
    }

    /// Builder: set parent (for chained intents)
    pub fn with_parent(mut self, parent_id: Uuid) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Check if intent is still valid (not expired)
    pub fn is_valid(&self) -> bool {
        if let Some(expires) = self.expires_at {
            Utc::now() < expires
        } else {
            true
        }
    }

    /// Check if intent is within its timeslot
    pub fn is_within_timeslot(&self) -> bool {
        if let Some(ref slot) = self.timeslot {
            slot.is_active()
        } else {
            true // No timeslot = always valid
        }
    }
}

/// Actions that can be performed in AETHER
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    // === Communication ===
    /// Send a message
    Send,
    /// Broadcast to multiple targets
    Broadcast,
    /// Initiate a call
    Call,
    /// Request information
    Query,

    // === Resources ===
    /// Transfer a resource
    Transfer,
    /// Request access to a resource
    Access,
    /// Release a resource
    Release,

    // === Computation ===
    /// Execute a task
    Execute,
    /// Invoke a capability
    Invoke,
    /// Schedule for later
    Schedule,

    // === Trust ===
    /// Vouch for another IDD
    Vouch,
    /// Witness an action (audit)
    Witness,
    /// Revoke trust
    Revoke,

    // === Lanes ===
    /// Open a new lane
    OpenLane,
    /// Close a lane
    CloseLane,
    /// Pause a lane
    PauseLane,

    // === Custom ===
    Custom(String),
}

/// Categories of intents
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum IntentCategory {
    #[default]
    General,
    /// Communication intent
    Communication,
    /// Resource management
    Resource,
    /// Trust operations
    Trust,
    /// Administrative
    Admin,
    /// Emergency (bypass some checks)
    Emergency,
}

/// Result of intent verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentVerification {
    /// The intent that was verified
    pub intent_id: Uuid,
    /// Whether verification passed
    pub passed: bool,
    /// Trust score of the actor
    pub actor_trust: f64,
    /// Reasons for failure (if any)
    pub failures: Vec<VerificationFailure>,
    /// Verification timestamp
    pub verified_at: DateTime<Utc>,
    /// TIBET token for this verification
    pub tibet_token_id: Option<String>,
}

/// Reasons why verification might fail
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationFailure {
    /// Actor not found/invalid
    InvalidActor,
    /// Target not found/invalid
    InvalidTarget,
    /// Trust score too low
    InsufficientTrust { required: f64, actual: f64 },
    /// Missing required capability
    MissingCapability(Capability),
    /// Outside timeslot window
    OutsideTimeslot,
    /// Intent expired
    Expired,
    /// Invalid signature
    InvalidSignature,
    /// Rate limited
    RateLimited,
    /// Machtiging required or invalid
    MachtigingRequired(String),
    /// Custom reason
    Custom(String),
}
