//! TIBET - Token-based Immutable Block for Ethical Transactions
//!
//! Every action in AETHER generates a TIBET token for audit.
//! The Dutch provenance model: ERIN, ERAAN, EROMHEEN, ERACHTER.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// TIBET Token - immutable audit record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TibetToken {
    /// Unique token ID
    pub id: String,
    /// Token type
    pub token_type: TokenType,
    /// Actor who created this token
    pub actor: String,
    /// Current state in lifecycle
    pub state: TokenState,

    // === ERIN (What's IN) ===
    /// The content/payload of this action
    pub erin: serde_json::Value,

    // === ERAAN (What's attached) ===
    /// Dependencies and references
    pub eraan: Vec<String>,

    // === EROMHEEN (Context around) ===
    /// Environmental context
    pub eromheen: serde_json::Value,

    // === ERACHTER (Intent behind) ===
    /// Why this action was taken
    pub erachter: String,

    /// Hardware anchor ID (ASP)
    pub node_hwid: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Parent token ID (for chains)
    pub parent_id: Option<String>,
    /// Hash of the token content
    pub hash: String,
    /// Signature (if signed)
    pub signature: Option<String>,
}

impl TibetToken {
    /// Create a new TIBET token
    pub fn new(
        token_type: TokenType,
        actor: impl Into<String>,
        erin: serde_json::Value,
        erachter: impl Into<String>,
    ) -> Self {
        let mut token = Self {
            id: format!("TIB-{}", Uuid::new_v4().to_string().replace("-", "")[..16].to_uppercase()),
            token_type,
            actor: actor.into(),
            state: TokenState::Created,
            erin,
            eraan: Vec::new(),
            eromheen: serde_json::Value::Null,
            erachter: erachter.into(),
            node_hwid: None,
            created_at: Utc::now(),
            parent_id: None,
            hash: String::new(),
            signature: None,
        };
        token.hash = token.compute_hash();
        token
    }

    /// Builder: set hardware anchor (ASP)
    pub fn with_hwid(mut self, hwid: impl Into<String>) -> Self {
        self.node_hwid = Some(hwid.into());
        self.hash = self.compute_hash();
        self
    }

    /// Builder: add dependency
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.eraan.push(dep.into());
        self.hash = self.compute_hash();
        self
    }

    /// Builder: set context
    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.eromheen = context;
        self.hash = self.compute_hash();
        self
    }

    /// Builder: set parent
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self.hash = self.compute_hash();
        self
    }

    /// Compute hash of token content
    fn compute_hash(&self) -> String {
        let content = format!(
            "{}:{}:{}:{}:{}:{:?}:{:?}",
            self.id,
            self.actor,
            self.erin,
            self.erachter,
            self.created_at.timestamp(),
            self.node_hwid,
            self.eraan
        );
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Verify the token hash
    pub fn verify_hash(&self) -> bool {
        self.hash == self.compute_hash()
    }

    /// Transition to a new state
    pub fn transition(&mut self, new_state: TokenState) -> bool {
        // Valid transitions
        let valid = matches!(
            (&self.state, &new_state),
            (TokenState::Created, TokenState::Detected)
                | (TokenState::Detected, TokenState::Classified)
                | (TokenState::Classified, TokenState::Mitigated)
                | (TokenState::Mitigated, TokenState::Resolved)
                | (TokenState::Created, TokenState::Resolved) // Fast path for simple operations
        );

        if valid {
            self.state = new_state;
            self.hash = self.compute_hash();
        }
        valid
    }
}

/// Types of TIBET tokens
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    /// An action was performed
    Action,
    /// A decision was made
    Decision,
    /// A message was sent
    Message,
    /// A workflow step
    Workflow,
    /// An audit event
    Audit,
    /// A trust change
    Trust,
    /// A lane operation
    Lane,
    /// Intent verification
    Intent,
}

/// Token lifecycle states
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TokenState {
    Created,
    Detected,
    Classified,
    Mitigated,
    Resolved,
}

/// TIBET token factory
#[derive(Clone)]
pub struct TibetFactory {
    /// Default actor for tokens
    default_actor: String,
    /// Hardware anchor ID (ASP)
    node_hwid: Option<String>,
}

impl TibetFactory {
    pub fn new(actor: impl Into<String>) -> Self {
        Self {
            default_actor: actor.into(),
            node_hwid: None,
        }
    }

    /// Set the hardware anchor for this factory
    pub fn with_hwid(mut self, hwid: impl Into<String>) -> Self {
        self.node_hwid = Some(hwid.into());
        self
    }

    /// Create an intent verification token
    pub fn intent_verified(&self, intent_id: Uuid, passed: bool, reason: &str) -> TibetToken {
        let mut token = TibetToken::new(
            TokenType::Intent,
            &self.default_actor,
            serde_json::json!({
                "intent_id": intent_id.to_string(),
                "passed": passed,
            }),
            format!("Intent verification: {}", reason),
        );
        if let Some(ref hwid) = self.node_hwid {
            token = token.with_hwid(hwid);
        }
        token
    }

    /// Create a lane opened token
    pub fn lane_opened(&self, lane_id: Uuid, from: &str, to: &str) -> TibetToken {
        let mut token = TibetToken::new(
            TokenType::Lane,
            &self.default_actor,
            serde_json::json!({
                "lane_id": lane_id.to_string(),
                "from": from,
                "to": to,
            }),
            format!("Lane opened: {} -> {}", from, to),
        );
        if let Some(ref hwid) = self.node_hwid {
            token = token.with_hwid(hwid);
        }
        token
    }

    /// Create a trust change token
    pub fn trust_changed(&self, idd_id: Uuid, old_trust: f64, new_trust: f64, reason: &str) -> TibetToken {
        let mut token = TibetToken::new(
            TokenType::Trust,
            &self.default_actor,
            serde_json::json!({
                "idd_id": idd_id.to_string(),
                "old_trust": old_trust,
                "new_trust": new_trust,
            }),
            format!("Trust change: {}", reason),
        );
        if let Some(ref hwid) = self.node_hwid {
            token = token.with_hwid(hwid);
        }
        token
    }

    /// Create an action token
    pub fn action(&self, action: &str, target: &str, payload: serde_json::Value) -> TibetToken {
        let mut token = TibetToken::new(
            TokenType::Action,
            &self.default_actor,
            serde_json::json!({
                "action": action,
                "target": target,
                "payload": payload,
            }),
            format!("Action: {} on {}", action, target),
        );
        if let Some(ref hwid) = self.node_hwid {
            token = token.with_hwid(hwid);
        }
        token
    }
}

impl Default for TibetFactory {
    fn default() -> Self {
        Self::new("oomllama")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = TibetToken::new(
            TokenType::Action,
            "test_actor",
            serde_json::json!({"test": true}),
            "Test reason",
        );

        assert!(token.id.starts_with("TIB-"));
        assert!(token.verify_hash());
    }

    #[test]
    fn test_token_transition() {
        let mut token = TibetToken::new(
            TokenType::Workflow,
            "test",
            serde_json::Value::Null,
            "test",
        );

        assert!(token.transition(TokenState::Detected));
        assert!(token.transition(TokenState::Classified));
        assert!(!token.transition(TokenState::Created)); // Invalid transition
    }
}
