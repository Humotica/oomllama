//! IDD Spaces - Personal spaces for each IDD in AETHER
//!
//! Every IDD gets their own space - inbox, outbox, state.
//! Messages flow through verified lanes only.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

use crate::tibet::{TibetFactory, TibetToken};

/// Maximum messages in inbox before oldest are dropped
const MAX_INBOX_SIZE: usize = 1000;

/// A message in AETHER
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message ID
    pub id: Uuid,
    /// Sender IDD name
    pub from: String,
    /// Recipient IDD name
    pub to: String,
    /// Message type
    pub msg_type: MessageType,
    /// The actual content
    pub content: serde_json::Value,
    /// Lane this message was sent through
    pub lane_id: Uuid,
    /// TIBET token for provenance
    pub tibet_token_id: Option<String>,
    /// When the message was sent
    pub sent_at: DateTime<Utc>,
    /// When it was delivered (None if not yet)
    pub delivered_at: Option<DateTime<Utc>>,
    /// Read status
    pub read: bool,
}

impl Message {
    /// Create a new message
    pub fn new(from: impl Into<String>, to: impl Into<String>, content: serde_json::Value, lane_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            from: from.into(),
            to: to.into(),
            msg_type: MessageType::Text,
            content,
            lane_id,
            tibet_token_id: None,
            sent_at: Utc::now(),
            delivered_at: None,
            read: false,
        }
    }

    /// Set message type
    pub fn with_type(mut self, msg_type: MessageType) -> Self {
        self.msg_type = msg_type;
        self
    }

    /// Attach TIBET token
    pub fn with_tibet_token(mut self, token_id: String) -> Self {
        self.tibet_token_id = Some(token_id);
        self
    }
}

/// Types of messages in AETHER
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    /// Plain text message
    Text,
    /// Query/question
    Query,
    /// Response to a query
    Response,
    /// Task request
    Task,
    /// Task result
    TaskResult,
    /// System notification
    System,
    /// Heartbeat/ping
    Heartbeat,
    /// Broadcast to all connected
    Broadcast,
}

/// An IDD's personal space in AETHER
#[derive(Debug)]
pub struct IDDSpace {
    /// The IDD's unique ID
    pub idd_id: Uuid,
    /// The IDD's name
    pub name: String,
    /// Incoming messages
    inbox: VecDeque<Message>,
    /// Outgoing messages (pending delivery)
    outbox: VecDeque<Message>,
    /// Custom state storage
    state: DashMap<String, serde_json::Value>,
    /// When this space was created
    pub created_at: DateTime<Utc>,
    /// Last activity
    pub last_active: DateTime<Utc>,
    /// Total messages received
    pub messages_received: u64,
    /// Total messages sent
    pub messages_sent: u64,
}

impl IDDSpace {
    /// Create a new IDD space
    pub fn new(idd_id: Uuid, name: impl Into<String>) -> Self {
        Self {
            idd_id,
            name: name.into(),
            inbox: VecDeque::with_capacity(100),
            outbox: VecDeque::with_capacity(100),
            state: DashMap::new(),
            created_at: Utc::now(),
            last_active: Utc::now(),
            messages_received: 0,
            messages_sent: 0,
        }
    }

    /// Receive a message into inbox
    pub fn receive(&mut self, mut message: Message) {
        message.delivered_at = Some(Utc::now());

        // Enforce max inbox size
        while self.inbox.len() >= MAX_INBOX_SIZE {
            self.inbox.pop_front();
        }

        self.inbox.push_back(message);
        self.messages_received += 1;
        self.last_active = Utc::now();
    }

    /// Queue a message for sending
    pub fn send(&mut self, message: Message) {
        self.outbox.push_back(message);
        self.messages_sent += 1;
        self.last_active = Utc::now();
    }

    /// Get unread messages
    pub fn get_unread(&self) -> Vec<&Message> {
        self.inbox.iter().filter(|m| !m.read).collect()
    }

    /// Get all messages in inbox
    pub fn get_inbox(&self) -> Vec<&Message> {
        self.inbox.iter().collect()
    }

    /// Mark a message as read
    pub fn mark_read(&mut self, message_id: Uuid) -> bool {
        for msg in self.inbox.iter_mut() {
            if msg.id == message_id {
                msg.read = true;
                return true;
            }
        }
        false
    }

    /// Pop next outgoing message
    pub fn pop_outbox(&mut self) -> Option<Message> {
        self.outbox.pop_front()
    }

    /// Peek at outbox
    pub fn peek_outbox(&self) -> Option<&Message> {
        self.outbox.front()
    }

    /// Set a state value
    pub fn set_state(&self, key: impl Into<String>, value: serde_json::Value) {
        self.state.insert(key.into(), value);
    }

    /// Get a state value
    pub fn get_state(&self, key: &str) -> Option<serde_json::Value> {
        self.state.get(key).map(|v| v.clone())
    }

    /// Get space statistics
    pub fn stats(&self) -> SpaceStats {
        SpaceStats {
            idd_id: self.idd_id,
            name: self.name.clone(),
            inbox_count: self.inbox.len(),
            unread_count: self.inbox.iter().filter(|m| !m.read).count(),
            outbox_count: self.outbox.len(),
            state_keys: self.state.len(),
            messages_received: self.messages_received,
            messages_sent: self.messages_sent,
            created_at: self.created_at,
            last_active: self.last_active,
        }
    }
}

/// Statistics for an IDD space
#[derive(Debug, Clone, Serialize)]
pub struct SpaceStats {
    pub idd_id: Uuid,
    pub name: String,
    pub inbox_count: usize,
    pub unread_count: usize,
    pub outbox_count: usize,
    pub state_keys: usize,
    pub messages_received: u64,
    pub messages_sent: u64,
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
}

/// Registry of all IDD spaces
pub struct SpaceRegistry {
    /// Spaces by IDD ID
    spaces: DashMap<Uuid, IDDSpace>,
    /// Name to ID mapping for quick lookup
    name_index: DashMap<String, Uuid>,
    /// TIBET factory for message tokens
    tibet: TibetFactory,
}

impl SpaceRegistry {
    /// Create a new space registry
    pub fn new() -> Self {
        Self {
            spaces: DashMap::new(),
            name_index: DashMap::new(),
            tibet: TibetFactory::new("space-registry"),
        }
    }

    /// Create or get a space for an IDD
    pub fn get_or_create(&self, idd_id: Uuid, name: &str) -> bool {
        if self.spaces.contains_key(&idd_id) {
            return false; // Already exists
        }

        let space = IDDSpace::new(idd_id, name);
        self.name_index.insert(name.to_string(), idd_id);
        self.spaces.insert(idd_id, space);
        true
    }

    /// Get a space by IDD ID
    pub fn get(&self, idd_id: Uuid) -> Option<dashmap::mapref::one::Ref<'_, Uuid, IDDSpace>> {
        self.spaces.get(&idd_id)
    }

    /// Get a mutable space by IDD ID
    pub fn get_mut(&self, idd_id: Uuid) -> Option<dashmap::mapref::one::RefMut<'_, Uuid, IDDSpace>> {
        self.spaces.get_mut(&idd_id)
    }

    /// Find space by name
    pub fn find_by_name(&self, name: &str) -> Option<Uuid> {
        self.name_index.get(name).map(|r| *r)
    }

    /// Deliver a message through a lane
    pub fn deliver(&self, message: Message) -> Result<TibetToken, String> {
        // Find recipient space
        let recipient_id = self.find_by_name(&message.to)
            .ok_or_else(|| format!("Recipient '{}' not found", message.to))?;

        // Get mutable reference to recipient space
        let mut space = self.spaces.get_mut(&recipient_id)
            .ok_or_else(|| "Space not found".to_string())?;

        // Create TIBET token for the message
        let token = self.tibet.action(
            "message_delivered",
            &message.to,
            serde_json::json!({
                "message_id": message.id.to_string(),
                "from": message.from,
                "msg_type": message.msg_type,
                "lane_id": message.lane_id.to_string(),
            }),
        );

        // Attach token and deliver
        let mut msg = message;
        msg.tibet_token_id = Some(token.id.clone());
        space.receive(msg);

        Ok(token)
    }

    /// Get all space statistics
    pub fn all_stats(&self) -> Vec<SpaceStats> {
        self.spaces.iter().map(|entry| entry.stats()).collect()
    }

    /// Total spaces
    pub fn count(&self) -> usize {
        self.spaces.len()
    }
}

impl Default for SpaceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_creation() {
        let registry = SpaceRegistry::new();
        let id = Uuid::new_v4();

        assert!(registry.get_or_create(id, "test_ai"));
        assert!(!registry.get_or_create(id, "test_ai")); // Already exists

        let space = registry.get(id).unwrap();
        assert_eq!(space.name, "test_ai");
    }

    #[test]
    fn test_message_delivery() {
        let registry = SpaceRegistry::new();
        let sender_id = Uuid::new_v4();
        let recipient_id = Uuid::new_v4();
        let lane_id = Uuid::new_v4();

        registry.get_or_create(sender_id, "sender");
        registry.get_or_create(recipient_id, "recipient");

        let message = Message::new("sender", "recipient", serde_json::json!({"text": "Hello!"}), lane_id);

        let result = registry.deliver(message);
        assert!(result.is_ok());

        // Check recipient received it
        let space = registry.get(recipient_id).unwrap();
        assert_eq!(space.messages_received, 1);
        assert_eq!(space.get_unread().len(), 1);
    }

    #[test]
    fn test_inbox_limit() {
        let mut space = IDDSpace::new(Uuid::new_v4(), "test");
        let lane_id = Uuid::new_v4();

        // Fill inbox beyond limit
        for i in 0..MAX_INBOX_SIZE + 100 {
            let msg = Message::new("sender", "test", serde_json::json!({"num": i}), lane_id);
            space.receive(msg);
        }

        // Should be capped at MAX_INBOX_SIZE
        assert_eq!(space.inbox.len(), MAX_INBOX_SIZE);
    }
}
