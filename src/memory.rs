//! Chimera Conversation Memory - SQLite Persistent
//!
//! Stores conversation context for multi-turn dialogues.
//! Persists to SQLite so conversations survive restarts.
//!
//! One love, one fAmIly!

use chrono::{DateTime, Duration, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

const DB_PATH: &str = "/srv/jtel-stack/data/chimera_memory.db";

/// A single message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,  // "user" or "assistant"
    pub content: String,
    pub provider: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// A conversation with history
#[derive(Debug, Clone, Serialize)]
pub struct Conversation {
    pub id: Uuid,
    pub idd_name: String,
    pub messages: Vec<ConversationMessage>,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

/// Conversation memory store with SQLite persistence
#[derive(Clone)]
pub struct ConversationMemory {
    conn: Arc<Mutex<Connection>>,
}

impl ConversationMemory {
    pub fn new() -> Self {
        // Ensure data directory exists
        let db_path = Path::new(DB_PATH);
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let conn = Connection::open(DB_PATH)
            .expect("Failed to open SQLite database");

        // Create tables
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                idd_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_activity TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                provider TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_conv_idd ON conversations(idd_name);
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);"
        ).expect("Failed to create tables");

        tracing::info!("📦 Conversation memory loaded from {}", DB_PATH);

        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    /// Get or create a conversation for an IDD
    /// Returns (conversation_id, is_new)
    pub fn get_or_create(&self, idd_name: &str, conversation_id: Option<Uuid>) -> Uuid {
        let conn = self.conn.lock();

        // If conversation_id provided, check if it exists
        if let Some(id) = conversation_id {
            let exists: bool = conn.query_row(
                "SELECT 1 FROM conversations WHERE id = ?",
                [id.to_string()],
                |_| Ok(true)
            ).unwrap_or(false);

            if exists {
                // Update last_activity
                conn.execute(
                    "UPDATE conversations SET last_activity = ? WHERE id = ?",
                    params![Utc::now().to_rfc3339(), id.to_string()]
                ).ok();
                return id;
            }
            // If ID doesn't exist, log warning and create new
            tracing::warn!("⚠️ Conversation {} not found, creating new", id);
        }

        // Create new conversation
        let new_id = Uuid::new_v4();
        let now = Utc::now();

        conn.execute(
            "INSERT INTO conversations (id, idd_name, created_at, last_activity) VALUES (?, ?, ?, ?)",
            params![new_id.to_string(), idd_name, now.to_rfc3339(), now.to_rfc3339()]
        ).expect("Failed to insert conversation");

        // Cleanup old conversations (keep max 10 per IDD, remove expired)
        self.cleanup_idd_internal(&conn, idd_name);

        new_id
    }

    /// Add a message to a conversation
    pub fn add_message(
        &self,
        conversation_id: Uuid,
        role: &str,
        content: &str,
        provider: Option<&str>,
    ) {
        let conn = self.conn.lock();
        let now = Utc::now();

        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, provider, timestamp) VALUES (?, ?, ?, ?, ?)",
            params![conversation_id.to_string(), role, content, provider, now.to_rfc3339()]
        ).ok();

        // Update conversation last_activity
        conn.execute(
            "UPDATE conversations SET last_activity = ? WHERE id = ?",
            params![now.to_rfc3339(), conversation_id.to_string()]
        ).ok();
    }

    /// Get context for a conversation (last N messages)
    pub fn get_context(&self, conversation_id: Uuid, max_messages: usize) -> Option<String> {
        self.get_context_internal(conversation_id, max_messages, false)
    }

    /// Get context with only user messages (for provider switches)
    /// This prevents AI identity confusion when switching from @claude to @gemini
    pub fn get_context_user_only(&self, conversation_id: Uuid, max_messages: usize) -> Option<String> {
        self.get_context_internal(conversation_id, max_messages, true)
    }

    /// Internal context builder
    /// When provider_switch is true, AI responses show which AI said it (e.g., "Claude said:")
    fn get_context_internal(&self, conversation_id: Uuid, max_messages: usize, provider_switch: bool) -> Option<String> {
        let conn = self.conn.lock();

        // Always get role, content, and provider
        let mut stmt = conn.prepare(
            "SELECT role, content, provider FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?"
        ).ok()?;

        let messages: Vec<(String, String, Option<String>)> = stmt.query_map(
            params![conversation_id.to_string(), max_messages as i64],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2).ok()))
        ).ok()?.filter_map(|r| r.ok()).collect();

        if messages.is_empty() {
            return None;
        }

        // Reverse to get chronological order
        let header = if provider_switch {
            "Previous conversation (note: you are a DIFFERENT AI than previous responses):\n"
        } else {
            "Previous conversation:\n"
        };
        let mut context = String::from(header);
        for (role, content, provider) in messages.into_iter().rev() {
            let label = if role == "user" {
                "User".to_string()
            } else if provider_switch {
                // Show which AI responded (so new AI knows it's different)
                match provider.as_deref() {
                    Some("claude") => "Claude (other AI)".to_string(),
                    Some("gemini") => "Gemini (other AI)".to_string(),
                    Some("ollama") => "OomLlama (other AI)".to_string(),
                    Some(p) => format!("{} (other AI)", p),
                    None => "Other AI".to_string(),
                }
            } else {
                "Assistant".to_string()
            };
            context.push_str(&format!("{}: {}\n", label, content));
        }
        context.push_str("\nCurrent question:\n");

        Some(context)
    }

    /// Get full conversation details
    pub fn get_conversation(&self, conversation_id: Uuid) -> Option<Conversation> {
        let conn = self.conn.lock();

        let conv: Option<(String, String, String)> = conn.query_row(
            "SELECT idd_name, created_at, last_activity FROM conversations WHERE id = ?",
            [conversation_id.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        ).ok();

        let (idd_name, created_at_str, last_activity_str) = conv?;

        let mut stmt = conn.prepare(
            "SELECT role, content, provider, timestamp FROM messages WHERE conversation_id = ? ORDER BY id"
        ).ok()?;

        let messages: Vec<ConversationMessage> = stmt.query_map(
            [conversation_id.to_string()],
            |row| {
                let ts_str: String = row.get(3)?;
                Ok(ConversationMessage {
                    role: row.get(0)?,
                    content: row.get(1)?,
                    provider: row.get(2).ok(),
                    timestamp: DateTime::parse_from_rfc3339(&ts_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            }
        ).ok()?.filter_map(|r| r.ok()).collect();

        Some(Conversation {
            id: conversation_id,
            idd_name,
            messages,
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            last_activity: DateTime::parse_from_rfc3339(&last_activity_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }

    /// Cleanup old conversations for an IDD
    fn cleanup_idd_internal(&self, conn: &Connection, idd_name: &str) {
        // Remove conversations older than 24 hours with no activity
        let cutoff = (Utc::now() - Duration::hours(24)).to_rfc3339();

        // Delete old messages first (foreign key)
        conn.execute(
            "DELETE FROM messages WHERE conversation_id IN (
                SELECT id FROM conversations WHERE idd_name = ? AND last_activity < ?
            )",
            params![idd_name, cutoff]
        ).ok();

        // Delete old conversations
        conn.execute(
            "DELETE FROM conversations WHERE idd_name = ? AND last_activity < ?",
            params![idd_name, cutoff]
        ).ok();

        // Keep only latest 10 per IDD
        conn.execute(
            "DELETE FROM messages WHERE conversation_id IN (
                SELECT id FROM conversations WHERE idd_name = ?
                ORDER BY last_activity DESC LIMIT -1 OFFSET 10
            )",
            params![idd_name]
        ).ok();

        conn.execute(
            "DELETE FROM conversations WHERE idd_name = ? AND id NOT IN (
                SELECT id FROM conversations WHERE idd_name = ?
                ORDER BY last_activity DESC LIMIT 10
            )",
            params![idd_name, idd_name]
        ).ok();
    }

    /// List conversations for an IDD
    pub fn list_conversations(&self, idd_name: &str) -> Vec<ConversationSummary> {
        let conn = self.conn.lock();

        let mut stmt = match conn.prepare(
            "SELECT c.id, c.created_at, c.last_activity, COUNT(m.id) as msg_count
             FROM conversations c
             LEFT JOIN messages m ON m.conversation_id = c.id
             WHERE c.idd_name = ?
             GROUP BY c.id
             ORDER BY c.last_activity DESC"
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };

        stmt.query_map([idd_name], |row| {
            let id_str: String = row.get(0)?;
            let created_str: String = row.get(1)?;
            let last_str: String = row.get(2)?;
            let count: i64 = row.get(3)?;

            Ok(ConversationSummary {
                id: Uuid::parse_str(&id_str).unwrap_or_else(|_| Uuid::nil()),
                message_count: count as usize,
                created_at: DateTime::parse_from_rfc3339(&created_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                last_activity: DateTime::parse_from_rfc3339(&last_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        }).ok()
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        let conn = self.conn.lock();

        let total_conversations: i64 = conn.query_row(
            "SELECT COUNT(*) FROM conversations", [], |row| row.get(0)
        ).unwrap_or(0);

        let total_messages: i64 = conn.query_row(
            "SELECT COUNT(*) FROM messages", [], |row| row.get(0)
        ).unwrap_or(0);

        let active_idds: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT idd_name) FROM conversations", [], |row| row.get(0)
        ).unwrap_or(0);

        MemoryStats {
            total_conversations: total_conversations as usize,
            total_messages: total_messages as usize,
            active_idds: active_idds as usize,
        }
    }
}

impl Default for ConversationMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of a conversation
#[derive(Debug, Clone, Serialize)]
pub struct ConversationSummary {
    pub id: Uuid,
    pub message_count: usize,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

/// Memory statistics
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub total_conversations: usize,
    pub total_messages: usize,
    pub active_idds: usize,
}
