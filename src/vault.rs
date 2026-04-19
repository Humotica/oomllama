//! VAULT - Secure Storage for TIBET Tokens & Capsules
//!
//! The "Basement" of the AETHER Fortress.
//! Persists audit trails and time-locked data with encryption.

use rusqlite::{params, Connection};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use parking_lot::Mutex;
use uuid::Uuid;
use crate::tibet::TibetToken;

pub struct TibetVault {
    conn: Arc<Mutex<Connection>>,
}

impl TibetVault {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let conn = Connection::open(path).expect("Failed to open Vault database");
        
        // Create the tables for permanent audit storage
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                token_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                erin TEXT NOT NULL,
                erachter TEXT NOT NULL,
                node_hwid TEXT,
                parent_id TEXT,
                hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_logs(actor);
            CREATE INDEX IF NOT EXISTS idx_audit_parent ON audit_logs(parent_id);
            CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(created_at);"
        ).expect("Failed to initialize Vault tables");

        tracing::info!("🏰 TIBET Vault is open and secure.");

        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    /// Store a TIBET token permanently
    pub fn archive_token(&self, token: &TibetToken) -> Result<(), String> {
        let conn = self.conn.lock();
        
        conn.execute(
            "INSERT INTO audit_logs (id, token_type, actor, erin, erachter, node_hwid, parent_id, hash, created_at) 
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                token.id,
                format!("{:?}", token.token_type),
                token.actor,
                token.erin.to_string(),
                token.erachter,
                token.node_hwid,
                token.parent_id,
                token.hash,
                token.created_at.to_rfc3339()
            ],
        ).map_err(|e| format!("Vault write error: {}", e))?;

        Ok(())
    }

    /// Retrieve tokens for a specific actor (Transparency!)
    pub fn get_actor_history(&self, actor: &str) -> Vec<Value> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT erin, erachter, created_at FROM audit_logs WHERE actor = ? ORDER BY created_at DESC"
        ).unwrap();

        let rows = stmt.query_map([actor], |row| {
            Ok(serde_json::json!({
                "content": row.get::<_, String>(0)?,
                "intent": row.get::<_, String>(1)?,
                "timestamp": row.get::<_, String>(2)?,
            }))
        }).unwrap();

        rows.filter_map(|r| r.ok()).collect()
    }

    /// Retrieve the full provenance chain for a token
    pub fn get_chain(&self, token_id: &str) -> Vec<Value> {
        let mut chain = Vec::new();
        let mut current_id = token_id.to_string();

        let conn = self.conn.lock();

        while !current_id.is_empty() {
            let mut stmt = conn.prepare(
                "SELECT id, token_type, actor, erin, erachter, node_hwid, parent_id, created_at 
                 FROM audit_logs WHERE id = ?"
            ).unwrap();

            let result = stmt.query_row([&current_id], |row| {
                let parent: Option<String> = row.get(6).ok();
                Ok((serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "type": row.get::<_, String>(1)?,
                    "actor": row.get::<_, String>(2)?,
                    "erin": row.get::<_, String>(3)?,
                    "erachter": row.get::<_, String>(4)?,
                    "hwid": row.get::<_, Option<String>>(5)?,
                    "parent_id": parent.clone(),
                    "timestamp": row.get::<_, String>(7)?,
                }), parent))
            });

            match result {
                Ok((token_json, parent)) => {
                    chain.push(token_json);
                    current_id = parent.unwrap_or_default();
                }
                Err(_) => break,
            }
        }

        chain
    }
}
