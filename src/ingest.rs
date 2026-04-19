//! Vault-Gated Ingest & Legacy Adapter
//!
//! Secure, time-gated file parsing based on Intent.
//! Ingesting the "Old World" into AETHER safely.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::intent::Intent;
use crate::tibet::TibetFactory;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestJob {
    pub id: Uuid,
    pub source_url: String,
    pub target_vault: String,
    pub allowed_timeslot: Option<Uuid>,
    pub status: IngestStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IngestStatus {
    Queued,
    Parsing,
    Refining,
    Vaulted,
    Rejected(String),
}

pub struct IngestManager {
    pub refinery_enabled: bool,
}

impl IngestManager {
    pub fn new() -> Self {
        Self { refinery_enabled: true }
    }

    /// Request a secure ingest from an external source
    pub fn request_ingest(&self, intent: &Intent, source: &str, factory: &TibetFactory) -> Result<IngestJob, String> {
        // Validate intent for ingest
        if intent.min_trust < 0.6 {
            return Err("Insufficient trust for secure ingest".to_string());
        }

        let job = IngestJob {
            id: Uuid::new_v4(),
            source_url: source.to_string(),
            target_vault: "MAIN-VAULT".to_string(),
            allowed_timeslot: intent.timeslot.as_ref().map(|s| s.id),
            status: IngestStatus::Queued,
        };

        // Tibet audit of the job start
        factory.action(
            "IngestStarted",
            &intent.actor.name,
            serde_json::to_value(&job).unwrap(),
        );

        Ok(job)
    }
}
