//! Signed batch reports for CHIMERA-SCANNER.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::tibet::{TibetFactory, TibetToken};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItemReport {
    pub path: String,
    pub hash: String,
    pub changed: bool,
    pub size_bytes: u64,
    pub event_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchReport {
    pub batch_id: String,
    pub created_at: DateTime<Utc>,
    pub total_items: usize,
    pub changed_items: usize,
    pub chunks_total: usize,
    pub items: Vec<BatchItemReport>,
    pub tibet_token: Option<TibetToken>,
}

impl BatchReport {
    pub fn new(items: Vec<BatchItemReport>, chunks_total: usize) -> Self {
        let changed_items = items.iter().filter(|i| i.changed).count();
        Self {
            batch_id: format!("CHIM-{}", Uuid::new_v4().to_string().replace("-", "")[..16].to_uppercase()),
            created_at: Utc::now(),
            total_items: items.len(),
            changed_items,
            chunks_total,
            items,
            tibet_token: None,
        }
    }

    pub fn sign(&mut self, tibet: &TibetFactory) {
        let token = tibet.action(
            "BatchReport",
            "CHIMERA-SCANNER",
            serde_json::json!({
                "batch_id": self.batch_id,
                "total_items": self.total_items,
                "changed_items": self.changed_items,
                "chunks_total": self.chunks_total,
            }),
        );
        self.tibet_token = Some(token);
    }
}
