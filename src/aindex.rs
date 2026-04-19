//! AINDEX - The AETHER Knowledge Index
//!
//! Verified storage for Refined data.
//! Integrates Refinery cleaning with Sentinel embeddings.
//!
//! "If it's in the AIndex, it's JIS Approved."

use serde::{Deserialize, Serialize};
use std::fs::{OpenOptions};
use std::io::{Write};
use std::path::PathBuf;
use uuid::Uuid;

use crate::refinery::{RefineResult, PurityLevel};
use crate::sentinel::SentinelClassifier;

/// A record in the AIndex
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIndexRecord {
    pub id: Uuid,
    pub content: String,
    pub vector: Vec<f32>,
    pub purity: PurityLevel,
    pub source: String,
    pub tibet_token: String,
    pub indexed_at: chrono::DateTime<chrono::Utc>,
}

pub struct AIndex {
    storage_path: PathBuf,
}

impl AIndex {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            storage_path: path.into(),
        }
    }

    /// Index a RefineResult using a classifier for embeddings
    pub fn index(
        &self,
        refined: RefineResult,
        source: &str,
        classifier: &SentinelClassifier
    ) -> Result<AIndexRecord, Box<dyn std::error::Error + Send + Sync>> {
        
        // 1. Generate Embedding (using the same engine as Sentinel)
        let vector = classifier.text_to_embedding(&refined.content)?;

        // 2. Create the Record
        let record = AIndexRecord {
            id: refined.id,
            content: refined.content,
            vector,
            purity: refined.purity,
            source: source.to_string(),
            tibet_token: refined.tibet_token.unwrap_or_else(|| "UNSTAMPED".to_string()),
            indexed_at: chrono::Utc::now(),
        };

        // 3. Persist to storage (JSONL for PoC)
        self.persist(&record)?;

        Ok(record)
    }

    fn persist(&self, record: &AIndexRecord) -> std::io::Result<()> {
        let json = serde_json::to_string(record)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.storage_path)?;
        
        writeln!(file, "{}", json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refinery::{Refinery, PurityLevel};
    use crate::sentinel::SentinelClassifier;
    use std::fs;

    #[test]
    fn test_aindex_storage() {
        let test_db = "test_aindex.jsonl";
        let _ = fs::remove_file(test_db); // Clean start
        
        let aindex = AIndex::new(test_db);
        let refinery = Refinery::new();
        let classifier = SentinelClassifier::new("mock").unwrap();

        // 1. Purify
        let refined = refinery.purify("Clean data about Rust programming.", "unit_test");
        
        // 2. Index
        let result = aindex.index(refined, "unit_test", &classifier);
        
        assert!(result.is_ok());
        let record = result.unwrap();
        assert_eq!(record.purity, PurityLevel::Crystal);
        assert!(!record.vector.is_empty());

        // Cleanup
        let _ = fs::remove_file(test_db);
    }
}
