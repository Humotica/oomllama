//! Vector storage format for CHIMERA-SCANNER.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMeta {
    pub path: String,
    pub hash: String,
    pub mtime: i64,
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub meta: VectorMeta,
    pub text: String,
}
