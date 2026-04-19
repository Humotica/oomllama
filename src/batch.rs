//! Batch processing for CHIMERA-SCANNER.
//!
//! SHA256 diff engine + chunk extraction + GPU embeddings.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::embedding::EmbeddingEngine;
use crate::report::{BatchItemReport, BatchReport};
use crate::vector::{VectorMeta, VectorRecord};

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub db_path: String,
    pub chunk_size: usize,
    pub overlap: usize,
    pub max_bytes: usize,
    pub gpu_index: usize,
    pub enable_embeddings: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            db_path: "data/chimera_batch.db".to_string(),
            chunk_size: 1200,
            overlap: 200,
            max_bytes: 2_000_000,
            gpu_index: 1,           // GPU 1 for embeddings (GPU 0 for inference)
            enable_embeddings: false, // Off by default, enable with --embed flag
        }
    }
}

pub struct BatchProcessor {
    cfg: BatchConfig,
    embedder: Option<Arc<EmbeddingEngine>>,
}

impl BatchProcessor {
    pub fn new(cfg: BatchConfig) -> Self {
        let embedder = if cfg.enable_embeddings {
            match EmbeddingEngine::new(cfg.gpu_index) {
                Ok(engine) => {
                    println!("🧠 EmbeddingEngine initialized on GPU {}", cfg.gpu_index);
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    eprintln!("⚠️ Failed to init EmbeddingEngine: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let processor = Self { cfg, embedder };
        let _ = processor.init_db();
        processor
    }

    /// Create with a shared embedding engine (for daemon mode)
    pub fn with_embedder(cfg: BatchConfig, embedder: Arc<EmbeddingEngine>) -> Self {
        let processor = Self {
            cfg,
            embedder: Some(embedder),
        };
        let _ = processor.init_db();
        processor
    }

    fn init_db(&self) -> rusqlite::Result<()> {
        let conn = Connection::open(&self.cfg.db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS file_hashes (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                mtime INTEGER NOT NULL
            )",
            [],
        )?;
        Ok(())
    }

    pub fn process_path(&self, path: &Path, event_kind: &str) -> io::Result<(BatchReport, Vec<VectorRecord>)> {
        let mut items = Vec::new();
        let mut vectors = Vec::new();

        if path.is_file() {
            if let Some((item, chunks)) = self.handle_file(path, event_kind)? {
                items.push(item);
                vectors.extend(chunks);
            }
        }

        let mut report = BatchReport::new(items, vectors.len());
        report.sign(&crate::tibet::TibetFactory::new("chimera-batch"));
        Ok((report, vectors))
    }

    fn handle_file(&self, path: &Path, event_kind: &str) -> io::Result<Option<(BatchItemReport, Vec<VectorRecord>)>> {
        let bytes = fs::read(path)?;
        let mtime = modified_ts(path)?;
        let hash = sha256_hex(&bytes);

        let changed = self.is_changed(path, &hash, mtime)?;
        if !changed {
            return Ok(None);
        }

        let text = String::from_utf8_lossy(&bytes).to_string();
        let chunks = chunk_text(&text, self.cfg.chunk_size, self.cfg.overlap);
        let total_chunks = chunks.len();
        let mut vectors = Vec::new();
        for (idx, chunk) in chunks.into_iter().enumerate() {
            let summary = summarize(&chunk, 160);

            // Generate embedding if engine is available
            let vector = if let Some(ref embedder) = self.embedder {
                match embedder.embed(&chunk) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("⚠️ Embedding failed for chunk {}: {}", idx, e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new() // No embedder, leave empty
            };

            vectors.push(VectorRecord {
                id: Uuid::new_v4().to_string(),
                vector,
                meta: VectorMeta {
                    path: path.display().to_string(),
                    hash: hash.clone(),
                    mtime,
                    chunk_index: idx,
                    total_chunks,
                    summary,
                },
                text: chunk,
            });
        }

        let item = BatchItemReport {
            path: path.display().to_string(),
            hash,
            changed: true,
            size_bytes: bytes.len() as u64,
            event_kind: event_kind.to_string(),
        };

        Ok(Some((item, vectors)))
    }

    fn is_changed(&self, path: &Path, hash: &str, mtime: i64) -> io::Result<bool> {
        let conn = Connection::open(&self.cfg.db_path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let existing: Option<(String, i64)> = conn
            .query_row(
                "SELECT hash, mtime FROM file_hashes WHERE path = ?1",
                params![path.display().to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        if let Some((prev_hash, prev_mtime)) = existing {
            if prev_hash == hash && prev_mtime == mtime {
                return Ok(false);
            }
        }

        conn.execute(
            "INSERT INTO file_hashes (path, hash, mtime) VALUES (?1, ?2, ?3)\n             ON CONFLICT(path) DO UPDATE SET hash = excluded.hash, mtime = excluded.mtime",
            params![path.display().to_string(), hash, mtime],
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(true)
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        out.push(chunk);
        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    out
}

pub fn summarize(text: &str, max_len: usize) -> String {
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return truncate_ascii(trimmed, max_len);
        }
    }
    truncate_ascii(text.trim(), max_len)
}

fn truncate_ascii(text: &str, max_len: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(max_len) {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            out.push('?');
        }
    }
    out
}

fn modified_ts(path: &Path) -> io::Result<i64> {
    let metadata = fs::metadata(path)?;
    let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let epoch = modified.duration_since(UNIX_EPOCH).unwrap_or_default();
    Ok(epoch.as_secs() as i64)
}

pub fn write_vectors_jsonl(path: &Path, vectors: &[VectorRecord]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buf = String::new();
    for record in vectors {
        let line = serde_json::to_string(record).unwrap_or_default();
        buf.push_str(&line);
        buf.push('\n');
    }
    fs::write(path, buf.as_bytes())
}
