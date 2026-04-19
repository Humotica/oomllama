//! CHIMERA-SCANNER - The All-Seeing Sheriff
//!
//! Real-time file system watcher that logs every change to TIBET
//! and updates KmBiT memory.

use notify::{Watcher, RecursiveMode, Config, Event};
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::time::Duration;
use uuid::Uuid;

use crate::tibet::{TibetFactory, TokenType};
use crate::tasks::{Task, TaskStatus, TaskType};
use crate::embedding::EmbeddingEngine;
use crate::refinery::Refinery;
use crate::aindex::{AIndex, AIndexRecord};
use std::sync::Arc;

pub struct ChimeraScanner {
    path: String,
    tibet: TibetFactory,
    engine: Option<Arc<EmbeddingEngine>>,
    refinery: Arc<Refinery>,
    aindex: Arc<AIndex>,
}

impl ChimeraScanner {
    pub fn new(path: &str, aindex: Arc<AIndex>) -> Self {
        // Initialize engine on GPU 1
        let engine = EmbeddingEngine::new(1).ok().map(Arc::new);
        
        Self {
            path: path.to_string(),
            tibet: TibetFactory::new("chimera-scanner"),
            engine,
            refinery: Arc::new(Refinery::new()),
            aindex,
        }
    }

    /// Start the real-time watcher
    pub fn start(&self) -> notify::Result<()> {
        // 1. Initial Scan (Unleash the Tiger!)
        let _ = self.initial_scan();

        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::RecommendedWatcher::new(tx, Config::default())?;
        watcher.watch(Path::new(&self.path), RecursiveMode::Recursive)?;

        println!("📡 CHIMERA-SCANNER active on: {}", self.path);

        for res in rx {
            match res {
                Ok(event) => self.handle_event(event),
                Err(e) => println!("watch error: {:?}", e),
            }
        }

        Ok(())
    }

    fn initial_scan(&self) -> std::io::Result<()> {
        use walkdir::WalkDir;
        
        println!("🐅 Initial scan started...");
        let mut count = 0;

        for entry in WalkDir::new(&self.path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file()) 
        {
            let path = entry.path();
            let path_str = path.to_str().unwrap_or_default();

            // Ignores
            if path_str.contains("/target/") || 
               path_str.contains("/.git/") || 
               path_str.contains("/data/kmbit/") ||
               path_str.contains("/node_modules/") 
            {
                continue;
            }

            // Only index code/docs
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if !["rs", "py", "md", "json", "toml", "txt", "js", "ts", "html"].contains(&ext) {
                continue;
            }

            // Process (Manual trigger of handle_logic)
            self.index_file(path_str);
            count += 1;
            
            if count % 100 == 0 {
                println!("🐅 Indexed {} files...", count);
            }
        }

        println!("🐅 Initial scan complete! Total files in KmBiT: {}", count);
        Ok(())
    }

    fn index_file(&self, path_str: &str) {
        // Log to TIBET
        let token = self.tibet.action(
            "FileIndexed",
            "CHIMERA-SCANNER",
            serde_json::json!({ "path": path_str, "mode": "initial_scan" }),
        );

        if let Some(ref engine) = self.engine {
            if let Ok(content) = std::fs::read_to_string(path_str) {
                if content.len() > 0 && content.len() < 100000 {
                    let refined = self.refinery.purify(&content, path_str);
                    if let Ok(vector) = engine.embed(&refined.content) {
                        let record = AIndexRecord {
                            id: refined.id,
                            content: refined.content,
                            vector,
                            purity: refined.purity,
                            source: path_str.to_string(),
                            tibet_token: token.id,
                            indexed_at: chrono::Utc::now(),
                        };
                        let _ = self.persist_record(&record);
                    }
                }
            }
        }
    }

    fn handle_event(&self, event: Event) {
        if event.kind.is_modify() || event.kind.is_create() {
            for path in event.paths {
                if let Some(path_str) = path.to_str() {
                    if path_str.contains("/target/") || path_str.contains("/.git/") || path_str.contains("/data/kmbit/") {
                        continue;
                    }
                    println!("🔍 Change detected: {}", path_str);
                    self.index_file(path_str);
                }
            }
        }
    }

    fn persist_record(&self, record: &AIndexRecord) -> std::io::Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;
        
        let json = serde_json::to_string(record)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("/srv/jtel-stack/data/kmbit/aindex.jsonl")?;
        
        writeln!(file, "{}", json)?;
        Ok(())
    }
}
