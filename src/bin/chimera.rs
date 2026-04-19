//! CHIMERA-SCANNER CLI (batch mode).
//!
//! Usage:
//!   chimera scan <path>              # Scan without embeddings
//!   chimera scan <path> --embed      # Scan WITH GPU embeddings
//!   chimera report <path>
//!   chimera diff <path>

use std::env;
use std::path::{Path, PathBuf};

use oomllama::batch::{BatchConfig, BatchProcessor, write_vectors_jsonl};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: chimera <scan|report|diff> <path> [--embed]");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --embed    Enable GPU embeddings (requires CUDA)");
        eprintln!("  --gpu N    Use GPU N for embeddings (default: 1)");
        std::process::exit(1);
    }
    let command = &args[1];
    let path = PathBuf::from(&args[2]);

    // Parse flags
    let enable_embed = args.iter().any(|a| a == "--embed");
    let gpu_index = args.iter()
        .position(|a| a == "--gpu")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    match command.as_str() {
        "scan" => run_scan(&path, false, enable_embed, gpu_index),
        "report" => run_scan(&path, true, enable_embed, gpu_index),
        "diff" => run_diff(&path, enable_embed, gpu_index),
        _ => {
            eprintln!("Unknown command: {}", command);
            std::process::exit(1);
        }
    }
}

fn run_scan(path: &Path, print_report: bool, enable_embed: bool, gpu_index: usize) {
    let cfg = BatchConfig {
        enable_embeddings: enable_embed,
        gpu_index,
        ..BatchConfig::default()
    };

    if enable_embed {
        println!("🚀 GPU embeddings enabled on GPU {}", gpu_index);
    }

    let batch = BatchProcessor::new(cfg);
    let files = collect_files(path);
    let total_files = files.len();
    let mut all_vectors = Vec::new();
    let mut all_items = Vec::new();

    for (i, file) in files.iter().enumerate() {
        if enable_embed {
            println!("📄 [{}/{}] Processing: {}", i + 1, total_files, file.display());
        }
        if let Ok((report, vectors)) = batch.process_path(file, "BatchScan") {
            all_items.extend(report.items);
            all_vectors.extend(vectors);
        }
    }

    let embedded_count = all_vectors.iter().filter(|v| !v.vector.is_empty()).count();
    let mut report = oomllama::report::BatchReport::new(all_items, all_vectors.len());
    report.sign(&oomllama::tibet::TibetFactory::new("chimera-cli"));

    let out_path = PathBuf::from("data/kmbit/batches").join(format!("{}.jsonl", report.batch_id));
    let _ = write_vectors_jsonl(&out_path, &all_vectors);

    println!("📦 Vectors written: {}", out_path.display());
    println!("   Total chunks: {}", all_vectors.len());
    if enable_embed {
        println!("   Embedded: {} (384 dimensions each)", embedded_count);
    }

    if print_report {
        let json = serde_json::to_string_pretty(&report).unwrap_or_default();
        println!("{}", json);
    }
}

fn run_diff(path: &Path, enable_embed: bool, gpu_index: usize) {
    // For now, diff is just a scan that reports changed files only.
    run_scan(path, true, enable_embed, gpu_index);
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if root.is_file() {
        files.push(root.to_path_buf());
        return files;
    }
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let p = path.to_string_lossy();
                if p.contains("/.git") || p.contains("/target") {
                    continue;
                }
                files.extend(collect_files(&path));
            } else {
                files.push(path);
            }
        }
    }
    files
}
