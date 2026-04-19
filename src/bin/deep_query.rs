//! KmBiT Deep Query Tool
//! Graaft diep in de AIndex om de architectonische verbanden te vinden.

use oomllama::SentinelClassifier;
use std::path::PathBuf;
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let aindex_path = "/srv/jtel-stack/data/kmbit/aindex.jsonl";
    let classifier = SentinelClassifier::new("models/sentinel")?;
    
    let query = "Hoe werkt de hot-swap van neurale kernen in de Sovereign Kernel?";
    println!("🕵️‍♂️ Deep Query: '{}'", query);
    
    // 1. Maak vector van de vraag
    let query_vec = classifier.text_to_embedding(query)?;
    
    // 2. Scan de index
    let content = std::fs::read_to_string(aindex_path)?;
    let mut results: Vec<(f32, String, String)> = Vec::new();

    for line in content.lines() {
        if let Ok(record) = serde_json::from_str::<oomllama::AIndexRecord>(line) {
            let score = cosine_similarity(&query_vec, &record.vector);
            results.push((score, record.source, record.content));
        }
    }

    // Sorteren op score
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("\n🧠 Top 3 Inzichten uit KmBiT:");
    println!("==================================================");
    for (i, (score, source, content)) in results.iter().take(3).enumerate() {
        println!("{}. [Score: {:.4}] Bron: {}", i + 1, score, source);
        let snippet = content.chars().take(150).collect::<String>().replace("\n", " ");
        println!("   Snippet: {}...", snippet);
        println!("--------------------------------------------------");
    }

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}
