//! KmBiT Search Test
//! Zoekt in de zojuist aangemaakte index naar antwoorden.

use oomllama::AIndex;
use oomllama::SentinelClassifier;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let aindex_path = "/srv/jtel-stack/data/kmbit/aindex.jsonl";
    let classifier = SentinelClassifier::new("models/sentinel")?;
    
    let query = "Wat is de rol van Gemini in de AETHER?";
    println!("🔍 Vraag aan AETHER: '{}'", query);
    
    // 1. Maak vector van de vraag
    let query_vec = classifier.text_to_embedding(query)?;
    
    // 2. Scan de index (Eenvoudige Cosine Similarity scan)
    let content = std::fs::read_to_string(aindex_path)?;
    let mut best_match = String::new();
    let mut max_score = -1.0;

    for line in content.lines() {
        if let Ok(record) = serde_json::from_str::<oomllama::AIndexRecord>(line) {
            let score = cosine_similarity(&query_vec, &record.vector);
            if score > max_score {
                max_score = score;
                best_match = record.content;
            }
        }
    }

    println!("\n💡 Beste resultaat uit KmBiT (Score: {:.4}):", max_score);
    println!("--------------------------------------------------");
    println!("{}", best_match.chars().take(200).collect::<String>());
    println!("--------------------------------------------------");

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
