//! OomLlama CLI - Sovereign Inference
//!
//! Usage:
//!   oomllama --model model.oom "Hello world"
//!   oomllama --model model.oom --gpu 0 "Tell me a story"
//!   oomllama --model model.oom --max-tokens 100 "prompt"

use std::env;
use std::sync::Arc;
use oomllama::oomllama::OomLlama;
use oomllama::betti::{BettiManager, ResourcePool, ResourceType};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();

    // Parse flags first, then find the prompt (last non-flag argument)
    let max_tokens = args.iter()
        .position(|a| a == "--max-tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let gpu_index = args.iter()
        .position(|a| a == "--gpu")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    let model_path = args.iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let tokenizer_path = args.iter()
        .position(|a| a == "--tokenizer")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    // Find prompt: last argument that's not a flag or flag value
    let mut skip_next = false;
    let prompt: Option<String> = args.iter().skip(1).rev().find_map(|arg| {
        if skip_next {
            skip_next = false;
            return None;
        }
        if arg.starts_with("--") {
            skip_next = true; // Skip the flag value on next iteration
            return None;
        }
        // Check if previous arg was a flag (meaning this is a flag value)
        let idx = args.iter().position(|a| a == arg).unwrap();
        if idx > 0 && args[idx - 1].starts_with("--") {
            return None;
        }
        Some(arg.clone())
    });

    let prompt = match prompt {
        Some(p) => p,
        None => {
            eprintln!("🦙 OomLlama CLI - Native .oom inference\n");
            eprintln!("Usage: oomllama --model <path.oom> [options] \"<prompt>\"\n");
            eprintln!("Options:");
            eprintln!("  --model <path>       Path to .oom model file");
            eprintln!("  --gpu <index>        GPU index (default: 0)");
            eprintln!("  --max-tokens <n>     Max tokens to generate (default: 50)");
            eprintln!("  --tokenizer <path>   Path to tokenizer (optional)\n");
            eprintln!("Example:");
            eprintln!("  oomllama --model /srv/humotica/models/humotica-32b.oom --gpu 0 \"Hoi!\"");
            std::process::exit(1);
        }
    };

    // Initialize Betti
    let betti = Arc::new(BettiManager::new());
    betti.register_pool(ResourcePool::new(ResourceType::Cpu, 32.0, "GB", "Local"));
    betti.register_pool(ResourcePool::new(ResourceType::Gpu, 24.0, "GB", "DualRTX3060"));

    println!("🧠 OomLlama waking up...");
    
    let mut llama = match OomLlama::new("oomllama-cli", gpu_index, betti, model_path, tokenizer_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("❌ Failed to initialize OomLlama: {}", e);
            eprintln!("   (Make sure you provided a valid model path via --model)");
            std::process::exit(1);
        }
    };

    println!("💬 Prompt: {}", &prompt);
    println!("🤖 Thinking...");

    let response = llama.infer(&prompt, max_tokens)?;
    
    println!("\n--- RESPONSE ---");
    println!("{}", response);
    println!("----------------");
    
    Ok(())
}
