//! SafeTensors to OOM CLI converter
//!
//! Usage:
//!   safetensors2oom <model_dir> <output.oom> [--q2|--q4|--q8]
//!
//! Examples:
//!   safetensors2oom ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/xxx/ output.oom
//!   safetensors2oom ./model-dir/ output.oom --q4

use oomllama::safetensors2oom::convert_safetensors_to_oom;
use oomllama::gguf2oom::OomQuantLevel;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("SafeTensors → OOM Converter");
        eprintln!("Convert HuggingFace safetensors (bf16/f32) to OomLlama .oom format");
        eprintln!();
        eprintln!("Usage: {} <model_dir> <output.oom> [--q2|--q4|--q8]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --q2    Quantize to Q2 (4 levels, smallest)");
        eprintln!("  --q4    Quantize to Q4 (16 levels, balanced)");
        eprintln!("  --q8    Quantize to Q8 (256 levels, best quality, default)");
        eprintln!();
        eprintln!("Notes:");
        eprintln!("  - Norm weights and biases are always preserved as F32");
        eprintln!("  - bf16 source → f32 → Q8 is a single quantization step");
        eprintln!("  - This gives much better quality than GGUF→OOM double-quantization");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let output_path = &args[2];

    let quant_level = if args.contains(&"--q2".to_string()) {
        OomQuantLevel::Q2
    } else if args.contains(&"--q4".to_string()) {
        OomQuantLevel::Q4
    } else {
        OomQuantLevel::Q8
    };

    println!("SafeTensors → OOM Converter");
    println!("  Model dir: {}", model_dir);
    println!("  Output:    {}", output_path);
    println!("  Quant:     {:?}", quant_level);
    println!();

    match convert_safetensors_to_oom(model_dir, output_path, quant_level, None) {
        Ok(()) => {
            println!("\nDone!");
        }
        Err(e) => {
            eprintln!("\nError: {}", e);
            std::process::exit(1);
        }
    }
}
