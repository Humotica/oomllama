//! GGUF to OOM Converter CLI
//!
//! Convert any GGUF model to OomLlama's native .oom format
//!
//! Usage:
//!   gguf2oom input.gguf output.oom          # Convert with Q8 (default, best quality)
//!   gguf2oom --q4 input.gguf output.oom     # Convert with Q4 (smaller)
//!   gguf2oom --q2 input.gguf output.oom     # Convert with Q2 (smallest, low quality)
//!   gguf2oom --info input.gguf
//!
//! One love, one fAmIly! 🦙

use std::env;
use std::process;
use oomllama::gguf2oom::{GgufReader, convert_gguf_to_oom_with_quant, OomQuantLevel};

fn print_usage() {
    eprintln!(r#"
🦙 GGUF to OOM Converter - OomLlama Format Tool

Usage:
    gguf2oom <input.gguf> <output.oom>        Convert GGUF to OOM Q8 (best quality)
    gguf2oom --q8 <input.gguf> <output.oom>   Convert with Q8 (256 levels, ~same size)
    gguf2oom --q4 <input.gguf> <output.oom>   Convert with Q4 (16 levels, ~2x smaller)
    gguf2oom --q2 <input.gguf> <output.oom>   Convert with Q2 (4 levels, ~4x smaller)
    gguf2oom --info <input.gguf>              Show GGUF file info
    gguf2oom --help                           Show this help

Quantization levels:
    Q8: 256 levels per value (8 bits) - best quality, same size as Q8 GGUF
    Q4: 16 levels per value (4 bits)  - good balance, ~2x smaller
    Q2: 4 levels per value (2 bits)   - smallest, noticeable quality loss

Examples:
    gguf2oom humotica-7b.gguf humotica-7b.oom
    gguf2oom --q4 /path/to/big-model.gguf smaller-model.oom
    gguf2oom --info /path/to/model.gguf

The converter:
1. Reads GGUF file (any quantization: Q4_K, Q8_0, etc.)
2. Dequantizes each tensor to FP32
3. Requantizes to OOM format (Q2/Q4/Q8)
4. Writes compact .oom file

One love, one fAmIly! 🦙
"#);
}

fn show_gguf_info(path: &str) {
    println!("\n🔍 GGUF File Info: {}\n", path);

    match GgufReader::open(path) {
        Ok(reader) => {
            println!("Header:");
            println!("  Version: {}", reader.header.version);
            println!("  Tensors: {}", reader.header.tensor_count);
            println!("  Metadata entries: {}", reader.header.metadata_kv_count);

            println!("\nMetadata:");
            for (key, value) in &reader.metadata {
                let display_val = if value.len() > 60 {
                    format!("{}...", &value[..60])
                } else {
                    value.clone()
                };
                println!("  {}: {}", key, display_val);
            }

            println!("\nTensors ({}):", reader.tensors.len());
            let mut total_elements: u64 = 0;
            for tensor in &reader.tensors {
                let elements: u64 = tensor.dims.iter().product();
                total_elements += elements;
                let dims_str: Vec<String> = tensor.dims.iter().map(|d| d.to_string()).collect();
                println!("  {} [{:?}] {:?} ({} elements)",
                    tensor.name,
                    dims_str.join("x"),
                    tensor.dtype,
                    elements
                );
            }

            println!("\nTotal elements: {} ({:.2} B params)", total_elements, total_elements as f64 / 1e9);

            // Estimate sizes
            let q8_bytes = total_elements as f64 + (total_elements as f64 / 256.0 * 12.0);
            let q4_bytes = (total_elements as f64 / 2.0) + (total_elements as f64 / 256.0 * 12.0);
            let q2_bytes = (total_elements as f64 / 4.0) + (total_elements as f64 / 256.0 * 12.0);
            println!("\nEstimated OOM sizes:");
            println!("  Q8: {:.2} GB (best quality)", q8_bytes / 1e9);
            println!("  Q4: {:.2} GB (good balance)", q4_bytes / 1e9);
            println!("  Q2: {:.2} GB (smallest)", q2_bytes / 1e9);
        }
        Err(e) => {
            eprintln!("❌ Error reading GGUF: {}", e);
            process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "--help" | "-h" => {
            print_usage();
        }
        "--info" | "-i" => {
            if args.len() < 3 {
                eprintln!("❌ Missing input file for --info");
                process::exit(1);
            }
            show_gguf_info(&args[2]);
        }
        "--q2" | "--q4" | "--q8" => {
            if args.len() < 4 {
                eprintln!("❌ Missing input or output file");
                print_usage();
                process::exit(1);
            }

            let quant_level = match args[1].as_str() {
                "--q2" => OomQuantLevel::Q2,
                "--q4" => OomQuantLevel::Q4,
                _ => OomQuantLevel::Q8,
            };
            let quant_name = match quant_level {
                OomQuantLevel::Q2 => "Q2 (4 levels)",
                OomQuantLevel::Q4 => "Q4 (16 levels)",
                OomQuantLevel::Q8 => "Q8 (256 levels)",
            };

            let input = &args[2];
            let output = &args[3];

            println!("🦙 GGUF → OOM Converter");
            println!("========================");
            println!("Input:  {}", input);
            println!("Output: {}", output);
            println!("Quant:  {}", quant_name);

            match convert_gguf_to_oom_with_quant(input, output, quant_level, None) {
                Ok(()) => {
                    println!("\n🎉 Conversion successful!");
                    println!("\nTest with:");
                    println!("  /opt/debain/bin/oomllama --model {} --gpu 0 \"Hello\"", output);
                }
                Err(e) => {
                    eprintln!("\n❌ Conversion failed: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            if args.len() < 3 {
                eprintln!("❌ Missing output file");
                print_usage();
                process::exit(1);
            }

            let input = &args[1];
            let output = &args[2];

            println!("🦙 GGUF → OOM Converter");
            println!("========================");
            println!("Input:  {}", input);
            println!("Output: {}", output);
            println!("Quant:  Q8 (256 levels, best quality)");

            match convert_gguf_to_oom_with_quant(input, output, OomQuantLevel::Q8, None) {
                Ok(()) => {
                    println!("\n🎉 Conversion successful!");
                    println!("\nTest with:");
                    println!("  /opt/debain/bin/oomllama --model {} --gpu 0 \"Hello\"", output);
                }
                Err(e) => {
                    eprintln!("\n❌ Conversion failed: {}", e);
                    process::exit(1);
                }
            }
        }
    }
}
