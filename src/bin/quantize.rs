//! OomLlama Quantization Tool (Big Lift Edition)
//!
//! Converts safetensors weights to OomLlama block format using streaming.
//! Can handle 70B+ models on limited RAM by memory-mapping inputs and
//! processing tensor-by-tensor.

use candle_core::{Device, DType};
use candle_core::safetensors::Load;
use std::env;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;
use oomllama::quant::{BlockQ2, BlockQ4};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: quantize <input_dir_or_file> <output.oom> <q2|q4>");
        std::process::exit(1);
    }

    let input_path_str = &args[1];
    let output_path = &args[2];
    let mode = &args[3];

    println!("⚖️  Quantizing ({}) {} to {} (Streaming Mode)...", mode, input_path_str, output_path);

    // Collect all safetensors files if directory, or just the file
    let input_path = Path::new(input_path_str);
    let files = if input_path.is_dir() {
        let mut paths: Vec<_> = std::fs::read_dir(input_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        paths.sort(); // Ensure consistent order
        paths
    } else {
        vec![input_path.to_path_buf()]
    };

    println!("📂 Found {} source files.", files.len());

    let device = Device::Cpu;
    
    let mut out_file = BufWriter::new(File::create(output_path)?);

    // Write Header
    out_file.write_all(b"OOML")?;
    out_file.write_all(&1u32.to_le_bytes())?;

    // We need to count total tensors first without loading data
    let mut total_tensors = 0;
    for file_path in &files {
        let weights = unsafe { candle_core::safetensors::MmapedSafetensors::new(file_path)? };
        total_tensors += weights.tensors().len();
    }
    out_file.write_all(&(total_tensors as u32).to_le_bytes())?;
    println!("🔢 Total tensors to process: {}", total_tensors);

    let mut processed = 0;

    for file_path in &files {
        println!("📄 Processing file: {:?}", file_path.file_name().unwrap());
        // Load one file at a time via mmap
        let weights = unsafe { candle_core::safetensors::MmapedSafetensors::new(file_path)? };
        
        for (name, view) in weights.tensors() {
            if processed % 10 == 0 {
                print!("\r⏳ Processing tensor {}/{}…", processed + 1, total_tensors);
                let _ = std::io::stdout().flush();
            }

            // Load single tensor into RAM (CPU)
            let tensor = view.load(&device)?;
            
            // Convert to F32
            let f32_tensor = match tensor.dtype() {
                DType::F32 => tensor,
                DType::F16 | DType::BF16 => tensor.to_dtype(DType::F32)?,
                _ => {
                    println!("\n⚠️ Skipping {} (unsupported dtype {:?})", name, tensor.dtype());
                    continue; 
                },
            };

            let flat_data = f32_tensor.flatten_all()?.to_vec1::<f32>()?;
            let block_size = 256;
            let num_blocks = (flat_data.len() + block_size - 1) / block_size;
            
            let name_bytes = name.as_bytes();
            out_file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            out_file.write_all(name_bytes)?;

            if mode == "q2" {
                out_file.write_all(&[2u8])?;
                out_file.write_all(&(num_blocks as u32).to_le_bytes())?;
                out_file.write_all(&(flat_data.len() as u32).to_le_bytes())?;
                for i in 0..num_blocks {
                    let start = i * block_size;
                    let end = (start + block_size).min(flat_data.len());
                    let q_block = BlockQ2::new(&flat_data[start..end]);
                    out_file.write_all(&q_block.scale.to_le_bytes())?;
                    out_file.write_all(&q_block.min.to_le_bytes())?;
                    out_file.write_all(&(q_block.data.len() as u32).to_le_bytes())?;
                    out_file.write_all(&q_block.data)?;
                }
            } else {
                out_file.write_all(&[4u8])?;
                out_file.write_all(&(num_blocks as u32).to_le_bytes())?;
                out_file.write_all(&(flat_data.len() as u32).to_le_bytes())?;
                for i in 0..num_blocks {
                    let start = i * block_size;
                    let end = (start + block_size).min(flat_data.len());
                    let q_block = BlockQ4::new(&flat_data[start..end]);
                    out_file.write_all(&q_block.scale.to_le_bytes())?;
                    out_file.write_all(&q_block.min.to_le_bytes())?;
                    out_file.write_all(&(q_block.data.len() as u32).to_le_bytes())?;
                    out_file.write_all(&q_block.data)?;
                }
            }
            processed += 1;
        }
    }

    println!("\n✅ Big Lift Complete! Processed {} tensors.", processed);
    Ok(())
}