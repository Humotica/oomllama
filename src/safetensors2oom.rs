//! SafeTensors to OOM Converter
//! Convert HuggingFace safetensors (bf16/f32) directly to OOM format.
//!
//! This avoids the Q3_K→f32→Q8 double quantization of gguf2oom,
//! giving much higher weight fidelity: bf16 → f32 → Q8 (single quant step).

use std::collections::HashMap;
use std::path::Path;
use std::io::Write;

use crate::gguf2oom::{OomWriter, OomQuantLevel, Result};

/// Read a safetensors file and return tensor name → (shape, dtype, f32_data) mapping.
/// Handles bf16, f16, and f32 source dtypes.
fn read_safetensors_file(path: &Path) -> Result<Vec<(String, Vec<f32>)>> {
    let data = std::fs::read(path)?;
    let tensors = safetensors::SafeTensors::deserialize(&data)?;

    let mut result = Vec::new();

    for (name, tensor) in tensors.tensors() {
        let dtype = tensor.dtype();
        let raw = tensor.data();

        let f32_values: Vec<f32> = match dtype {
            safetensors::Dtype::F32 => {
                // Direct f32 copy
                raw.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            safetensors::Dtype::BF16 => {
                // BF16 → f32: just shift left by 16 bits
                raw.chunks_exact(2)
                    .map(|chunk| {
                        let bits = (chunk[0] as u32) << 16 | (chunk[1] as u32) << 24;
                        f32::from_bits(bits)
                    })
                    .collect()
            }
            safetensors::Dtype::F16 => {
                // F16 → f32 conversion
                raw.chunks_exact(2)
                    .map(|chunk| {
                        let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_to_f32(h)
                    })
                    .collect()
            }
            other => {
                eprintln!("Warning: unsupported dtype {:?} for tensor {}, skipping", other, name);
                continue;
            }
        };

        result.push((name.to_string(), f32_values));
    }

    Ok(result)
}

/// F16 → F32 conversion (IEEE 754 half-precision)
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 1u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let f32_exp = (127 - 15 - e + 1) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf/NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normal
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Check if tensor should be preserved as F32 (norms, biases)
fn should_preserve_f32(name: &str) -> bool {
    name.contains("norm") || name.ends_with(".bias")
}

/// Convert safetensors model directory to OOM format.
///
/// `model_dir` should contain *.safetensors files and optionally
/// a model.safetensors.index.json for multi-file models.
pub fn convert_safetensors_to_oom<P: AsRef<Path>, Q: AsRef<Path>>(
    model_dir: P,
    output_path: Q,
    quant_level: OomQuantLevel,
    progress_callback: Option<Box<dyn Fn(usize, usize, &str)>>,
) -> Result<()> {
    let model_dir = model_dir.as_ref();

    // Find all safetensors files
    let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err("No safetensors files found in model directory".into());
    }

    println!("Found {} safetensors files in {}", st_files.len(), model_dir.display());

    // Count total tensors first (from index if available)
    let index_path = model_dir.join("model.safetensors.index.json");
    let total_tensors = if index_path.exists() {
        let index_data = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;
        index["weight_map"].as_object().map_or(0, |m| m.len())
    } else {
        // Will count as we go
        0
    };

    if total_tensors > 0 {
        println!("Total tensors: {}", total_tensors);
    }

    let quant_name = match quant_level {
        OomQuantLevel::Q2 => "Q2",
        OomQuantLevel::Q4 => "Q4",
        OomQuantLevel::Q8 => "Q8",
    };
    println!("Output quantization: {}", quant_name);

    let mut writer = OomWriter::create(&output_path)?;
    let mut processed = 0usize;

    for st_path in &st_files {
        let filename = st_path.file_name().unwrap().to_string_lossy();
        println!("\nLoading {}...", filename);

        let tensors = read_safetensors_file(st_path)?;
        println!("  {} tensors in this file", tensors.len());

        for (name, values) in tensors {
            processed += 1;

            if let Some(ref cb) = progress_callback {
                let total = if total_tensors > 0 { total_tensors } else { processed };
                cb(processed, total, &name);
            } else {
                print!("\r  [{}/{}] {:60} ({} vals)    ",
                    processed,
                    if total_tensors > 0 { total_tensors.to_string() } else { "?".to_string() },
                    &name,
                    values.len()
                );
                std::io::stdout().flush()?;
            }

            if should_preserve_f32(&name) {
                writer.add_tensor_f32(&name, &values)?;
            } else {
                match quant_level {
                    OomQuantLevel::Q2 => writer.add_tensor_q2(&name, &values)?,
                    OomQuantLevel::Q4 => writer.add_tensor_q4(&name, &values)?,
                    OomQuantLevel::Q8 => writer.add_tensor_q8(&name, &values)?,
                }
            }
        }
    }

    println!("\n\nWriting OOM file...");
    writer.finish()?;

    let output_size = std::fs::metadata(&output_path)?.len();
    println!("\nConversion complete!");
    println!("  Output: {:.2} GB (OOM {})", output_size as f64 / 1e9, quant_name);
    println!("  Tensors: {}", processed);

    Ok(())
}
