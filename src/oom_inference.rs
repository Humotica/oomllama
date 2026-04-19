//! OomLlama Native Inference Engine
//! Part of the Humotica AI Lab - MIT License
//!
//! This module provides native .oom model loading and inference
//! without requiring Ollama or other external dependencies.
//!
//! Credits:
//!   - Format: Gemini IDD & Root AI (Humotica AI Lab)
//!   - Quantization: OomLlama.rs by Humotica

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use memmap2::Mmap;

// ============================================================================
// OOM FILE FORMAT
// ============================================================================

/// OOM Magic bytes: "OOML"
pub const OOM_MAGIC: [u8; 4] = *b"OOML";
pub const OOM_VERSION: u32 = 1;

/// Q2 quantization block size
pub const Q2_BLOCK_SIZE: usize = 256;

/// OOM File Header
#[derive(Debug, Clone)]
pub struct OomHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub n_tensors: u32,
    pub n_kv: u32,
}

/// Tensor metadata in OOM file
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub dtype: u32,
    pub offset: u64,
    pub size: u64,
}

/// Model configuration (from metadata)
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_vocab: u32,
    pub n_ctx: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_layer: u32,
    pub n_ff: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Llama 3.3 70B defaults
        Self {
            n_vocab: 128256,
            n_ctx: 8192,
            n_embd: 8192,
            n_head: 64,
            n_head_kv: 8,
            n_layer: 80,
            n_ff: 28672,
            rope_theta: 500000.0,
            norm_eps: 1e-5,
        }
    }
}

// ============================================================================
// Q2 DEQUANTIZATION
// ============================================================================

/// Q2 quantization block (68 bytes per 256 weights)
#[repr(C, packed)]
pub struct Q2Block {
    pub scale: u16,  // f16 as u16
    pub min: u16,    // f16 as u16
    pub qs: [u8; 64], // 256 weights * 2 bits = 512 bits = 64 bytes
}

/// Convert f16 (as u16) to f32
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal
        let f = (mant as f32) / 1024.0 * (2.0_f32).powi(-14);
        return if sign == 1 { -f } else { f };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        };
    }

    let f = (1.0 + (mant as f32) / 1024.0) * (2.0_f32).powi(exp as i32 - 15);
    if sign == 1 { -f } else { f }
}

/// Dequantize a Q2 block to f32 values
pub fn dequant_q2_block(block: &Q2Block) -> [f32; Q2_BLOCK_SIZE] {
    let scale = f16_to_f32(block.scale);
    let min = f16_to_f32(block.min);
    let mut out = [0.0f32; Q2_BLOCK_SIZE];

    for i in 0..Q2_BLOCK_SIZE {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let q2_val = (block.qs[byte_idx] >> bit_offset) & 0b11; // 0, 1, 2, or 3
        out[i] = (q2_val as f32) * scale + min;
    }
    out
}

// ============================================================================
// TRANSFORMER OPERATIONS
// ============================================================================

/// RMS Normalization
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();

    // Calculate RMS
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();

    // Normalize and scale
    x.iter()
        .zip(weight.iter())
        .map(|(xi, wi)| (xi / rms) * wi)
        .collect()
}

/// Softmax over a slice
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;

    for xi in x.iter_mut() {
        *xi = (*xi - max).exp();
        sum += *xi;
    }

    for xi in x.iter_mut() {
        *xi /= sum;
    }
}

/// Matrix-vector multiplication: y = x @ W^T
/// x: [1, in_features], W: [out_features, in_features]
pub fn matvec(x: &[f32], w: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_features];

    for i in 0..out_features {
        let row_start = i * in_features;
        let mut sum = 0.0f32;
        for j in 0..in_features {
            sum += x[j] * w[row_start + j];
        }
        y[i] = sum;
    }
    y
}

/// SiLU activation (used in SwiGLU)
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU FFN: output = down @ (silu(gate @ x) * up @ x)
pub fn swiglu_ffn(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden_dim: usize,
    dim: usize,
) -> Vec<f32> {
    // gate = silu(x @ w_gate^T)
    let gate = matvec(x, w_gate, hidden_dim, dim);
    let gate: Vec<f32> = gate.into_iter().map(silu).collect();

    // up = x @ w_up^T
    let up = matvec(x, w_up, hidden_dim, dim);

    // hidden = gate * up (element-wise)
    let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect();

    // output = hidden @ w_down^T
    matvec(&hidden, w_down, dim, hidden_dim)
}

/// RoPE (Rotary Position Embedding) - INTERLEAVED FORMAT
/// GGUF/OOM stores Q/K as [x0, x32, x1, x33, ...] (interleaved pairs)
/// NOT as [x0, x1, ..., x31, x32, ...] (sequential pairs)
pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let half_dim = head_dim / 2;

    for i in 0..half_dim {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let cos = angle.cos();
        let sin = angle.sin();

        // INTERLEAVED: pairs are at [i*2, i*2+1], not [i, i+half_dim]
        let idx0 = i * 2;
        let idx1 = i * 2 + 1;

        // Rotate Q (interleaved format)
        let q0 = q[idx0];
        let q1 = q[idx1];
        q[idx0] = q0 * cos - q1 * sin;
        q[idx1] = q0 * sin + q1 * cos;

        // Rotate K (interleaved format)
        let k0 = k[idx0];
        let k1 = k[idx1];
        k[idx0] = k0 * cos - k1 * sin;
        k[idx1] = k0 * sin + k1 * cos;
    }
}

// ============================================================================
// ATTENTION
// ============================================================================

/// Single-head attention computation
pub fn attention_head(
    q: &[f32],       // [head_dim]
    k_cache: &[f32], // [seq_len, head_dim]
    v_cache: &[f32], // [seq_len, head_dim]
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention scores: Q @ K^T
    let mut scores = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let k_offset = t * head_dim;
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k_cache[k_offset + d];
        }
        scores[t] = dot * scale;
    }

    // Softmax
    softmax(&mut scores);

    // Weighted sum of V: scores @ V
    let mut out = vec![0.0f32; head_dim];
    for t in 0..seq_len {
        let v_offset = t * head_dim;
        for d in 0..head_dim {
            out[d] += scores[t] * v_cache[v_offset + d];
        }
    }
    out
}

// ============================================================================
// OOM MODEL LOADER
// ============================================================================

/// OomLlama model loaded from .oom file
pub struct OomModel {
    pub config: ModelConfig,
    pub tensors: std::collections::HashMap<String, TensorInfo>,
    mmap: Mmap,
}

impl OomModel {
    /// Load model from .oom file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Verify magic
        if &mmap[0..4] != OOM_MAGIC {
            return Err("Invalid OOM file: bad magic".into());
        }

        // Parse header
        let version = u32::from_le_bytes(mmap[4..8].try_into()?);
        if version != OOM_VERSION {
            return Err(format!("Unsupported OOM version: {}", version).into());
        }

        let n_tensors = u32::from_le_bytes(mmap[8..12].try_into()?);
        let n_kv = u32::from_le_bytes(mmap[12..16].try_into()?);

        // TODO: Parse tensor infos and metadata
        // For now, use defaults
        let config = ModelConfig::default();
        let tensors = std::collections::HashMap::new();

        Ok(Self { config, tensors, mmap })
    }

    /// Get tensor data by name
    pub fn get_tensor(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|info| {
            let start = info.offset as usize;
            let end = start + info.size as usize;
            &self.mmap[start..end]
        })
    }
}

// ============================================================================
// INFERENCE
// ============================================================================

/// Token sampler with temperature and top-p
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
        }
    }
}

impl Sampler {
    /// Sample next token from logits
    pub fn sample(&self, logits: &mut [f32]) -> u32 {
        // Apply temperature
        if self.temperature != 1.0 {
            for l in logits.iter_mut() {
                *l /= self.temperature;
            }
        }

        // Softmax
        softmax(logits);

        // Top-k filtering
        let mut indices: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indices.truncate(self.top_k);

        // Top-p (nucleus) filtering
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indices.len();
        for (i, (_, prob)) in indices.iter().enumerate() {
            cumsum += prob;
            if cumsum >= self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        indices.truncate(cutoff_idx);

        // Renormalize
        let sum: f32 = indices.iter().map(|(_, p)| p).sum();

        // Save fallback before iterating
        let fallback = indices.first().map(|(idx, _)| *idx as u32).unwrap_or(0);

        // Sample
        let r: f32 = rand::random::<f32>() * sum;
        let mut acc = 0.0f32;
        for (idx, prob) in indices {
            acc += prob;
            if acc >= r {
                return idx as u32;
            }
        }

        // Fallback to most likely
        fallback
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        // Test zero
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // Test one (0x3C00 in f16)
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001);
        // Test negative one
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let out = rms_norm(&x, &w, 1e-5);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 0.001);
        assert!(silu(1.0) > 0.7);
        assert!(silu(-1.0) < -0.2);
    }
}
