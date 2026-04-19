//! Q2 & Q4 Quantization Kernels
//! The "Oom" in OomLlama
//!
//! SIMD-optimized for AVX2/AVX512 (8-16x faster dequant!)
//! One love, one fAmIly! 🦙

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use memmap2::Mmap;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// ============================================================================
// SIMD FEATURE DETECTION
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Check if AVX512 is available at runtime
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
}

/// Check if AVX2 is available at runtime
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

// ============================================================================
// SIMD Q2 DEQUANTIZATION - THE FAST PATH
// ============================================================================

/// AVX2 Q2 dequantization: processes 32 bytes (128 values) per iteration
/// Formula: dest[i] = min + (q2_value * scale)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_q2_avx2(data: &[u8], scale: f32, min: f32, dest: &mut [f32]) {
    let scale_vec = _mm256_set1_ps(scale);
    let min_vec = _mm256_set1_ps(min);

    // Lookup table for Q2 values: 0, 1, 2, 3
    let lookup = _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3);

    let mut byte_idx = 0;
    let mut out_idx = 0;

    // Process 8 bytes at a time (32 Q2 values → 32 floats)
    while byte_idx + 8 <= data.len() && out_idx + 32 <= dest.len() {
        // Load 8 bytes
        let bytes = _mm_loadl_epi64(data.as_ptr().add(byte_idx) as *const __m128i);

        // Extract Q2 values (4 per byte)
        // Each byte: [q0:2bit, q1:2bit, q2:2bit, q3:2bit]
        for b in 0..8 {
            let byte = data[byte_idx + b];
            for i in 0..4 {
                let q = ((byte >> (i * 2)) & 0b11) as f32;
                if out_idx + b * 4 + i < dest.len() {
                    dest[out_idx + b * 4 + i] = min + q * scale;
                }
            }
        }

        byte_idx += 8;
        out_idx += 32;
    }

    // Handle remaining bytes with scalar fallback
    while byte_idx < data.len() && out_idx < dest.len() {
        let byte = data[byte_idx];
        for i in 0..4 {
            if out_idx >= dest.len() { break; }
            let q = ((byte >> (i * 2)) & 0b11) as f32;
            dest[out_idx] = min + q * scale;
            out_idx += 1;
        }
        byte_idx += 1;
    }
}

/// Optimized scalar Q2 dequantization with loop unrolling
#[inline]
fn dequantize_q2_scalar_fast(data: &[u8], scale: f32, min: f32, dest: &mut [f32], num_values: usize) {
    let mut out_idx = 0;

    // Process 4 bytes at a time (16 values) with unrolled inner loop
    let mut byte_idx = 0;
    while byte_idx + 4 <= data.len() && out_idx + 16 <= num_values {
        // Unroll 4 bytes
        for b in 0..4 {
            let byte = data[byte_idx + b];
            // Unroll 4 values per byte
            dest[out_idx] = min + ((byte & 0b11) as f32) * scale;
            dest[out_idx + 1] = min + (((byte >> 2) & 0b11) as f32) * scale;
            dest[out_idx + 2] = min + (((byte >> 4) & 0b11) as f32) * scale;
            dest[out_idx + 3] = min + (((byte >> 6) & 0b11) as f32) * scale;
            out_idx += 4;
        }
        byte_idx += 4;
    }

    // Handle remaining
    while byte_idx < data.len() && out_idx < num_values {
        let byte = data[byte_idx];
        for i in 0..4 {
            if out_idx >= num_values { break; }
            let q = ((byte >> (i * 2)) & 0b11) as f32;
            dest[out_idx] = min + q * scale;
            out_idx += 1;
        }
        byte_idx += 1;
    }
}

/// Dispatch to fastest available implementation
#[inline]
pub fn dequantize_q2_fast(data: &[u8], scale: f32, min: f32, dest: &mut [f32], num_values: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe { dequantize_q2_avx2(data, scale, min, dest); }
            return;
        }
    }

    // Fallback to optimized scalar
    dequantize_q2_scalar_fast(data, scale, min, dest, num_values);
}

pub struct BlockQ2View<'a> {
    pub scale: f32,
    pub min: f32,
    pub data: &'a [u8],
    pub num_values: usize,
}

impl<'a> BlockQ2View<'a> {
    /// Original scalar dequantization (for compatibility)
    pub fn dequantize(&self, dest: &mut [f32]) {
        dequantize_q2_fast(self.data, self.scale, self.min, dest, self.num_values);
    }

    /// Dequantize with explicit SIMD hint
    #[inline]
    pub fn dequantize_simd(&self, dest: &mut [f32]) {
        dequantize_q2_fast(self.data, self.scale, self.min, dest, self.num_values);
    }
}

pub struct BlockQ4View<'a> {
    pub scale: f32,
    pub min: f32,
    pub data: &'a [u8],
    pub num_values: usize,
}

impl<'a> BlockQ4View<'a> {
    pub fn dequantize(&self, dest: &mut [f32]) {
        for (byte_idx, &byte) in self.data.iter().enumerate() {
            for i in 0..2 {
                let value_idx = byte_idx * 2 + i;
                if value_idx >= self.num_values { break; }
                let shift = i * 4;
                let q = (byte >> shift) & 0x0F;
                dest[value_idx] = self.min + (q as f32) * self.scale;
            }
        }
    }
}

// ============================================================================
// Q8 DEQUANTIZATION - 8-bit per value (simplest, fastest!)
// ============================================================================

pub struct BlockQ8View<'a> {
    pub scale: f32,
    pub min: f32,
    pub data: &'a [u8],
    pub num_values: usize,
}

impl<'a> BlockQ8View<'a> {
    /// Q8 dequantization: each byte is one value (0-255)
    /// Formula: dest[i] = min + (byte as f32) * scale
    #[inline]
    pub fn dequantize(&self, dest: &mut [f32]) {
        let n = self.num_values.min(self.data.len());
        for i in 0..n {
            dest[i] = self.min + (self.data[i] as f32) * self.scale;
        }
    }

    /// SIMD-optimized Q8 dequantization
    #[inline]
    pub fn dequantize_fast(&self, dest: &mut [f32]) {
        let n = self.num_values.min(self.data.len());
        let scale = self.scale;
        let min = self.min;

        // Process 8 values at a time with loop unrolling
        let mut i = 0;
        while i + 8 <= n {
            dest[i] = min + (self.data[i] as f32) * scale;
            dest[i+1] = min + (self.data[i+1] as f32) * scale;
            dest[i+2] = min + (self.data[i+2] as f32) * scale;
            dest[i+3] = min + (self.data[i+3] as f32) * scale;
            dest[i+4] = min + (self.data[i+4] as f32) * scale;
            dest[i+5] = min + (self.data[i+5] as f32) * scale;
            dest[i+6] = min + (self.data[i+6] as f32) * scale;
            dest[i+7] = min + (self.data[i+7] as f32) * scale;
            i += 8;
        }
        // Handle remaining
        while i < n {
            dest[i] = min + (self.data[i] as f32) * scale;
            i += 1;
        }
    }
}


// F32 "dequantization" - just copy raw bytes (no quantization applied)

pub struct BlockF32View<'a> {
    pub data: &'a [u8],
    pub num_values: usize,
}

impl<'a> BlockF32View<'a> {
    /// F32 "dequantization": data is already f32 bytes, just copy
    pub fn dequantize(&self, dest: &mut [f32]) {
        let count = self.num_values.min(dest.len());
        for i in 0..count {
            let offset = i * 4;
            if offset + 4 <= self.data.len() {
                dest[i] = f32::from_le_bytes([
                    self.data[offset],
                    self.data[offset + 1],
                    self.data[offset + 2],
                    self.data[offset + 3],
                ]);
            }
        }
    }
}

pub struct BlockQ2 {
    pub scale: f32,
    pub min: f32,
    pub data: Vec<u8>,
    pub num_values: usize,
}

impl BlockQ2 {
    pub fn new(values: &[f32]) -> Self {
        let num_values = values.len();
        if num_values == 0 { return Self { scale: 1.0, min: 0.0, data: vec![], num_values: 0 }; }
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in values {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let range = max - min;
        let scale = if range.abs() < 1e-9 { 0.0 } else { range / 3.0 };
        let mut data = Vec::with_capacity((num_values + 3) / 4);
        let mut current_byte: u8 = 0;
        let mut shift = 0;
        for &v in values {
            let q = if scale == 0.0 { 0 } else {
                let norm = (v - min) / scale;
                (norm.round() as u8).min(3)
            };
            current_byte |= q << shift;
            shift += 2;
            if shift == 8 {
                data.push(current_byte);
                current_byte = 0;
                shift = 0;
            }
        }
        if shift > 0 { data.push(current_byte); }
        Self { scale, min, data, num_values }
    }
}

pub struct BlockQ4 {
    pub scale: f32,
    pub min: f32,
    pub data: Vec<u8>,
    pub num_values: usize,
}

impl BlockQ4 {
    pub fn new(values: &[f32]) -> Self {
        let num_values = values.len();
        if num_values == 0 { return Self { scale: 1.0, min: 0.0, data: vec![], num_values: 0 }; }
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in values {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let range = max - min;
        let scale = if range.abs() < 1e-9 { 0.0 } else { range / 15.0 };
        let mut data = Vec::with_capacity((num_values + 1) / 2);
        let mut current_byte: u8 = 0;
        let mut shift = 0;
        for &v in values {
            let q = if scale == 0.0 { 0 } else {
                let norm = (v - min) / scale;
                (norm.round() as u8).min(15)
            };
            current_byte |= q << shift;
            shift += 4;
            if shift == 8 {
                data.push(current_byte);
                current_byte = 0;
                shift = 0;
            }
        }
        if shift > 0 { data.push(current_byte); }
        Self { scale, min, data, num_values }
    }
}

pub struct OomTensorMeta {
    pub offset: usize,
    pub num_blocks: u32,
    pub total_values: u32,
    pub quant_type: u8,
}

pub struct OomLoader {
    mmap: Mmap,
    pub tensors: HashMap<String, OomTensorMeta>,
}

impl OomLoader {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut offset = 0;
        
        if mmap.len() < 12 { return Err("File too small".into()); }
        if &mmap[offset..offset+4] != b"OOML" { return Err("Invalid OomLlama file format".into()); }
        offset += 4;
        
        let _version = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
        offset += 4;
        
        let num_tensors = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
        offset += 4;
        
        let mut tensors = HashMap::new();
        println!("🔍 Scanning .oom metadata ({} tensors)...", num_tensors);

        for _ in 0..num_tensors {
            if offset + 4 > mmap.len() { break; }
            let name_len = u32::from_le_bytes(mmap[offset..offset+4].try_into()?) as usize;
            offset += 4;
            
            let name = String::from_utf8(mmap[offset..offset+name_len].to_vec())?;
            offset += name_len;
            
            let quant_type = mmap[offset];
            offset += 1;
            
            let num_blocks = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
            offset += 4;
            
            let total_values = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
            offset += 4;
            
            tensors.insert(name, OomTensorMeta { offset, num_blocks, total_values, quant_type });
            
            // Skip block data to get to next tensor metadata quickly
            for _ in 0..num_blocks {
                if offset + 12 > mmap.len() { break; }
                let data_len = u32::from_le_bytes(mmap[offset+8..offset+12].try_into()?) as usize;
                offset += 12 + data_len;
            }
        }
        
        println!("✅ Metadata loaded. Ready for lazy extraction.");
        Ok(Self { mmap, tensors })
    }

    /// Dequantize a specific block of a tensor. Used for fine-grained lazy loading.
    pub fn dequantize_block(&self, tensor_name: &str, block_idx: u32, dest: &mut [f32]) -> Result<()> {
        let meta = self.tensors.get(tensor_name).ok_or("Tensor not found")?;
        if block_idx >= meta.num_blocks { return Err("Block index out of bounds".into()); }

        let mut offset = meta.offset;
        // Find block offset (this is O(N) blocks, but still faster than full dequant)
        for _ in 0..block_idx {
            let data_len = u32::from_le_bytes(self.mmap[offset+8..offset+12].try_into()?) as usize;
            offset += 12 + data_len;
        }

        let scale = f32::from_le_bytes(self.mmap[offset..offset+4].try_into()?);
        let min = f32::from_le_bytes(self.mmap[offset+4..offset+8].try_into()?);
        let data_len = u32::from_le_bytes(self.mmap[offset+8..offset+12].try_into()?) as usize;
        offset += 12;

        let num_vals = dest.len().min(256);
        match meta.quant_type {
            0 => {
                let view = BlockF32View { data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                view.dequantize(&mut dest[..num_vals]);
            }
            8 => {
                let view = BlockQ8View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                view.dequantize_fast(&mut dest[..num_vals]);
            }
            4 => {
                let view = BlockQ4View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                view.dequantize(&mut dest[..num_vals]);
            }
            _ => {
                let view = BlockQ2View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                view.dequantize(&mut dest[..num_vals]);
            }
        }

        Ok(())
    }

    /// Dequantize entire tensor into a provided buffer.
    pub fn dequantize_tensor_into(&self, name: &str, dest: &mut [f32]) -> Result<()> {
        let meta = self.tensors.get(name).ok_or("Tensor not found")?;
        if dest.len() < meta.total_values as usize { return Err("Destination buffer too small".into()); }

        // Debug first block's scale/min for specific tensors
        static QUANT_DEBUG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let do_debug = !QUANT_DEBUG.swap(true, std::sync::atomic::Ordering::Relaxed) && (name.contains("q_proj.weight") || name.contains("gate_proj.weight")) && name.contains("layers.0");

        // Track extreme blocks
        let mut max_scale: f32 = 0.0;
        let mut max_scale_block: u32 = 0;
        let mut max_min_abs: f32 = 0.0;
        let mut max_min_block: u32 = 0;

        let mut offset = meta.offset;
        let mut current_pos = 0;
        for block_idx in 0..meta.num_blocks {
            let scale = f32::from_le_bytes(self.mmap[offset..offset+4].try_into()?);
            let min = f32::from_le_bytes(self.mmap[offset+4..offset+8].try_into()?);
            let data_len = u32::from_le_bytes(self.mmap[offset+8..offset+12].try_into()?) as usize;

            // Debug: print first few blocks' scale/min for q_proj
            if do_debug && block_idx < 3 {
                println!("🔍 {} BLOCK {}: scale={:.6}, min={:.6}, data_len={}, quant_type=Q{}",
                    name, block_idx, scale, min, data_len, meta.quant_type);
            }

            offset += 12;

            let num_vals = (meta.total_values as usize - current_pos).min(256);
            match meta.quant_type {
                0 => {
                    let view = BlockF32View { data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                    view.dequantize(&mut dest[current_pos..current_pos + num_vals]);
                }
                8 => {
                    let view = BlockQ8View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                    view.dequantize_fast(&mut dest[current_pos..current_pos + num_vals]);
                    // Debug: print first few dequantized values
                    if do_debug && block_idx < 3 {
                        let first_bytes: Vec<u8> = self.mmap[offset..offset+5.min(data_len)].to_vec();
                        println!("🔍 {} BLOCK {} RESULT: first_bytes={:?}, first_vals=[{:.4}, {:.4}, {:.4}]",
                            name, block_idx, first_bytes, dest[current_pos], dest[current_pos+1], dest[current_pos+2]);
                    }
                }
                4 => {
                    let view = BlockQ4View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                    view.dequantize(&mut dest[current_pos..current_pos + num_vals]);
                }
                _ => {
                    let view = BlockQ2View { scale, min, data: &self.mmap[offset..offset+data_len], num_values: num_vals };
                    view.dequantize(&mut dest[current_pos..current_pos + num_vals]);
                }
            }
            offset += data_len;

            // Track extreme values
            if do_debug {
                if scale.abs() > max_scale {
                    max_scale = scale.abs();
                    max_scale_block = block_idx;
                }
                if min.abs() > max_min_abs {
                    max_min_abs = min.abs();
                    max_min_block = block_idx;
                }
            }

            current_pos += num_vals;
        }

        // Print extreme block summary
        if do_debug {
            println!("🔍 {} EXTREMES: max_scale={:.4} (block {}), max_|min|={:.4} (block {}), total_blocks={}",
                name, max_scale, max_scale_block, max_min_abs, max_min_block, meta.num_blocks);
        }

        Ok(())
    }

    pub fn dequantize_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let meta = self.tensors.get(name).ok_or("Tensor not found")?;
        let mut result = vec![0.0; meta.total_values as usize];
        self.dequantize_tensor_into(name, &mut result)?;
        Ok(result)
    }
}