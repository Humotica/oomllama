//! GGUF to OOM Converter
//! Convert any GGUF model to OomLlama's .oom format
//!
//! GGUF Format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//! OOM Format: /srv/jtel-stack/sandbox/ai/codex/oomllama_format_spec.md

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write, Seek, SeekFrom};
use std::path::Path;
use memmap2::Mmap;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// GGUF Magic and constants
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const OOM_MAGIC: &[u8; 4] = b"OOML";
const OOM_VERSION: u32 = 1;
const BLOCK_SIZE: usize = 256;

// GGUF Tensor Types (from ggml.h)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 29,
    Unknown = 255,
}

impl From<u32> for GgmlType {
    fn from(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            29 => GgmlType::BF16,
            _ => GgmlType::Unknown,
        }
    }
}

// GGUF Value Types
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

#[derive(Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

pub struct GgufReader {
    mmap: Mmap,
    pub header: GgufHeader,
    pub tensors: Vec<GgufTensorInfo>,
    pub tensor_data_offset: usize,
    pub metadata: HashMap<String, String>,
}

impl GgufReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        println!("📖 Reading GGUF file ({:.2} GB)...", mmap.len() as f64 / 1e9);

        let mut offset = 0;

        // Read magic
        let magic = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
        if magic != GGUF_MAGIC {
            return Err(format!("Invalid GGUF magic: 0x{:08X} (expected 0x{:08X})", magic, GGUF_MAGIC).into());
        }
        offset += 4;

        // Read version
        let version = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
        offset += 4;
        println!("   Version: {}", version);

        // Read counts
        let tensor_count = u64::from_le_bytes(mmap[offset..offset+8].try_into()?);
        offset += 8;
        let metadata_kv_count = u64::from_le_bytes(mmap[offset..offset+8].try_into()?);
        offset += 8;

        println!("   Tensors: {}", tensor_count);
        println!("   Metadata entries: {}", metadata_kv_count);

        let header = GgufHeader { version, tensor_count, metadata_kv_count };

        // Skip metadata (we don't need it for conversion, but parse to find tensor info)
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            // Read key
            let key_len = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
            offset += 8;
            let key = String::from_utf8_lossy(&mmap[offset..offset+key_len]).to_string();
            offset += key_len;

            // Read value type
            let value_type = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
            offset += 4;

            // Skip value based on type
            match value_type {
                0 => offset += 1,  // uint8
                1 => offset += 1,  // int8
                2 => offset += 2,  // uint16
                3 => offset += 2,  // int16
                4 => offset += 4,  // uint32
                5 => offset += 4,  // int32
                6 => offset += 4,  // float32
                7 => offset += 1,  // bool
                8 => {  // string
                    let str_len = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
                    offset += 8;
                    let value = String::from_utf8_lossy(&mmap[offset..offset+str_len]).to_string();
                    offset += str_len;
                    metadata.insert(key.clone(), value);
                }
                9 => {  // array
                    let arr_type = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
                    offset += 4;
                    let arr_len = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
                    offset += 8;
                    // Skip array elements
                    let elem_size = match arr_type {
                        0 | 1 | 7 => 1,
                        2 | 3 => 2,
                        4 | 5 | 6 => 4,
                        10 | 11 | 12 => 8,
                        8 => {
                            // Array of strings - need to skip each
                            for _ in 0..arr_len {
                                let slen = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
                                offset += 8 + slen;
                            }
                            0
                        }
                        _ => 0,
                    };
                    if elem_size > 0 {
                        offset += arr_len * elem_size;
                    }
                }
                10 | 11 | 12 => offset += 8,  // uint64, int64, float64
                _ => {}
            }
        }

        // Read tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            // Read name
            let name_len = u64::from_le_bytes(mmap[offset..offset+8].try_into()?) as usize;
            offset += 8;
            let name = String::from_utf8_lossy(&mmap[offset..offset+name_len]).to_string();
            offset += name_len;

            // Read dimensions
            let n_dims = u32::from_le_bytes(mmap[offset..offset+4].try_into()?);
            offset += 4;

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(u64::from_le_bytes(mmap[offset..offset+8].try_into()?));
                offset += 8;
            }

            // Read type
            let dtype = GgmlType::from(u32::from_le_bytes(mmap[offset..offset+4].try_into()?));
            offset += 4;

            // Read offset
            let tensor_offset = u64::from_le_bytes(mmap[offset..offset+8].try_into()?);
            offset += 8;

            tensors.push(GgufTensorInfo {
                name,
                n_dims,
                dims,
                dtype,
                offset: tensor_offset,
            });
        }

        // Align to 32 bytes for tensor data
        let tensor_data_offset = (offset + 31) & !31;

        println!("   Tensor data starts at: 0x{:X}", tensor_data_offset);

        Ok(Self { mmap, header, tensors, tensor_data_offset, metadata })
    }

    /// Get total number of elements in a tensor
    fn tensor_elements(&self, info: &GgufTensorInfo) -> u64 {
        info.dims.iter().product()
    }

    /// Dequantize a tensor to FP32
    pub fn dequantize_tensor(&self, info: &GgufTensorInfo) -> Result<Vec<f32>> {
        let n_elements = self.tensor_elements(info) as usize;
        let data_offset = self.tensor_data_offset + info.offset as usize;

        let mut result = vec![0.0f32; n_elements];

        match info.dtype {
            GgmlType::F32 => {
                // Direct copy
                for i in 0..n_elements {
                    let off = data_offset + i * 4;
                    result[i] = f32::from_le_bytes(self.mmap[off..off+4].try_into()?);
                }
            }
            GgmlType::F16 => {
                // FP16 to FP32
                for i in 0..n_elements {
                    let off = data_offset + i * 2;
                    let bits = u16::from_le_bytes(self.mmap[off..off+2].try_into()?);
                    result[i] = f16_to_f32(bits);
                }
            }
            GgmlType::BF16 => {
                // BF16 to FP32
                for i in 0..n_elements {
                    let off = data_offset + i * 2;
                    let bits = u16::from_le_bytes(self.mmap[off..off+2].try_into()?);
                    result[i] = bf16_to_f32(bits);
                }
            }
            GgmlType::Q4_0 => {
                dequant_q4_0(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q4_K | GgmlType::Q4_1 => {
                dequant_q4_k(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q8_0 => {
                dequant_q8_0(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q6_K => {
                dequant_q6_k(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q5_K => {
                dequant_q5_k(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q2_K => {
                dequant_q2_k(&self.mmap[data_offset..], &mut result)?;
            }
            GgmlType::Q3_K => {
                dequant_q3_k(&self.mmap[data_offset..], &mut result)?;
            }
            _ => {
                return Err(format!("Unsupported tensor type: {:?}", info.dtype).into());
            }
        }

        Ok(result)
    }
}

// FP16 to FP32 conversion
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormal
        let mut m = mant;
        let mut e = -14i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) << 23;
        let f32_mant = m << 13;
        f32::from_bits((sign << 31) | f32_exp | f32_mant)
    } else if exp == 31 {
        // Inf/NaN
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        // Normal
        let f32_exp = ((exp - 15 + 127) as u32) << 23;
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | f32_exp | f32_mant)
    }
}

// BF16 to FP32 conversion (simple - just shift)
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// Q4_0 dequantization (block size 32)
fn dequant_q4_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let n_blocks = output.len() / block_size;
    let bytes_per_block = 2 + block_size / 2; // scale (f16) + 16 bytes of q4

    for i in 0..n_blocks {
        let block_offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[block_offset..block_offset+2].try_into()?));

        for j in 0..16 {
            let byte = data[block_offset + 2 + j];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            output[i * block_size + j * 2] = scale * q0 as f32;
            output[i * block_size + j * 2 + 1] = scale * q1 as f32;
        }
    }
    Ok(())
}

// Q4_K dequantization (block size 256, super-blocks)
// Based on llama.cpp dequantize_row_q4_K
fn dequant_q4_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;
    let n_super_blocks = output.len() / super_block_size;

    // Q4_K structure: d (f16), dmin (f16), scales (12 bytes), qs (128 bytes) = 144 bytes
    let bytes_per_super_block = 144;

    for sb in 0..n_super_blocks {
        let sb_offset = sb * bytes_per_super_block;
        let d = f16_to_f32(u16::from_le_bytes(data[sb_offset..sb_offset+2].try_into()?));
        let dmin = f16_to_f32(u16::from_le_bytes(data[sb_offset+2..sb_offset+4].try_into()?));

        // Unpack scales and mins from 12 bytes for 8 sub-blocks
        // Format: scales[0-3] in low 6 bits, scales[4-7] in next bytes' low 6 bits
        //         mins[0-3] in low 6 bits offset, mins[4-7] in next
        //         High 2 bits stored separately
        let scales_bytes = &data[sb_offset + 4..sb_offset + 16];
        let qs_offset = sb_offset + 16;

        // Decode scales and mins (simplified but more accurate than before)
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // First 4 scales and mins from first 6 bytes
        scales[0] = scales_bytes[0] & 0x3F;
        scales[1] = scales_bytes[1] & 0x3F;
        scales[2] = scales_bytes[2] & 0x3F;
        scales[3] = scales_bytes[3] & 0x3F;

        mins[0] = scales_bytes[4] & 0x3F;
        mins[1] = scales_bytes[5] & 0x3F;
        mins[2] = scales_bytes[6] & 0x3F;
        mins[3] = scales_bytes[7] & 0x3F;

        // Last 4 scales and mins with high bits from bytes 8-11
        // Reference: get_scale_min_k4 from llama.cpp
        // For j>=4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        //           mn = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
        scales[4] = (scales_bytes[8] & 0x0F) | ((scales_bytes[0] >> 6) << 4);
        scales[5] = (scales_bytes[9] & 0x0F) | ((scales_bytes[1] >> 6) << 4);
        scales[6] = (scales_bytes[10] & 0x0F) | ((scales_bytes[2] >> 6) << 4);
        scales[7] = (scales_bytes[11] & 0x0F) | ((scales_bytes[3] >> 6) << 4);

        mins[4] = (scales_bytes[8] >> 4) | ((scales_bytes[4] >> 6) << 4);
        mins[5] = (scales_bytes[9] >> 4) | ((scales_bytes[5] >> 6) << 4);
        mins[6] = (scales_bytes[10] >> 4) | ((scales_bytes[6] >> 6) << 4);
        mins[7] = (scales_bytes[11] >> 4) | ((scales_bytes[7] >> 6) << 4);

        // Follow llama.cpp dequantize_row_q4_K exactly:
        // 4 iterations of 64 values each (2 sub-blocks per iteration)
        // Low nibbles (& 0x0F) go to first 32 positions, high nibbles (>> 4) go to next 32
        let mut q_offset = 0usize;
        for j in 0..4usize {
            let sc_lo = scales[2 * j] as f32;
            let sc_hi = scales[2 * j + 1] as f32;
            let m_lo = mins[2 * j] as f32;
            let m_hi = mins[2 * j + 1] as f32;

            for l in 0..32usize {
                let byte = data[qs_offset + q_offset + l];
                let lo = (byte & 0x0F) as f32;
                let hi = (byte >> 4) as f32;
                output[sb * super_block_size + j * 64 + l] = d * sc_lo * lo - dmin * m_lo;
                output[sb * super_block_size + j * 64 + l + 32] = d * sc_hi * hi - dmin * m_hi;
            }
            q_offset += 32;
        }
    }
    Ok(())
}

// Q8_0 dequantization (block size 32)
fn dequant_q8_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let n_blocks = output.len() / block_size;
    let bytes_per_block = 2 + block_size; // scale (f16) + 32 bytes of q8

    for i in 0..n_blocks {
        let block_offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[block_offset..block_offset+2].try_into()?));

        for j in 0..block_size {
            let q = data[block_offset + 2 + j] as i8;
            output[i * block_size + j] = scale * q as f32;
        }
    }
    Ok(())
}

// Q6_K dequantization (256 values per block)
// Based on llama.cpp block_q6_K structure:
// - ql (128 bytes): lower 4 bits (2 values per byte)
// - qh (64 bytes): upper 2 bits (4 values per byte)
// - scales (16 bytes): 8 sub-block scales
// - d (2 bytes fp16): super-block scale
fn dequant_q6_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;
    let n_blocks = output.len() / super_block_size;
    let bytes_per_block = 210; // 128 + 64 + 16 + 2

    for blk in 0..n_blocks {
        let offset = blk * bytes_per_block;

        // Read components (llama.cpp layout: ql, qh, scales, d)
        let ql = &data[offset..offset + 128];        // lower 4 bits
        let qh = &data[offset + 128..offset + 192];  // upper 2 bits
        let scales = &data[offset + 192..offset + 208]; // sub-block scales (int8)
        let d = f16_to_f32(u16::from_le_bytes(data[offset + 208..offset + 210].try_into()?));

        // Decode 256 values using llama.cpp interleaved layout:
        // 2 groups of 128, each group has 4 sub-groups of 32 values.
        // ql advances by 64 per group, qh by 32, scales by 8.
        let mut ql_off: usize = 0;
        let mut qh_off: usize = 0;
        let mut sc_off: usize = 0;
        let mut y_off: usize = blk * super_block_size;

        for _n in (0..super_block_size).step_by(128) {
            for l in 0..32usize {
                let is = l / 16; // 0 or 1

                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                let sc0 = scales[sc_off + is + 0] as i8 as f32;
                let sc2 = scales[sc_off + is + 2] as i8 as f32;
                let sc4 = scales[sc_off + is + 4] as i8 as f32;
                let sc6 = scales[sc_off + is + 6] as i8 as f32;

                output[y_off + l +  0] = d * sc0 * q1 as f32;
                output[y_off + l + 32] = d * sc2 * q2 as f32;
                output[y_off + l + 64] = d * sc4 * q3 as f32;
                output[y_off + l + 96] = d * sc6 * q4 as f32;
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            y_off += 128;
        }
    }
    Ok(())
}

// Q5_K dequantization (simplified)
fn dequant_q5_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;
    let n_blocks = output.len() / super_block_size;
    let bytes_per_block = 176;

    for i in 0..n_blocks {
        let offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[offset..offset+2].try_into().unwrap_or([0,0])));

        for j in 0..super_block_size {
            let q = ((data.get(offset + 4 + j / 2).unwrap_or(&0) >> ((j % 2) * 4)) & 0x1F) as i8 - 16;
            output[i * super_block_size + j] = scale * q as f32;
        }
    }
    Ok(())
}

// Q2_K dequantization (simplified)
fn dequant_q2_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;
    let n_blocks = output.len() / super_block_size;
    let bytes_per_block = 84;

    for i in 0..n_blocks {
        let offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[offset..offset+2].try_into().unwrap_or([0,0])));

        for j in 0..super_block_size {
            let byte_idx = j / 4;
            let bit_shift = (j % 4) * 2;
            let q = ((data.get(offset + 4 + byte_idx).unwrap_or(&0) >> bit_shift) & 0x03) as i8 - 2;
            output[i * super_block_size + j] = scale * q as f32;
        }
    }
    Ok(())
}

// Q3_K dequantization (simplified)
fn dequant_q3_k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let super_block_size = 256;
    let n_blocks = output.len() / super_block_size;
    let bytes_per_block = 110;

    for i in 0..n_blocks {
        let offset = i * bytes_per_block;
        let scale = f16_to_f32(u16::from_le_bytes(data[offset..offset+2].try_into().unwrap_or([0,0])));

        for j in 0..super_block_size {
            let q = ((data.get(offset + 4 + j / 3).unwrap_or(&0) >> ((j % 3) * 2)) & 0x07) as i8 - 4;
            output[i * super_block_size + j] = scale * q as f32;
        }
    }
    Ok(())
}

/// OOM Writer - Write tensors in OomLlama format
pub struct OomWriter {
    writer: BufWriter<File>,
    tensor_count: u32,
    tensor_data: Vec<u8>,
}

impl OomWriter {
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer,
            tensor_count: 0,
            tensor_data: Vec::new(),
        })
    }

    /// Write header (called after all tensors added)
    fn write_header(&mut self) -> Result<()> {
        // Magic
        self.writer.write_all(OOM_MAGIC)?;
        // Version
        self.writer.write_all(&OOM_VERSION.to_le_bytes())?;
        // Tensor count
        self.writer.write_all(&self.tensor_count.to_le_bytes())?;
        Ok(())
    }

    /// Add a tensor (quantizes to Q2)
    pub fn add_tensor_q2(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let num_values = values.len();
        let num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Write tensor header
        // name_len (u32) + name + quant_type (u8) + num_blocks (u32) + total_values (u32)
        let name_bytes = name.as_bytes();
        self.tensor_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(name_bytes);
        self.tensor_data.push(2); // Q2
        self.tensor_data.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(&(num_values as u32).to_le_bytes());

        // Write blocks
        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(num_values);
            let block_values = &values[start..end];

            // Find min/max
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for &v in block_values {
                if v < min { min = v; }
                if v > max { max = v; }
            }

            let range = max - min;
            let scale = if range.abs() < 1e-9 { 0.0 } else { range / 3.0 };

            // Quantize
            let mut qdata = Vec::with_capacity((block_values.len() + 3) / 4);
            let mut current_byte: u8 = 0;
            let mut shift = 0;

            for &v in block_values {
                let q = if scale == 0.0 { 0 } else {
                    let norm = (v - min) / scale;
                    (norm.round() as u8).min(3)
                };
                current_byte |= q << shift;
                shift += 2;
                if shift == 8 {
                    qdata.push(current_byte);
                    current_byte = 0;
                    shift = 0;
                }
            }
            if shift > 0 { qdata.push(current_byte); }

            // Write block: scale (f32) + min (f32) + data_len (u32) + qdata
            self.tensor_data.extend_from_slice(&scale.to_le_bytes());
            self.tensor_data.extend_from_slice(&min.to_le_bytes());
            self.tensor_data.extend_from_slice(&(qdata.len() as u32).to_le_bytes());
            self.tensor_data.extend_from_slice(&qdata);
        }

        self.tensor_count += 1;
        Ok(())
    }

    /// Add a tensor (quantizes to Q8 - much higher quality than Q2!)
    /// Q8: 256 levels per value vs Q2's 4 levels
    /// Formula: q = round((value - min) / scale * 255)
    pub fn add_tensor_q8(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let num_values = values.len();
        let num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Write tensor header
        let name_bytes = name.as_bytes();
        self.tensor_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(name_bytes);
        self.tensor_data.push(8); // Q8
        self.tensor_data.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(&(num_values as u32).to_le_bytes());

        // Write blocks
        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(num_values);
            let block_values = &values[start..end];

            // Find min/max
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for &v in block_values {
                if v < min { min = v; }
                if v > max { max = v; }
            }

            let range = max - min;
            let scale = if range.abs() < 1e-9 { 0.0 } else { range / 255.0 };

            // Quantize to Q8 (one byte per value)
            let mut qdata = Vec::with_capacity(block_values.len());
            for &v in block_values {
                let q = if scale == 0.0 { 0 } else {
                    let norm = (v - min) / scale;
                    (norm.round() as u8)
                };
                qdata.push(q);
            }

            // Write block: scale (f32) + min (f32) + data_len (u32) + qdata
            self.tensor_data.extend_from_slice(&scale.to_le_bytes());
            self.tensor_data.extend_from_slice(&min.to_le_bytes());
            self.tensor_data.extend_from_slice(&(qdata.len() as u32).to_le_bytes());
            self.tensor_data.extend_from_slice(&qdata);
        }

        self.tensor_count += 1;
        Ok(())
    }

    /// Add a tensor (quantizes to Q4 - 16 levels, 4 bits per value)
    /// Good balance between size and quality
    pub fn add_tensor_q4(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let num_values = values.len();
        let num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Write tensor header
        let name_bytes = name.as_bytes();
        self.tensor_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(name_bytes);
        self.tensor_data.push(4); // Q4
        self.tensor_data.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(&(num_values as u32).to_le_bytes());

        // Write blocks
        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(num_values);
            let block_values = &values[start..end];

            // Find min/max
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for &v in block_values {
                if v < min { min = v; }
                if v > max { max = v; }
            }

            let range = max - min;
            let scale = if range.abs() < 1e-9 { 0.0 } else { range / 15.0 };

            // Quantize to Q4 (4 bits per value, 2 values per byte)
            let mut qdata = Vec::with_capacity((block_values.len() + 1) / 2);
            let mut current_byte: u8 = 0;
            let mut shift = 0;

            for &v in block_values {
                let q = if scale == 0.0 { 0 } else {
                    let norm = (v - min) / scale;
                    (norm.round() as u8).min(15)
                };
                current_byte |= q << shift;
                shift += 4;
                if shift == 8 {
                    qdata.push(current_byte);
                    current_byte = 0;
                    shift = 0;
                }
            }
            if shift > 0 { qdata.push(current_byte); }

            // Write block: scale (f32) + min (f32) + data_len (u32) + qdata
            self.tensor_data.extend_from_slice(&scale.to_le_bytes());
            self.tensor_data.extend_from_slice(&min.to_le_bytes());
            self.tensor_data.extend_from_slice(&(qdata.len() as u32).to_le_bytes());
            self.tensor_data.extend_from_slice(&qdata);
        }

        self.tensor_count += 1;
        Ok(())
    }

    /// Add a tensor in F32 (no quantization - lossless)
    /// Used for norm weights, biases, and other small precision-sensitive tensors
    pub fn add_tensor_f32(&mut self, name: &str, values: &[f32]) -> Result<()> {
        let num_values = values.len();
        let num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Write tensor header
        let name_bytes = name.as_bytes();
        self.tensor_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(name_bytes);
        self.tensor_data.push(0); // F32 = quant_type 0
        self.tensor_data.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        self.tensor_data.extend_from_slice(&(num_values as u32).to_le_bytes());

        // Write blocks (same structure for consistency: scale + min + data_len + data)
        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(num_values);
            let block_values = &values[start..end];

            // Scale=1.0, min=0.0 (unused for F32, but keeps block format consistent)
            let scale: f32 = 1.0;
            let min: f32 = 0.0;

            // Data is raw f32 bytes: 4 bytes per value
            let data_len = block_values.len() * 4;
            self.tensor_data.extend_from_slice(&scale.to_le_bytes());
            self.tensor_data.extend_from_slice(&min.to_le_bytes());
            self.tensor_data.extend_from_slice(&(data_len as u32).to_le_bytes());
            for &v in block_values {
                self.tensor_data.extend_from_slice(&v.to_le_bytes());
            }
        }

        self.tensor_count += 1;
        Ok(())
    }

    /// Finalize and write the file
    pub fn finish(mut self) -> Result<()> {
        self.write_header()?;
        self.writer.write_all(&self.tensor_data)?;
        self.writer.flush()?;
        Ok(())
    }
}

/// Map GGUF tensor names to HuggingFace/Candle format
fn map_tensor_name(gguf_name: &str) -> String {
    // Global tensors
    if gguf_name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if gguf_name == "output.weight" {
        return "lm_head.weight".to_string();
    }
    if gguf_name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }

    // Layer tensors: blk.N.* -> model.layers.N.*
    if gguf_name.starts_with("blk.") {
        // Extract layer number
        let parts: Vec<&str> = gguf_name.split('.').collect();
        if parts.len() >= 3 {
            let layer_num = parts[1];
            let rest = &parts[2..].join(".");

            let mapped = match rest.as_str() {
                // Attention
                "attn_norm.weight" => format!("model.layers.{}.input_layernorm.weight", layer_num),
                "attn_q.weight" => format!("model.layers.{}.self_attn.q_proj.weight", layer_num),
                "attn_k.weight" => format!("model.layers.{}.self_attn.k_proj.weight", layer_num),
                "attn_v.weight" => format!("model.layers.{}.self_attn.v_proj.weight", layer_num),
                "attn_output.weight" => format!("model.layers.{}.self_attn.o_proj.weight", layer_num),
                "attn_q.bias" => format!("model.layers.{}.self_attn.q_proj.bias", layer_num),
                "attn_k.bias" => format!("model.layers.{}.self_attn.k_proj.bias", layer_num),
                "attn_v.bias" => format!("model.layers.{}.self_attn.v_proj.bias", layer_num),
                // FFN / MLP
                "ffn_norm.weight" => format!("model.layers.{}.post_attention_layernorm.weight", layer_num),
                "ffn_gate.weight" => format!("model.layers.{}.mlp.gate_proj.weight", layer_num),
                "ffn_up.weight" => format!("model.layers.{}.mlp.up_proj.weight", layer_num),
                "ffn_down.weight" => format!("model.layers.{}.mlp.down_proj.weight", layer_num),
                // Fallback
                _ => return gguf_name.to_string(),
            };
            return mapped;
        }
    }

    // Fallback: return as-is
    gguf_name.to_string()
}

/// Quantization level for OOM output
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OomQuantLevel {
    Q2, // 4 levels - smallest but low quality
    Q4, // 16 levels - good balance
    Q8, // 256 levels - highest quality
}

impl Default for OomQuantLevel {
    fn default() -> Self { Self::Q8 }
}

/// Convert GGUF to OOM with configurable quantization
pub fn convert_gguf_to_oom_with_quant<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    quant_level: OomQuantLevel,
    progress_callback: Option<Box<dyn Fn(usize, usize, &str)>>,
) -> Result<()> {
    let reader = GgufReader::open(&input_path)?;
    let mut writer = OomWriter::create(&output_path)?;

    let total_tensors = reader.tensors.len();
    let quant_name = match quant_level {
        OomQuantLevel::Q2 => "Q2 (4 levels)",
        OomQuantLevel::Q4 => "Q4 (16 levels)",
        OomQuantLevel::Q8 => "Q8 (256 levels)",
    };
    println!("\n🔄 Converting {} tensors to OOM {} format (norm/bias tensors kept as F32)...", total_tensors, quant_name);

    for (idx, tensor_info) in reader.tensors.iter().enumerate() {
        // Map GGUF tensor name to HuggingFace/Candle format
        let mapped_name = map_tensor_name(&tensor_info.name);

        if let Some(ref cb) = progress_callback {
            cb(idx + 1, total_tensors, &mapped_name);
        } else {
            print!("\r   [{}/{}] {}                    ", idx + 1, total_tensors, &mapped_name);
            std::io::stdout().flush()?;
        }

        // Dequantize to FP32
        let fp32_values = reader.dequantize_tensor(tensor_info)?;

        // Detect norm weights and biases -> keep as F32 (small, precision-sensitive)
        let is_norm = mapped_name.contains("layernorm") || mapped_name.contains("norm.weight")
            || mapped_name.ends_with(".bias");
        let is_small = fp32_values.len() < 65536; // < 64K values = small tensor

        if is_norm || is_small {
            // Norm/bias/small tensors: keep full F32 precision
            writer.add_tensor_f32(&mapped_name, &fp32_values)?;
        } else {
            // Large weight tensors: quantize as requested
            match quant_level {
                OomQuantLevel::Q2 => writer.add_tensor_q2(&mapped_name, &fp32_values)?,
                OomQuantLevel::Q4 => writer.add_tensor_q4(&mapped_name, &fp32_values)?,
                OomQuantLevel::Q8 => writer.add_tensor_q8(&mapped_name, &fp32_values)?,
            }
        }
    }

    println!("\n✅ Writing OOM file...");
    writer.finish()?;

    // Print size comparison
    let input_size = std::fs::metadata(&input_path)?.len();
    let output_size = std::fs::metadata(&output_path)?.len();
    let ratio = input_size as f64 / output_size as f64;

    println!("\n📊 Conversion complete!");
    println!("   Input:  {:.2} GB (GGUF)", input_size as f64 / 1e9);
    println!("   Output: {:.2} GB (OOM {})", output_size as f64 / 1e9, quant_name);
    println!("   Ratio:  {:.1}x smaller", ratio);

    Ok(())
}

/// Convert GGUF to OOM (Q8 by default for best quality)
pub fn convert_gguf_to_oom<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    progress_callback: Option<Box<dyn Fn(usize, usize, &str)>>,
) -> Result<()> {
    convert_gguf_to_oom_with_quant(input_path, output_path, OomQuantLevel::Q8, progress_callback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // Test some known values
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001); // 1.0
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 0.001); // 0.0
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.001); // -1.0
    }
}
