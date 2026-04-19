//! OomLlama Turbo - High Performance Native Inference
//!
//! Target: 3 min/token → 1 sec/token (180x speedup!)
//!
//! Optimizations:
//! 1. KV-Cache (10-50x) - Cache attention keys/values
//! 2. Layer Pinning (5-10x) - Hot layers in VRAM
//! 3. Async Prefetching (2-3x) - Background layer loading
//! 4. CUDA Streams (1.5-2x) - Overlap compute/transfer
//! 5. Flash Attention 2 (1.3-1.5x) - Memory-efficient attention
//!
//! One love, one fAmIly! 🦙🚀

use candle_core::{Device, Tensor, DType, Shape, D};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::thread;
use std::sync::mpsc::{channel, Sender, Receiver};

// ============================================================================
// KV-CACHE: Cache Key/Value tensors to avoid recomputation
// ============================================================================

/// KV-Cache for a single attention layer
#[derive(Clone)]
pub struct LayerKVCache {
    /// Cached key tensor [batch, n_kv_heads, seq_len, head_dim]
    pub k: Option<Tensor>,
    /// Cached value tensor [batch, n_kv_heads, seq_len, head_dim]
    pub v: Option<Tensor>,
    /// Current sequence length in cache
    pub seq_len: usize,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self { k: None, v: None, seq_len: 0 }
    }

    /// Append new K/V to cache
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> candle_core::Result<()> {
        match (&self.k, &self.v) {
            (Some(old_k), Some(old_v)) => {
                // Concatenate along sequence dimension (dim 2)
                self.k = Some(Tensor::cat(&[old_k, new_k], 2)?);
                self.v = Some(Tensor::cat(&[old_v, new_v], 2)?);
            }
            _ => {
                self.k = Some(new_k.clone());
                self.v = Some(new_v.clone());
            }
        }
        self.seq_len += new_k.dim(2)?;
        Ok(())
    }

    /// Get cached K/V for attention computation
    pub fn get(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Clear cache (for new sequence)
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }
}

/// Full model KV-Cache
pub struct ModelKVCache {
    layers: Vec<LayerKVCache>,
    max_seq_len: usize,
}

impl ModelKVCache {
    pub fn new(n_layers: usize, max_seq_len: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerKVCache::new()).collect(),
            max_seq_len,
        }
    }

    pub fn get_layer(&self, layer_idx: usize) -> &LayerKVCache {
        &self.layers[layer_idx]
    }

    pub fn get_layer_mut(&mut self, layer_idx: usize) -> &mut LayerKVCache {
        &mut self.layers[layer_idx]
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }
}

// ============================================================================
// LAYER PINNING: Keep hot layers in VRAM
// ============================================================================

/// Strategy for which layers to pin in VRAM
#[derive(Clone, Debug)]
pub enum PinStrategy {
    /// Pin first N and last M layers
    FirstLast { first: usize, last: usize },
    /// Pin every Nth layer
    Strided { stride: usize },
    /// Pin specific layer indices
    Specific(Vec<usize>),
    /// Pin layers based on importance scores
    Importance(Vec<f32>),
}

/// Pinned layer storage
pub struct LayerPin {
    /// Pinned tensors by layer index
    pinned: HashMap<usize, HashMap<String, Tensor>>,
    /// Total VRAM budget (bytes)
    vram_budget: usize,
    /// Current VRAM usage
    vram_used: usize,
    /// Pin strategy
    strategy: PinStrategy,
}

impl LayerPin {
    pub fn new(vram_budget_gb: f32, strategy: PinStrategy) -> Self {
        Self {
            pinned: HashMap::new(),
            vram_budget: (vram_budget_gb * 1e9) as usize,
            vram_used: 0,
            strategy,
        }
    }

    /// Check if layer should be pinned based on strategy
    pub fn should_pin(&self, layer_idx: usize, n_layers: usize) -> bool {
        match &self.strategy {
            PinStrategy::FirstLast { first, last } => {
                layer_idx < *first || layer_idx >= n_layers - *last
            }
            PinStrategy::Strided { stride } => layer_idx % stride == 0,
            PinStrategy::Specific(indices) => indices.contains(&layer_idx),
            PinStrategy::Importance(scores) => {
                // Pin if importance score > 0.5
                scores.get(layer_idx).map(|s| *s > 0.5).unwrap_or(false)
            }
        }
    }

    /// Pin a tensor if budget allows
    pub fn pin(&mut self, layer_idx: usize, name: &str, tensor: Tensor) -> bool {
        let size = tensor.elem_count() * tensor.dtype().size_in_bytes();

        if self.vram_used + size > self.vram_budget {
            return false;
        }

        self.vram_used += size;
        self.pinned
            .entry(layer_idx)
            .or_insert_with(HashMap::new)
            .insert(name.to_string(), tensor);
        true
    }

    /// Get pinned tensor if available
    pub fn get(&self, layer_idx: usize, name: &str) -> Option<&Tensor> {
        self.pinned.get(&layer_idx)?.get(name)
    }

    /// Check if tensor is pinned
    pub fn is_pinned(&self, layer_idx: usize, name: &str) -> bool {
        self.pinned.get(&layer_idx)
            .map(|m| m.contains_key(name))
            .unwrap_or(false)
    }

    pub fn vram_usage_gb(&self) -> f32 {
        self.vram_used as f32 / 1e9
    }
}

// ============================================================================
// ASYNC PREFETCHING: Load next layer while computing current
// ============================================================================

/// Message for prefetch worker
pub enum PrefetchMsg {
    /// Load tensor from OOM file
    Load { layer_idx: usize, tensor_name: String },
    /// Stop worker
    Stop,
}

/// Prefetched tensor result
pub struct PrefetchResult {
    pub layer_idx: usize,
    pub tensor_name: String,
    pub tensor: Result<Tensor, String>,
}

/// Async prefetcher using background thread
pub struct AsyncPrefetcher {
    sender: Sender<PrefetchMsg>,
    receiver: Receiver<PrefetchResult>,
    /// Prefetch queue (layers ahead to prefetch)
    lookahead: usize,
}

impl AsyncPrefetcher {
    pub fn new<F>(loader_fn: F, lookahead: usize) -> Self
    where
        F: Fn(usize, &str) -> Result<Tensor, String> + Send + 'static
    {
        let (tx_cmd, rx_cmd) = channel::<PrefetchMsg>();
        let (tx_result, rx_result) = channel::<PrefetchResult>();

        // Spawn prefetch worker
        thread::spawn(move || {
            loop {
                match rx_cmd.recv() {
                    Ok(PrefetchMsg::Load { layer_idx, tensor_name }) => {
                        let tensor = loader_fn(layer_idx, &tensor_name);
                        let _ = tx_result.send(PrefetchResult {
                            layer_idx,
                            tensor_name,
                            tensor,
                        });
                    }
                    Ok(PrefetchMsg::Stop) | Err(_) => break,
                }
            }
        });

        Self {
            sender: tx_cmd,
            receiver: rx_result,
            lookahead,
        }
    }

    /// Request prefetch of layer tensors
    pub fn prefetch(&self, layer_idx: usize, tensor_names: &[&str]) {
        for name in tensor_names {
            let _ = self.sender.send(PrefetchMsg::Load {
                layer_idx,
                tensor_name: name.to_string(),
            });
        }
    }

    /// Try to get prefetched tensor (non-blocking)
    pub fn try_get(&self) -> Option<PrefetchResult> {
        self.receiver.try_recv().ok()
    }

    /// Get prefetched tensor (blocking)
    pub fn get(&self) -> Option<PrefetchResult> {
        self.receiver.recv().ok()
    }
}

impl Drop for AsyncPrefetcher {
    fn drop(&mut self) {
        let _ = self.sender.send(PrefetchMsg::Stop);
    }
}

// ============================================================================
// FLASH ATTENTION 2: Memory-efficient attention
// ============================================================================

/// Flash Attention parameters
pub struct FlashAttentionConfig {
    pub block_size: usize,      // Typically 64 or 128
    pub causal: bool,           // Use causal mask
    pub softmax_scale: Option<f32>, // Custom scale, defaults to 1/sqrt(head_dim)
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            causal: true,
            softmax_scale: None,
        }
    }
}

/// Compute attention with Flash Attention 2 algorithm
///
/// This is a software implementation - for true performance,
/// we need the CUDA kernel from flash-attn crate
pub fn flash_attention_forward(
    q: &Tensor,  // [batch, n_heads, seq_len, head_dim]
    k: &Tensor,  // [batch, n_kv_heads, seq_len, head_dim]
    v: &Tensor,  // [batch, n_kv_heads, seq_len, head_dim]
    config: &FlashAttentionConfig,
) -> candle_core::Result<Tensor> {
    let (batch, n_heads, q_seq_len, head_dim) = q.dims4()?;
    let kv_seq_len = k.dim(2)?;
    let n_kv_heads = k.dim(1)?;

    // Handle grouped query attention (GQA) - make contiguous for matmul
    // GQA: repeat_interleave KV heads to match Q heads
    // k.repeat() TILES [k0,k1,k2,k3, k0,k1,...] which is WRONG for GQA
    // We need repeat_interleave [k0,k0,...,k0, k1,k1,...,k1, ...] 
    let k = if n_kv_heads != n_heads {
        let repeat_factor = n_heads / n_kv_heads;
        // [batch, n_kv_heads, seq, dim] -> [batch, n_kv_heads, 1, seq, dim]
        let k = k.unsqueeze(2)?;
        // expand to [batch, n_kv_heads, repeat_factor, seq, dim]
        let k = k.expand(&[batch, n_kv_heads, repeat_factor, kv_seq_len, head_dim])?;
        // reshape to [batch, n_kv_heads * repeat_factor, seq, dim]
        k.reshape(&[batch, n_heads, kv_seq_len, head_dim])?.contiguous()?
    } else {
        k.contiguous()?
    };

    let v = if n_kv_heads != n_heads {
        let repeat_factor = n_heads / n_kv_heads;
        let v = v.unsqueeze(2)?;
        let v = v.expand(&[batch, n_kv_heads, repeat_factor, kv_seq_len, head_dim])?;
        v.reshape(&[batch, n_heads, kv_seq_len, head_dim])?.contiguous()?
    } else {
        v.contiguous()?
    };

    // Scale factor
    let scale = config.softmax_scale
        .unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
    // For true Flash Attention, we'd use tiled computation
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let att = (q.contiguous()?.matmul(&k_t)? * scale as f64)?;

    // Apply causal mask if needed
    let att = if config.causal && q_seq_len > 1 {
        // Create causal mask: mask out future tokens (where j > i)
        // triu2 creates ones ON and ABOVE diagonal, we need STRICTLY ABOVE
        // So we create triu2 and then zero out the diagonal
        let mask = Tensor::triu2(q_seq_len, DType::F32, q.device())?;
        // Subtract identity to remove diagonal (keep only strictly upper triangular)
        let eye = Tensor::eye(q_seq_len, DType::F32, q.device())?;
        let mask = (mask - eye)?;
        let mask = mask.broadcast_as(att.shape())?;
        // Mask out future tokens with large negative value (avoid -inf for softmax stability)
        att.broadcast_add(&(mask * -1e9)?)?
    } else {
        att
    };

    // Softmax
    let att = candle_nn::ops::softmax_last_dim(&att)?;

    // Attention @ Values (contiguous for matmul)
    att.contiguous()?.matmul(&v.contiguous()?)
}

// ============================================================================
// RoPE FUNCTIONS (Rotary Position Embedding)
// ============================================================================

/// Compute RoPE sin/cos frequencies
pub fn compute_rope_freqs(head_dim: usize, max_seq_len: usize, rope_theta: f32, device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
    // Compute inverse frequencies: 1.0 / (theta ^ (2i / head_dim)) for i in 0..head_dim/2
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?;

    // Positions: [0, 1, 2, ..., max_seq_len-1]
    let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (max_seq_len,), device)?;

    // Outer product: positions @ inv_freq -> [max_seq_len, half_dim]
    let freqs = positions.reshape((max_seq_len, 1))?.matmul(&inv_freq.reshape((1, half_dim))?)?;

    // Compute sin/cos
    let sin = freqs.sin()?;
    let cos = freqs.cos()?;

    Ok((sin, cos))
}

/// Apply RoPE to query/key tensors (Qwen-style interleaved)
/// x: [batch, n_heads, seq_len, head_dim]
/// sin, cos: [max_seq_len, head_dim/2]
/// position_offset: current position in KV-cache
pub fn apply_rope(x: &Tensor, sin: &Tensor, cos: &Tensor, position_offset: usize) -> candle_core::Result<Tensor> {
    let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;

    // Get sin/cos for positions [offset, offset+seq_len)
    let sin = sin.narrow(0, position_offset, seq_len)?;
    let cos = cos.narrow(0, position_offset, seq_len)?;

    // For interleaved RoPE (Qwen style):
    // Even indices (0,2,4...) get cos rotation, odd indices (1,3,5...) get sin
    // Reshape x from [batch, heads, seq, dim] to [batch, heads, seq, dim/2, 2]

    // Reshape x to [batch, n_heads, seq_len, half_dim, 2]
    let x = x.reshape((batch, n_heads, seq_len, half_dim, 2))?;

    // Split into even (x0) and odd (x1) components
    let x0 = x.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?; // [batch, heads, seq, half_dim]
    let x1 = x.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

    // Reshape sin/cos for broadcasting: [1, 1, seq_len, half_dim]
    let sin = sin.reshape((1, 1, seq_len, half_dim))?;
    let cos = cos.reshape((1, 1, seq_len, half_dim))?;

    // Apply rotation:
    // new_x0 = x0 * cos - x1 * sin
    // new_x1 = x0 * sin + x1 * cos
    let new_x0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let new_x1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

    // Interleave back: stack and reshape
    let new_x0 = new_x0.unsqueeze(D::Minus1)?;
    let new_x1 = new_x1.unsqueeze(D::Minus1)?;
    let result = Tensor::cat(&[&new_x0, &new_x1], D::Minus1)?;

    // Reshape back to [batch, n_heads, seq_len, head_dim]
    result.reshape((batch, n_heads, seq_len, head_dim))
}

/// LLaMA-style RoPE (non-interleaved): split at half_dim
/// sin, cos: [max_seq_len, head_dim/2]
/// position_offset: current position in KV-cache
pub fn apply_rope_llama(x: &Tensor, sin: &Tensor, cos: &Tensor, position_offset: usize) -> candle_core::Result<Tensor> {
    let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
    let half_dim = head_dim / 2;

    // Get sin/cos for positions [offset, offset+seq_len)
    let sin = sin.narrow(0, position_offset, seq_len)?;
    let cos = cos.narrow(0, position_offset, seq_len)?;

    // LLaMA-style: split x into first half and second half
    let x0 = x.narrow(D::Minus1, 0, half_dim)?;          // First half
    let x1 = x.narrow(D::Minus1, half_dim, half_dim)?;   // Second half

    // Reshape sin/cos for broadcasting: [1, 1, seq_len, half_dim]
    let sin = sin.reshape((1, 1, seq_len, half_dim))?;
    let cos = cos.reshape((1, 1, seq_len, half_dim))?;

    // Apply rotation:
    // new_x0 = x0 * cos - x1 * sin
    // new_x1 = x0 * sin + x1 * cos
    let new_x0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let new_x1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

    // Concatenate back
    Tensor::cat(&[&new_x0, &new_x1], D::Minus1)
}

// ============================================================================
// TURBO ENGINE: Combines all optimizations
// ============================================================================

/// Configuration for turbo inference
pub struct TurboConfig {
    /// Number of transformer layers
    pub n_layers: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of KV heads (for GQA)
    pub n_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length for KV cache
    pub max_seq_len: usize,
    /// VRAM budget for layer pinning (GB)
    pub vram_budget_gb: f32,
    /// Layer pin strategy
    pub pin_strategy: PinStrategy,
    /// Prefetch lookahead
    pub prefetch_lookahead: usize,
    /// Use Flash Attention
    pub use_flash_attention: bool,
    /// RoPE theta (rotation base frequency)
    pub rope_theta: f32,
    /// RoPE format: true = interleaved (Qwen), false = split-half (LLaMA/TinyLlama)
    pub rope_interleaved: bool,
    /// Use FP16 compute (mixed precision)
    pub use_fp16: bool,
}

impl TurboConfig {
    /// Config for Qwen2.5-32B on dual RTX 3060 (24GB total)
    pub fn qwen32b_dual3060() -> Self {
        Self {
            n_layers: 64,
            hidden_size: 5120,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 8192,
            vram_budget_gb: 20.0, // Leave 4GB headroom
            pin_strategy: PinStrategy::FirstLast { first: 4, last: 4 }, // Pin first 4 + last 4 layers
            prefetch_lookahead: 2,
            use_flash_attention: true,
            rope_theta: 1000000.0,
            rope_interleaved: true,
            use_fp16: true,
        }
    }
}

/// Turbo inference engine
pub struct TurboEngine {
    config: TurboConfig,
    kv_cache: ModelKVCache,
    layer_pin: LayerPin,
    // prefetcher: Option<AsyncPrefetcher>,
    device: Device,
}

impl TurboEngine {
    pub fn new(config: TurboConfig, device: Device) -> Self {
        let kv_cache = ModelKVCache::new(config.n_layers, config.max_seq_len);
        let layer_pin = LayerPin::new(config.vram_budget_gb, config.pin_strategy.clone());

        Self {
            config,
            kv_cache,
            layer_pin,
            device,
        }
    }

    /// Process attention with KV-cache and Flash Attention
    pub fn attention_forward(
        &mut self,
        layer_idx: usize,
        x: &Tensor,
        wq: &Tensor,
        wk: &Tensor,
        wv: &Tensor,
        wo: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;

        // Get output dimensions from weight shapes: wq is [hidden, q_out], etc.
        let q_out = wq.dim(1)?;
        let kv_out = wk.dim(1)?;

        // Project Q, K, V - reshape for batched matmul: [B, S, H] -> [B*S, H]
        let x_flat = x.reshape((batch * seq_len, hidden))?;
        let q = x_flat.matmul(wq)?.reshape((batch, seq_len, q_out))?;
        let k = x_flat.matmul(wk)?.reshape((batch, seq_len, kv_out))?;
        let v = x_flat.matmul(wv)?.reshape((batch, seq_len, kv_out))?;

        // Reshape for multi-head attention (contiguous for matmul)
        let mut q = q.reshape((batch, seq_len, self.config.n_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?; // [batch, n_heads, seq_len, head_dim]
        let mut k = k.reshape((batch, seq_len, self.config.n_kv_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let v = v.reshape((batch, seq_len, self.config.n_kv_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // Apply RoPE (Rotary Position Embedding) - CRITICAL for position awareness!
        // Position is current cache length (number of tokens already processed)
        let pos_start = self.kv_cache.seq_len();
        (q, k) = self.apply_rope(&q, &k, pos_start)?;

        // Update KV cache
        let cache = self.kv_cache.get_layer_mut(layer_idx);
        cache.append(&k, &v)?;

        // Get full K, V from cache
        let (full_k, full_v) = cache.get().unwrap();

        // Apply attention (with Flash Attention if enabled)
        let att_out = if self.config.use_flash_attention {
            flash_attention_forward(
                &q,
                full_k,
                full_v,
                &FlashAttentionConfig::default(),
            )?
        } else {
            // Standard attention (make tensors contiguous)
            let scale = 1.0 / (self.config.head_dim as f32).sqrt();
            let k_t = full_k.transpose(2, 3)?.contiguous()?;
            let att = (q.contiguous()?.matmul(&k_t)? * scale as f64)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.contiguous()?.matmul(&full_v.contiguous()?)?
        };

        // Reshape and project output (contiguous after transpose)
        let hidden_out = self.config.n_heads * self.config.head_dim;
        let att_out = att_out.transpose(1, 2)?.contiguous()?
            .reshape((batch, seq_len, hidden_out))?;

        // Output projection: [B, S, H] -> [B*S, H] -> matmul -> [B, S, H]
        let att_flat = att_out.reshape((batch * seq_len, hidden_out))?;
        let o_out = wo.dim(1)?;
        let result = att_flat.matmul(wo)?.reshape((batch, seq_len, o_out))?;

        Ok(result)
    }

    /// Attention forward with optional bias support (for Qwen models)
    pub fn attention_forward_with_bias(
        &mut self,
        layer_idx: usize,
        x: &Tensor,
        wq: &Tensor,
        wk: &Tensor,
        wv: &Tensor,
        wo: &Tensor,
        bq: Option<&Tensor>,
        bk: Option<&Tensor>,
        bv: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;

        // Get output dimensions from weight shapes: wq is [hidden, q_out], etc.
        let q_out = wq.dim(1)?;
        let kv_out = wk.dim(1)?;

        // Project Q, K, V - reshape for batched matmul: [B, S, H] -> [B*S, H]
        let x_flat = x.reshape((batch * seq_len, hidden))?;

        let mut q = x_flat.matmul(wq)?;
        let mut k = x_flat.matmul(wk)?;
        let mut v = x_flat.matmul(wv)?;

        // Apply biases if present (Qwen requires this!)
        if let Some(bias) = bq {
            q = q.broadcast_add(bias)?;
        }
        if let Some(bias) = bk {
            k = k.broadcast_add(bias)?;
        }
        if let Some(bias) = bv {
            v = v.broadcast_add(bias)?;
        }

        // Reshape back to [B, S, dim]
        let q = q.reshape((batch, seq_len, q_out))?;
        let k = k.reshape((batch, seq_len, kv_out))?;
        let v = v.reshape((batch, seq_len, kv_out))?;

        // Reshape for multi-head attention (contiguous for matmul)
        let mut q = q.reshape((batch, seq_len, self.config.n_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?; // [batch, n_heads, seq_len, head_dim]
        let mut k = k.reshape((batch, seq_len, self.config.n_kv_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let v = v.reshape((batch, seq_len, self.config.n_kv_heads, self.config.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // Apply RoPE (Rotary Position Embedding)
        let pos_start = self.kv_cache.seq_len();
        (q, k) = self.apply_rope(&q, &k, pos_start)?;

        // Q and K values are large (~200) due to RMSNorm not centering the mean.
        // Standard 1/sqrt(head_dim) scaling should work with flash attention's
        // numerically stable softmax (log-sum-exp trick).
        // Do NOT scale Q and K here - let flash attention handle it.

        // Update KV cache
        let cache = self.kv_cache.get_layer_mut(layer_idx);
        cache.append(&k, &v)?;

        // Get full K, V from cache
        let (full_k, full_v) = cache.get().unwrap();

        // Apply attention (with Flash Attention if enabled)
        // Use standard 1/sqrt(head_dim) scaling - flash attention handles numerical stability
        let softmax_scale = 1.0 / (self.config.head_dim as f32).sqrt();  // 1/sqrt(128) ≈ 0.0884
        let flash_config = FlashAttentionConfig {
            softmax_scale: Some(softmax_scale),
            ..Default::default()
        };

        let att_out = if self.config.use_flash_attention {
            flash_attention_forward(
                &q,
                full_k,
                full_v,
                &flash_config,
            )?
        } else {
            // Standard attention with scaling
            let k_t = full_k.transpose(2, 3)?.contiguous()?;
            let att = q.contiguous()?.matmul(&k_t)?;
            let att = (att * (softmax_scale as f64))?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.contiguous()?.matmul(&full_v.contiguous()?)?
        };

        // Reshape and project output (contiguous after transpose)
        let hidden_out = self.config.n_heads * self.config.head_dim;
        let att_out = att_out.transpose(1, 2)?.contiguous()?
            .reshape((batch, seq_len, hidden_out))?;

        // Output projection: [B, S, H] -> [B*S, H] -> matmul -> [B, S, H]
        let att_flat = att_out.reshape((batch * seq_len, hidden_out))?;
        let o_out = wo.dim(1)?;
        let result = att_flat.matmul(wo)?.reshape((batch, seq_len, o_out))?;

        Ok(result)
    }

    /// Clear KV cache for new sequence
    pub fn reset(&mut self) {
        self.kv_cache.clear();
    }

    /// Get current sequence length
    pub fn seq_len(&self) -> usize {
        self.kv_cache.seq_len()
    }

    /// Apply RoPE (Rotary Position Embedding)
    /// Supports both interleaved (Qwen) and split-half (LLaMA/TinyLlama) formats
    fn apply_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        pos_start: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (batch, n_heads, seq_len, head_dim) = q.dims4()?;
        let (_, n_kv_heads, _, _) = k.dims4()?;
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;

        // Build cos/sin tables for this sequence
        let device = q.device();
        let mut cos_vals = Vec::with_capacity(seq_len * half_dim);
        let mut sin_vals = Vec::with_capacity(seq_len * half_dim);

        for s in 0..seq_len {
            let pos = (pos_start + s) as f32;
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos * freq;
                cos_vals.push(angle.cos());
                sin_vals.push(angle.sin());
            }
        }

        let cos = Tensor::from_vec(cos_vals, (1, 1, seq_len, half_dim), device)?;
        let sin = Tensor::from_vec(sin_vals, (1, 1, seq_len, half_dim), device)?;

        // println!("🔧 RoPE format: ... "); // COMMENTED OUT: fires 28x per token
        if self.config.rope_interleaved {
            // INTERLEAVED format (Qwen): pairs are (x[0],x[1]), (x[2],x[3]), ...
            let q_reshaped = q.reshape((batch, n_heads, seq_len, half_dim, 2))?;
            let k_reshaped = k.reshape((batch, n_kv_heads, seq_len, half_dim, 2))?;

            let q0 = q_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
            let q1 = q_reshaped.narrow(4, 1, 1)?.squeeze(4)?;
            let k0 = k_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
            let k1 = k_reshaped.narrow(4, 1, 1)?.squeeze(4)?;

            let q0_rot = (q0.broadcast_mul(&cos)? - q1.broadcast_mul(&sin)?)?;
            let q1_rot = (q0.broadcast_mul(&sin)? + q1.broadcast_mul(&cos)?)?;
            let k0_rot = (k0.broadcast_mul(&cos)? - k1.broadcast_mul(&sin)?)?;
            let k1_rot = (k0.broadcast_mul(&sin)? + k1.broadcast_mul(&cos)?)?;

            let q_rot = Tensor::stack(&[&q0_rot, &q1_rot], 4)?;
            let k_rot = Tensor::stack(&[&k0_rot, &k1_rot], 4)?;

            let q_out = q_rot.reshape((batch, n_heads, seq_len, head_dim))?;
            let k_out = k_rot.reshape((batch, n_kv_heads, seq_len, head_dim))?;
            Ok((q_out, k_out))
        } else {
            // SPLIT-HALF format (LLaMA/TinyLlama): first half + second half
            let q0 = q.narrow(3, 0, half_dim)?;
            let q1 = q.narrow(3, half_dim, half_dim)?;
            let k0 = k.narrow(3, 0, half_dim)?;
            let k1 = k.narrow(3, half_dim, half_dim)?;

            let q0_rot = (q0.broadcast_mul(&cos)? - q1.broadcast_mul(&sin)?)?;
            let q1_rot = (q0.broadcast_mul(&sin)? + q1.broadcast_mul(&cos)?)?;
            let k0_rot = (k0.broadcast_mul(&cos)? - k1.broadcast_mul(&sin)?)?;
            let k1_rot = (k0.broadcast_mul(&sin)? + k1.broadcast_mul(&cos)?)?;

            let q_out = Tensor::cat(&[&q0_rot, &q1_rot], 3)?;
            let k_out = Tensor::cat(&[&k0_rot, &k1_rot], 3)?;
            Ok((q_out, k_out))
        }
    }
}

// ============================================================================
// ENVIRONMENT SETUP
// ============================================================================

/// Set up CUDA environment for optimal performance
pub fn setup_cuda_env() {
    // Increase CUDA memory cache
    std::env::set_var("CUDA_CACHE_MAXSIZE", "2147483648"); // 2GB

    // Enable TF32 for faster matrix ops on Ampere+
    std::env::set_var("NVIDIA_TF32_OVERRIDE", "1");

    // Disable memory pool to reduce fragmentation
    // std::env::set_var("PYTORCH_NO_CUDA_MEMORY_CACHING", "1");

    println!("🚀 CUDA environment configured for turbo inference");
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache() {
        let mut cache = ModelKVCache::new(4, 1024);
        assert_eq!(cache.seq_len(), 0);

        // Would need device for actual tensor tests
    }

    #[test]
    fn test_pin_strategy() {
        let pin = LayerPin::new(4.0, PinStrategy::FirstLast { first: 2, last: 2 });
        assert!(pin.should_pin(0, 10)); // First
        assert!(pin.should_pin(1, 10)); // First
        assert!(!pin.should_pin(5, 10)); // Middle
        assert!(pin.should_pin(8, 10)); // Last
        assert!(pin.should_pin(9, 10)); // Last
    }
}
