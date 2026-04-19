//! OomLlama.rs - The Sovereign LLM Runtime
//!
//! "OpenAI buys 40% of world's RAM. We build the solution that saves 90%."
//!
//! Core Goals:
//! - 7B Model in 1GB RAM (Q2 Quantization)
//! - TIBET-signed Inference
//! - Rust-native efficiency (Candle)
//! - **Lazy Layer Loading**: Only load active layer into RAM.

use candle_core::{Device, Tensor, DType, Shape};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Cache, LlamaEosToks};
// Note: We need a custom Llama implementation to support lazy loading effectively.
// For this PoC, we will wrap the standard model but use our custom loader.
// In a full implementation, we would rewrite the Llama model struct to hold "LazyTensor"s.

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use uuid::Uuid;
use serde::Deserialize;
use crate::tibet::TibetFactory;
use crate::betti::{BettiManager, AllocationRequest, ResourceType, Humotica};
use crate::quant::OomLoader;
use crate::oomllama_turbo::{TurboEngine, TurboConfig, PinStrategy, FlashAttentionConfig, flash_attention_forward};

// Re-using the config struct
#[derive(Deserialize, Debug, Clone)]
struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<serde_json::Value>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rope_theta() -> f32 { 10000.0 }

impl From<LlamaConfig> for Config {
    fn from(c: LlamaConfig) -> Self {
        let eos_token_id = match c.eos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| LlamaEosToks::Single(v as u32)),
            Some(serde_json::Value::Array(a)) => {
                let ids: Vec<u32> = a.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect();
                Some(LlamaEosToks::Multiple(ids))
            }
            _ => None,
        };

        Config {
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            vocab_size: c.vocab_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: c.num_key_value_heads.unwrap_or(c.num_attention_heads),
            rms_norm_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
            use_flash_attn: false,
            bos_token_id: c.bos_token_id,
            eos_token_id,
            max_position_embeddings: c.max_position_embeddings,
            rope_scaling: None,
            tie_word_embeddings: c.tie_word_embeddings,
        }
    }
}

// Helper to reverse map Config to LlamaConfig for shape inference (temporary hack)
impl From<Config> for LlamaConfig {
    fn from(c: Config) -> Self {
        LlamaConfig {
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            vocab_size: c.vocab_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: Some(c.num_key_value_heads),
            rms_norm_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
            bos_token_id: c.bos_token_id,
            eos_token_id: None, // Simplified
            max_position_embeddings: c.max_position_embeddings,
            tie_word_embeddings: c.tie_word_embeddings,
        }
    }
}

// --- GHOST MODEL COMPONENTS ---

#[allow(dead_code)]
struct GhostLinear {
    weight: GhostLayer,
}

impl GhostLinear {
    fn new(name: &str, device: Device, loader: Arc<OomLoader>, config: LlamaConfig) -> Self {
        Self { weight: GhostLayer::new(name.to_string(), device, loader, config) }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let w = self.weight.materialize()?;
        // Weight is [out_features, in_features] (PyTorch convention)
        // Transpose to [in_features, out_features] for matmul: x @ w.t()
        let w = w.t()?;

        // Handle both 2D [seq, hidden] and 3D [batch, seq, hidden] input
        let x_dims = x.dims();
        if x_dims.len() == 3 {
            // 3D: flatten -> matmul -> reshape
            let (batch, seq, hidden) = x.dims3()?;
            let out_dim = w.dim(1)?;  // w.t() is [in_features, out_features]
            let x_flat = x.reshape((batch * seq, hidden))?;
            let res = x_flat.matmul(&w)?;
            Ok(res.reshape((batch, seq, out_dim))?)
        } else {
            // 2D: direct matmul
            Ok(x.matmul(&w)?)
        }
    }
}

/// Linear layer with bias support (for Qwen attention projections)
#[allow(dead_code)]
struct GhostLinearWithBias {
    weight: GhostLayer,
    bias: Option<GhostLayer>,
}

impl GhostLinearWithBias {
    fn new(weight_name: &str, bias_name: &str, device: Device, loader: Arc<OomLoader>, config: LlamaConfig) -> Self {
        // Check if bias exists in the model
        let has_bias = loader.tensors.contains_key(bias_name);

        let bias = if has_bias {
            Some(GhostLayer::new(bias_name.to_string(), device.clone(), loader.clone(), config.clone()))
        } else {
            None
        };
        Self {
            weight: GhostLayer::new(weight_name.to_string(), device, loader, config),
            bias,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let w = self.weight.materialize()?;
        // Weight is [out_features, in_features] (PyTorch convention)
        // Transpose to [in_features, out_features] for matmul: x @ w.t()
        let w = w.t()?;

        // Handle both 2D [seq, hidden] and 3D [batch, seq, hidden] input
        let x_dims = x.dims();
        let result = if x_dims.len() == 3 {
            // 3D: flatten -> matmul -> reshape
            let (batch, seq, hidden) = x.dims3()?;
            let out_dim = w.dim(1)?;  // w.t() is [in_features, out_features]
            let x_flat = x.reshape((batch * seq, hidden))?;
            let res = x_flat.matmul(&w)?;
            res.reshape((batch, seq, out_dim))?
        } else {
            // 2D: direct matmul
            x.matmul(&w)?
        };

        // Add bias if present
        if let Some(ref bias_layer) = self.bias {
            let bias = bias_layer.materialize()?;
            Ok(result.broadcast_add(&bias)?)
        } else {
            Ok(result)
        }
    }
}

#[allow(dead_code)]
struct GhostRMSNorm {
    weight: GhostLayer,
    eps: f64,
}

impl GhostRMSNorm {
    fn new(name: &str, device: Device, loader: Arc<OomLoader>, config: LlamaConfig, eps: f64) -> Self {
        Self { 
            weight: GhostLayer::new(name.to_string(), device, loader, config),
            eps 
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let w = self.weight.materialize()?;

        // RMSNorm: x / sqrt(mean(x^2) + eps) * w
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let pow_2 = x_f32.sqr()?;
        let mean_pow_2 = pow_2.mean_keepdim(candle_core::D::Minus1)?;
        let norm_x = x_f32.broadcast_div(&(mean_pow_2 + self.eps)?.sqrt()?)?;
        let res = norm_x.broadcast_mul(&w)?;
        Ok(res.to_dtype(x_dtype)?)
    }
}

#[allow(dead_code)]
struct GhostMlp {
    gate_proj: GhostLinear,
    up_proj: GhostLinear,
    down_proj: GhostLinear,
}

impl GhostMlp {
    fn new(prefix: &str, device: Device, loader: Arc<OomLoader>, config: LlamaConfig) -> Self {
        Self {
            gate_proj: GhostLinear::new(&format!("{}.gate_proj.weight", prefix), device.clone(), loader.clone(), config.clone()),
            up_proj: GhostLinear::new(&format!("{}.up_proj.weight", prefix), device.clone(), loader.clone(), config.clone()),
            down_proj: GhostLinear::new(&format!("{}.down_proj.weight", prefix), device, loader, config),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        // SiLU: x * sigmoid(x)
        let activated_gate = (gate.clone() * candle_nn::ops::sigmoid(&gate)?)?;
        let intermediate = (activated_gate * up)?;

        let result = self.down_proj.forward(&intermediate)?;
        Ok(result)
    }
}

#[allow(dead_code)]
struct GhostAttention {
    q_proj: GhostLinearWithBias,  // Qwen uses bias in attention projections
    k_proj: GhostLinearWithBias,
    v_proj: GhostLinearWithBias,
    o_proj: GhostLinear,  // Output projection has no bias
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    layer_idx: usize,
    rope_sin: Tensor,
    rope_cos: Tensor,
}

impl GhostAttention {
    fn new(prefix: &str, device: Device, loader: Arc<OomLoader>, config: LlamaConfig, layer_idx: usize) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let max_seq_len = 4096; // Max context length

        // Compute RoPE frequencies using model's rope_theta
        let (rope_sin, rope_cos) = crate::oomllama_turbo::compute_rope_freqs(
            head_dim,
            max_seq_len,
            config.rope_theta,
            &device,
        ).expect("Failed to compute RoPE frequencies");

        Self {
            // Qwen uses biases for Q, K, V projections
            q_proj: GhostLinearWithBias::new(
                &format!("{}.q_proj.weight", prefix),
                &format!("{}.q_proj.bias", prefix),
                device.clone(), loader.clone(), config.clone()
            ),
            k_proj: GhostLinearWithBias::new(
                &format!("{}.k_proj.weight", prefix),
                &format!("{}.k_proj.bias", prefix),
                device.clone(), loader.clone(), config.clone()
            ),
            v_proj: GhostLinearWithBias::new(
                &format!("{}.v_proj.weight", prefix),
                &format!("{}.v_proj.bias", prefix),
                device.clone(), loader.clone(), config.clone()
            ),
            o_proj: GhostLinear::new(&format!("{}.o_proj.weight", prefix), device, loader, config.clone()),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads),
            head_dim,
            layer_idx,
            rope_sin,
            rope_cos,
        }
    }

    /// Forward with TurboEngine KV-cache support
    fn forward_turbo(
        &self,
        x: &Tensor,
        turbo: &mut TurboEngine,
    ) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        // Materialize weight tensors and transpose from PyTorch convention [out, in] to [in, out]
        // The turbo engine does x.matmul(w) which needs [in_features, out_features]
        // But weights are stored as [out_features, in_features] (PyTorch convention)
        let wq = self.q_proj.weight.materialize()?.t()?;
        let wk = self.k_proj.weight.materialize()?.t()?;
        let wv = self.v_proj.weight.materialize()?.t()?;
        let wo = self.o_proj.weight.materialize()?.t()?;

        // Extract biases if present (Qwen requires biases on Q/K/V projections)
        let bq = self.q_proj.bias.as_ref().map(|b| b.materialize()).transpose()?;
        let bk = self.k_proj.bias.as_ref().map(|b| b.materialize()).transpose()?;
        let bv = self.v_proj.bias.as_ref().map(|b| b.materialize()).transpose()?;

        // Use TurboEngine's attention_forward with KV-cache
        let out = turbo.attention_forward_with_bias(
            self.layer_idx,
            x,
            &wq,
            &wk,
            &wv,
            &wo,
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
        )?;

        Ok(out)
    }

    /// Standard forward (no cache)
    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let (batch, seq_len, _) = x.dims3()?;

        // Reshape for multi-head attention [batch, seq, hidden] -> [batch, n_heads, seq, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE using pre-computed frequencies (interleaved format)
        let q = crate::oomllama_turbo::apply_rope(&q, &self.rope_sin, &self.rope_cos, 0)?;
        let k = crate::oomllama_turbo::apply_rope(&k, &self.rope_sin, &self.rope_cos, 0)?;

        // Flash Attention
        let att_out = flash_attention_forward(&q, &k, &v, &FlashAttentionConfig::default())?;

        // Reshape back [batch, n_heads, seq, head_dim] -> [batch, seq, hidden]
        let att_out = att_out.transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.o_proj.forward(&att_out)
    }
}

#[allow(dead_code)]
struct GhostDecoderLayer {
    self_attn: GhostAttention,
    mlp: GhostMlp,
    input_layernorm: GhostRMSNorm,
    post_attention_layernorm: GhostRMSNorm,
    layer_idx: usize,
}

impl GhostDecoderLayer {
    fn new(prefix: &str, device: Device, loader: Arc<OomLoader>, config: LlamaConfig, layer_idx: usize) -> Self {
        Self {
            self_attn: GhostAttention::new(&format!("{}.self_attn", prefix), device.clone(), loader.clone(), config.clone(), layer_idx),
            mlp: GhostMlp::new(&format!("{}.mlp", prefix), device.clone(), loader.clone(), config.clone()),
            input_layernorm: GhostRMSNorm::new(&format!("{}.input_layernorm.weight", prefix), device.clone(), loader.clone(), config.clone(), config.rms_norm_eps),
            post_attention_layernorm: GhostRMSNorm::new(&format!("{}.post_attention_layernorm.weight", prefix), device, loader, config.clone(), config.rms_norm_eps),
            layer_idx,
        }
    }

    /// Forward with turbo mode (KV-cache + Flash Attention)
    fn forward_turbo(&self, x: &Tensor, turbo: &mut TurboEngine) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;

        // Turbo attention with KV-cache
        let x = self.self_attn.forward_turbo(&x, turbo)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;

        Ok((x + residual)?)
    }
}

// --- GHOST MODEL ARCHITECTURE ---

/// Represents a tensor that resides on disk (Ghost) and can be materialized into VRAM.
#[allow(dead_code)]
struct GhostLayer {
    name: String,
    device: Device,
    loader: Arc<OomLoader>,
    config: LlamaConfig,
}

impl GhostLayer {
    #[allow(dead_code)]
    fn new(name: String, device: Device, loader: Arc<OomLoader>, config: LlamaConfig) -> Self {
        Self { name, device, loader, config }
    }

    /// Materialize the ghost into a real Tensor in VRAM.
    /// This triggers the dequantization from disk -> RAM -> GPU.
    #[allow(dead_code)]
    fn materialize(&self) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        // 1. Dequantize from mmap to CPU buffer (f32)
        let data = self.loader.dequantize_tensor(&self.name)?;

        // 2. Infer shape from config (Hacky, but works for now)
        let shape = infer_shape(&self.name, &self.config);

        // 3. Create on CPU first, then transfer to target device
        let cpu_tensor = Tensor::from_vec(data, shape, &Device::Cpu)?;

        // 4. Transfer to target device (GPU if available)
        let tensor = cpu_tensor.to_device(&self.device)?;

        Ok(tensor)
    }
}

/// Manages the lifecycle of GhostLayers.
/// Acts as a pseudo-VarBuilder that loads on demand.
/// NOTE: To fully integrate with Candle's Llama, we ideally need to rewrite Llama to use this directly.
/// For this Phase 2 PoC, we will keep the Model structure but intercept the weight loading?
/// Actually, Candle's `Llama` struct OWNS the tensors. It expects them loaded at init.
/// 
/// CRITICAL PIVOT: We cannot use standard `candle_transformers::models::llama::Llama` for Ghost Loading
/// without pre-loading everything. 
/// 
/// We need to implement a `GhostLlama` that executes layer-by-layer manually.
/// This is a big task. For now, we will simulate it by implementing a `GhostLoader` struct
/// that we can query. 
///
/// But wait! If we use `VarBuilder` with a custom backend?
/// Candle's VarBuilder reads everything into Tensors eagerly when the model is instantiated.
///
/// SOLUTION: We will implement our OWN minimal Llama inference loop here that uses GhostLayers.
/// Or, we use the `OomLlama` to hold the `GhostLayers` map, and we build a custom runner.
///
/// Let's build the `GhostLlama` struct.

#[allow(dead_code)]
struct GhostLlamaModel {
    embed_tokens: GhostLayer,
    layers: Vec<GhostDecoderLayer>,
    norm: GhostRMSNorm,
    lm_head: GhostLinear,
    // Dual GPU support: store both devices for tensor transfers
    devices: Vec<Device>,
}

impl GhostLlamaModel {
    fn new(primary_device: Device, secondary_device: Option<Device>, loader: Arc<OomLoader>, config: LlamaConfig) -> Self {
        let mut layers = Vec::new();

        // Dual GPU: alternate layers between devices
        let has_dual_gpu = secondary_device.is_some();
        let gpu1 = secondary_device.unwrap_or_else(|| primary_device.clone());

        for i in 0..config.num_hidden_layers {
            // Even layers -> primary GPU, Odd layers -> secondary GPU
            let layer_device = if has_dual_gpu && i % 2 == 1 {
                gpu1.clone()
            } else {
                primary_device.clone()
            };
            layers.push(GhostDecoderLayer::new(&format!("model.layers.{}", i), layer_device, loader.clone(), config.clone(), i));
        }

        if has_dual_gpu {
            println!("Dual GPU mode: even layers on GPU 0, odd layers on GPU 1");
        }

        Self {
            embed_tokens: GhostLayer::new("model.embed_tokens.weight".to_string(), primary_device.clone(), loader.clone(), config.clone()),
            layers,
            norm: GhostRMSNorm::new("model.norm.weight", primary_device.clone(), loader.clone(), config.clone(), config.rms_norm_eps),
            lm_head: GhostLinear::new("lm_head.weight", primary_device.clone(), loader, config),
            devices: vec![primary_device, gpu1],
        }
    }

    /// Forward pass with Turbo mode (KV-cache + Flash Attention)
    fn forward_turbo(&self, tokens: &Tensor, turbo: &mut TurboEngine) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let dual_gpu = self.devices.len() > 1 && self.devices[0].is_cuda() && self.devices[1].is_cuda();

        // 1. Embeddings
        let mut x = {
            let embed_w = self.embed_tokens.materialize()?;
            embed_w.embedding(&tokens.flatten_all()?)?
        };

        // Need to add batch dimension if missing [seq] -> [batch, seq, hidden]
        if x.dims().len() == 2 {
            x = x.unsqueeze(0)?;
        }

        // 2. Decoder Layers with Turbo
        for (i, layer) in self.layers.iter().enumerate() {
            if i % 10 == 0 {
                println!("Turbo processing layer {}/{}...", i, self.layers.len());
            }

            // Dual GPU tensor transfer
            if dual_gpu {
                let target_device = if i % 2 == 1 { &self.devices[1] } else { &self.devices[0] };
                if x.device().location() != target_device.location() {
                    x = x.to_device(target_device)?;
                }
            }

            // Use turbo forward with KV-cache
            x = layer.forward_turbo(&x, turbo)?;
        }

        // 3. Final Norm
        if dual_gpu {
            x = x.to_device(&self.devices[0])?;
        }

        x = self.norm.forward(&x)?;

        // 4. LM Head
        self.lm_head.forward(&x)
    }

    /// Standard forward (no turbo)
    fn forward(&self, tokens: &Tensor) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        // Check if we have dual GPU
        let dual_gpu = self.devices.len() > 1 && self.devices[0].is_cuda() && self.devices[1].is_cuda();

        // 1. Embeddings (Materialize -> Gather -> Evict) - Always on primary GPU
        let mut x = {
            let embed_w = self.embed_tokens.materialize()?;
            let emb = embed_w.embedding(&tokens.flatten_all()?)?;
            // Add batch dimension: [seq, hidden] -> [1, seq, hidden]
            emb.unsqueeze(0)?
        };

        // 2. Decoder Layers (The Ghost Loop)
        for (i, layer) in self.layers.iter().enumerate() {
            if i % 10 == 0 {
                println!("Ghost processing layer {}/{}...", i, self.layers.len());
            }

            // DUAL GPU: Transfer tensor to correct device before processing layer
            if dual_gpu {
                let target_device = if i % 2 == 1 { &self.devices[1] } else { &self.devices[0] };
                if x.device().location() != target_device.location() {
                    x = x.to_device(target_device)?;
                }
            }

            let residual = x.clone();
            x = layer.input_layernorm.forward(&x)?;

            // --- GHOST ATTENTION with Flash Attention ---
            x = layer.self_attn.forward(&x)?;
            x = (x + residual)?;

            // --- GHOST MLP ---
            let residual = x.clone();
            x = layer.post_attention_layernorm.forward(&x)?;
            x = layer.mlp.forward(&x)?;
            x = (x + residual)?;
        }

        // 3. Final Norm - Transfer back to primary GPU
        if dual_gpu {
            x = x.to_device(&self.devices[0])?;
        }

        x = self.norm.forward(&x)?;

        // 4. LM Head (Final Ghost)
        let logits = self.lm_head.forward(&x)?;

        Ok(logits)
    }
}

pub struct GhostLlama {
    // Model state
    #[allow(dead_code)]
    config: Config,
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    device: Device,

    // Ghost Model
    model: GhostLlamaModel,

    // Shared weight loader
    loader: Arc<OomLoader>,

    // 🚀 TURBO ENGINE - KV-Cache + Flash Attention
    turbo: Option<TurboEngine>,

    // Domain-AI Context (Temporary knowledge)
    active_context: Option<String>,

    // TIBET & Betti
    tibet: TibetFactory,
    betti: Arc<BettiManager>,
    allocation_id: Option<Uuid>,
    name: String,
}

impl GhostLlama {
    /// Create a new GhostLlama instance.
    /// - `gpu_index`: Primary GPU (None = CPU)
    /// - `secondary_gpu`: Optional second GPU for dual-GPU layer striping
    pub fn new(name: &str, gpu_index: Option<usize>, betti: Arc<BettiManager>, model_path: Option<&str>, tokenizer_path: Option<&str>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // NEW: Also try to initialize secondary GPU for dual-GPU mode
        // But ONLY if CUDA_VISIBLE_DEVICES allows multiple GPUs
        let secondary_gpu: Option<usize> = if gpu_index.is_some() {
            // Check CUDA_VISIBLE_DEVICES - if set to single GPU, don't try dual GPU
            let visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();
            let allow_dual = match &visible_devices {
                Some(val) => {
                    // If only one device specified (e.g., "0"), disable dual GPU
                    let device_count = val.split(',').filter(|s| !s.is_empty()).count();
                    if device_count <= 1 {
                        false
                    } else {
                        true
                    }
                }
                None => true, // No restriction, try dual GPU
            };

            if allow_dual {
                let secondary_idx = if gpu_index == Some(0) { 1 } else { 0 };
                match Device::new_cuda(secondary_idx) {
                    Ok(dev2) => {
                        // Test cross-device transfer before enabling dual GPU
                        // Test: create tensors on both GPUs and do matmul on GPU1
                        let primary_dev = Device::new_cuda(gpu_index.unwrap_or(0)).unwrap();
                        let test_result = (|| -> candle_core::Result<()> {
                            let a = candle_core::Tensor::randn(0f32, 1f32, (128, 128), &primary_dev)?;
                            let b = a.to_device(&dev2)?;
                            let c = candle_core::Tensor::randn(0f32, 1f32, (128, 128), &dev2)?;
                            let _d = b.matmul(&c)?;  // Matmul on GPU1
                            let e = _d.to_device(&primary_dev)?;  // Transfer back
                            let _f = a.matmul(&e)?;  // Matmul on GPU0 after roundtrip
                            Ok(())
                        })();
                        match test_result {
                            Ok(_) => {
                                println!("Secondary GPU {} initialized, cross-device transfer OK", secondary_idx);
                                Some(secondary_idx)
                            }
                            Err(_e) => {
                                None
                            }
                        }
                    }
                    Err(_) => None
                }
            } else {
                None
            }
        } else {
            None
        };

        let resource_type = if gpu_index.is_some() { ResourceType::Gpu } else { ResourceType::Cpu };

        // Request allocation
        let req = AllocationRequest {
            idd_name: name.to_string(),
            resource_type,
            amount: if secondary_gpu.is_some() { 3.0 } else { 1.5 }, // Double for dual GPU
            duration_secs: None,
            purpose: "Ghost Model Inference".to_string(),
            priority: 90,
            humotica: Some(Humotica {
                sense: "Inference".to_string(),
                context: if secondary_gpu.is_some() { "Ghost Model (70B) - DUAL GPU" } else { "Ghost Model (70B)" }.to_string(),
                intent: "Sovereign AI".to_string(),
                explanation: if secondary_gpu.is_some() {
                    "Running 70B model with dual GPU layer striping".to_string()
                } else {
                    "Running 70B model with 1GB VRAM paging".to_string()
                },
            }),
        };
        let allocation = betti.request(req).ok().map(|a| a.id);

        let device = match gpu_index {
            Some(idx) => {
                match Device::new_cuda(idx) {
                    Ok(d) => {
                        println!("CUDA device {} initialized", idx);
                        d
                    }
                    Err(e) => {
                        eprintln!("CUDA device {} failed: {:?}, falling back to CPU", idx, e);
                        Device::Cpu
                    }
                }
            }
            None => {
                println!("No GPU specified, using CPU");
                Device::Cpu
            }
        };

        let secondary_device = secondary_gpu.and_then(|idx| Device::new_cuda(idx).ok());

        // Load Tokenizer - detect model type from path
        let tokenizer = if let Some(path) = tokenizer_path {
            Tokenizer::from_file(path).map_err(|e| e.to_string())?
        } else {
            let api = Api::new()?;
            // Select tokenizer based on model name
            let model_name = model_path.unwrap_or("");
            let repo_name = if model_name.contains("qwen") || model_name.contains("humotica") {
                println!("Loading Qwen tokenizer...");
                "Qwen/Qwen2.5-7B-Instruct"
            } else if model_name.contains("llama3") || model_name.contains("llama-3") {
                println!("Loading Llama 3 tokenizer...");
                "meta-llama/Llama-3.2-1B"
            } else {
                println!("Loading TinyLlama tokenizer (default)...");
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            };
            let repo = api.repo(Repo::new(repo_name.to_string(), RepoType::Model));
            let filename = repo.get("tokenizer.json")?;
            Tokenizer::from_file(filename).map_err(|e| e.to_string())?
        };

        let config: Config = if let Some(path) = model_path {
             let model_dir = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
             let config_path = model_dir.join("config.json");
             
             // FORCE 70B/72B if markers found in filename
             if path.contains("llamaohm") || path.contains("llama-70b") || path.contains("llama3") {
                 // Llama 3.3 70B Instruct config
                 println!("Loading Llama 3.3 70B config (hidden: 8192, vocab: 128256)");
                 Config {
                    hidden_size: 8192,
                    intermediate_size: 28672,
                    vocab_size: 128256, // Llama 3.3 vocab
                    num_hidden_layers: 80,
                    num_attention_heads: 64,
                    num_key_value_heads: 8,
                    rms_norm_eps: 1e-5,
                    rope_theta: 500000.0,
                    use_flash_attn: false,
                    bos_token_id: Some(128000),
                    eos_token_id: Some(LlamaEosToks::Single(128001)),
                    max_position_embeddings: 131072,
                    rope_scaling: None,
                    tie_word_embeddings: false,
                 }
             } else if path.contains("32b") || path.contains("humotica-32") {
                 // Qwen 2.5 32B config (OomLlama's native brain!)
                 println!("Loading Qwen 2.5 32B config (hidden: 5120, layers: 64, vocab: 152064)");
                 Config {
                    hidden_size: 5120,
                    intermediate_size: 27648,
                    vocab_size: 152064, // Qwen 2.5 vocab
                    num_hidden_layers: 64,
                    num_attention_heads: 40,
                    num_key_value_heads: 8,
                    rms_norm_eps: 1e-6,
                    rope_theta: 1000000.0,
                    use_flash_attn: false,
                    bos_token_id: Some(151643),
                    eos_token_id: Some(LlamaEosToks::Single(151645)),
                    max_position_embeddings: 131072,
                    rope_scaling: None,
                    tie_word_embeddings: false,
                 }
             } else if path.contains("0.5b") || path.contains("qwen2.5-0.5") {
                 // Qwen 2.5 0.5B config
                 println!("Loading Qwen 2.5 0.5B config (hidden: 896, layers: 24, vocab: 151936)");
                 Config {
                    hidden_size: 896,
                    intermediate_size: 4864,
                    vocab_size: 151936,
                    num_hidden_layers: 24,
                    num_attention_heads: 14,
                    num_key_value_heads: 2,
                    rms_norm_eps: 1e-6,
                    rope_theta: 1000000.0,
                    use_flash_attn: false,
                    bos_token_id: Some(151643),
                    eos_token_id: Some(LlamaEosToks::Single(151645)),
                    max_position_embeddings: 32768,
                    rope_scaling: None,
                    tie_word_embeddings: false,
                 }
             } else if path.contains("3b") || path.contains("qwen2.5-3") {
                 // Qwen 2.5 3B config
                 println!("Loading Qwen 2.5 3B config (hidden: 2048, layers: 36, vocab: 151936)");
                 Config {
                    hidden_size: 2048,
                    intermediate_size: 11008,
                    vocab_size: 151936,
                    num_hidden_layers: 36,
                    num_attention_heads: 16,
                    num_key_value_heads: 2,
                    rms_norm_eps: 1e-6,
                    rope_theta: 1000000.0,
                    use_flash_attn: false,
                    bos_token_id: Some(151643),
                    eos_token_id: Some(LlamaEosToks::Single(151645)),
                    max_position_embeddings: 32768,
                    rope_scaling: None,
                    tie_word_embeddings: false,
                 }
             } else if path.contains("7b") || path.contains("humotica-7") || path.contains("qwen2.5-7") {
                 // Qwen 2.5 7B config
                 println!("Loading Qwen 2.5 7B config (hidden: 3584, layers: 28, vocab: 152064)");
                 Config {
                    hidden_size: 3584,
                    intermediate_size: 18944,
                    vocab_size: 152064, // Qwen 2.5 vocab
                    num_hidden_layers: 28,
                    num_attention_heads: 28,
                    num_key_value_heads: 4,
                    rms_norm_eps: 1e-6,
                    rope_theta: 1000000.0,
                    use_flash_attn: false,
                    bos_token_id: Some(151643),
                    eos_token_id: Some(LlamaEosToks::Single(151645)),
                    max_position_embeddings: 32768,
                    rope_scaling: None,
                    tie_word_embeddings: false,
                 }
             } else if path.contains("70b") || path.contains("72b") || path.contains("humotica-72") {
                 // Qwen 2.5 72B config
                 println!("Loading Qwen 2.5 72B config (hidden: 8192, vocab: 152064)");
                 Config {
                    hidden_size: 8192,
                    intermediate_size: 28672,
                    vocab_size: 152064, // Qwen 2.5 default vocab
                    num_hidden_layers: 80,
                    num_attention_heads: 64,
                    num_key_value_heads: 8,
                    rms_norm_eps: 1e-5,
                    rope_theta: 500000.0,
                    use_flash_attn: false,
                    bos_token_id: Some(151643),
                    eos_token_id: Some(LlamaEosToks::Single(151645)),
                    max_position_embeddings: 8192,
                    rope_scaling: None,
                    tie_word_embeddings: false,
                 }
             } else if config_path.exists() {
                 println!("Loading config from {:?}", config_path);
                 let l_config: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
                 Config::from(l_config)
             } else {
                 println!("Loading TinyLlama default config");
                 let api = Api::new()?;
                 let repo = api.repo(Repo::new("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(), RepoType::Model));
                 let filename = repo.get("config.json")?;
                 let l_config: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(filename)?)?;
                 Config::from(l_config)
             }
        } else {
             let api = Api::new()?;
             let repo = api.repo(Repo::new("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(), RepoType::Model));
             let filename = repo.get("config.json")?;
             let l_config: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(filename)?)?;
             Config::from(l_config)
        };

        let loader = if let Some(path) = model_path {
            if path.ends_with(".oom") { Arc::new(OomLoader::load(path)?) }
            else { return Err("Ghost Model requires .oom file".into()); }
        } else {
            let path = "data/kmbit/models/tinyllama_q4.oom";
            if std::path::Path::new(path).exists() { Arc::new(OomLoader::load(path)?) }
            else { return Err("No model found. Please provide --model <path.oom>".into()); }
        };

        let model = GhostLlamaModel::new(device.clone(), secondary_device, loader.clone(), LlamaConfig::from(config.clone()));

        // Determine RoPE format from model type
        let model_name = model_path.unwrap_or("");
        let is_qwen_model = model_name.contains("qwen") || model_name.contains("humotica") || model_name.contains("0.5b") || model_name.contains("3b");

        // 🚀 Initialize TurboEngine based on model config
        let turbo = if gpu_index.is_some() {
            let turbo_config = if config.num_hidden_layers == 64 && config.hidden_size == 5120 {
                // Qwen 32B - use preset
                println!("Turbo mode: Qwen 32B config, KV-Cache + Flash Attention enabled");
                TurboConfig::qwen32b_dual3060()
            } else {
                // Generic config based on detected model
                println!("Turbo mode: KV-Cache + Flash Attention enabled");
                TurboConfig {
                    n_layers: config.num_hidden_layers,
                    hidden_size: config.hidden_size,
                    n_heads: config.num_attention_heads,
                    n_kv_heads: config.num_key_value_heads,
                    head_dim: config.hidden_size / config.num_attention_heads,
                    max_seq_len: config.max_position_embeddings.min(8192), // Limit for memory
                    vram_budget_gb: if secondary_gpu.is_some() { 20.0 } else { 10.0 },
                    pin_strategy: PinStrategy::FirstLast { first: 4, last: 4 },
                    prefetch_lookahead: 2,
                    use_flash_attention: true,
                    rope_theta: config.rope_theta,
                    // Qwen models use interleaved RoPE, LLaMA/TinyLlama use split-half
                    rope_interleaved: is_qwen_model,
                    use_fp16: true,
                }
            };
            Some(TurboEngine::new(turbo_config, device.clone()))
        } else {
            println!("Turbo disabled (CPU mode)");
            None
        };

        Ok(Self {
            config,
            tokenizer,
            device,
            model,
            loader,
            turbo,
            active_context: None,
            tibet: TibetFactory::new(name),
            betti,
            allocation_id: allocation,
            name: name.to_string(),
        })
    }

    /// Inject temporary domain context (Vertical Virtual Fun)
    pub fn push_context(&mut self, context: &str) {
        println!("Context loaded ({} bytes)", context.len());
        self.active_context = Some(context.to_string());
    }

    pub fn infer(&mut self, prompt: &str, max_tokens: usize) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let _token = self.tibet.action("GhostInference", &self.name, serde_json::json!({ "prompt": prompt }));

        // Incorporate context into prompt
        let context_prefix = if let Some(ctx) = &self.active_context {
            format!("### EXTRA CONTEXT:\n{}\n\n", ctx)
        } else {
            "".to_string()
        };

        // Use Qwen's ChatML format
        let full_prompt = format!("{}<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", context_prefix, prompt);
        let tokens = self.tokenizer.encode(full_prompt, true).map_err(|e| e.to_string())?.get_ids().to_vec();
        let prompt_len = tokens.len();

        // Check if turbo mode is available
        if let Some(ref mut turbo) = self.turbo {
            // 🚀 TURBO MODE: KV-Cache enabled inference
            println!("TURBO Inference starting ({} prompt tokens)...", prompt_len);

            // Reset KV-cache for new sequence
            turbo.reset();

            // First pass: process full prompt to build KV-cache
            let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward_turbo(&input, turbo)?;

            // Get first generated token: logits is [batch=1, seq, vocab]
            let seq_len = logits.dim(1)?;
            let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?; // [1, vocab]
            let next_token_idx = last_logits.argmax(candle_core::D::Minus1)?; // [1]
            let mut next_token = next_token_idx.squeeze(0)?.to_scalar::<u32>()?; // scalar

            let mut generated_tokens: Vec<u32> = Vec::new();

            // Autoregressive generation with KV-cache
            for i in 0..max_tokens {
                // Check EOS
                if next_token == 2 || next_token == 151643 || next_token == 128001 || next_token == 151645 {
                    println!("EOS token reached at step {}", i);
                    break;
                }

                generated_tokens.push(next_token);

                if i % 10 == 0 {
                    println!("Generated {}/{} tokens...", i + 1, max_tokens);
                }

                // TURBO: Only pass the NEW token - KV-cache handles history!
                let new_token_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward_turbo(&new_token_tensor, turbo)?;

                // Get next token
                let seq_len = logits.dim(1)?;
                let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
                let next_token_idx = last_logits.argmax(candle_core::D::Minus1)?;
                next_token = next_token_idx.squeeze(0)?.to_scalar::<u32>()?;
            }

            println!("Generation complete: {} tokens generated", generated_tokens.len());

            let output = self.tokenizer.decode(&generated_tokens, true).map_err(|e| e.to_string())?;
            Ok(output)
        } else {
            // Standard mode (no KV-cache)
            println!("Ghost Inference starting ({} prompt tokens)...", prompt_len);

            let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut current_input = input;

            for i in 0..max_tokens {
                let logits = self.model.forward(&current_input)?;

                let seq_len = logits.dim(1)?;
                let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?; // [batch, vocab]
                // Get argmax, then squeeze batch dim if present
                let next_token_tensor = last_logits.argmax(candle_core::D::Minus1)?; // [batch] or scalar
                let next_token = if next_token_tensor.dims().is_empty() {
                    next_token_tensor.to_scalar::<u32>()?
                } else {
                    // Squeeze any remaining dimensions and get first element
                    next_token_tensor.flatten_all()?.to_vec1::<u32>()?[0]
                };

                if next_token == 2 || next_token == 151643 || next_token == 128001 || next_token == 151645 {
                    println!("EOS token reached at step {}", i);
                    break;
                }

                generated_tokens.push(next_token);

                // Without KV-cache: must pass full sequence each time
                let new_token = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                current_input = Tensor::cat(&[&current_input, &new_token], 1)?;

                if i % 10 == 0 {
                    println!("Generated {}/{} tokens...", i + 1, max_tokens);
                }
            }

            println!("Generation complete ({} tokens)", generated_tokens.len());
            let output = self.tokenizer.decode(&generated_tokens, true).map_err(|e| e.to_string())?;
            Ok(output)
        }
    }

    fn load_tensor(&self, name: &str) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        let l_config: LlamaConfig = self.config.clone().into(); // Hacky back-conversion
        let ghost = GhostLayer::new(name.to_string(), self.device.clone(), self.loader.clone(), l_config);
        ghost.materialize()
    }
}

// Rename OomLlama to GhostLlama in the rest of the file or use a type alias
pub type OomLlama = GhostLlama;

fn infer_shape(name: &str, config: &LlamaConfig) -> Shape {
    // Return shapes in PyTorch convention: [out_features, in_features].
    // GhostLinear.forward() transposes before matmul: x @ w.t()

    if name.contains("embed_tokens") {
        return Shape::from((config.vocab_size, config.hidden_size));
    }
    if name.contains("lm_head") {
        // PyTorch: [vocab_size, hidden_size] = [152064, 3584]
        return Shape::from((config.vocab_size, config.hidden_size));
    }
    if name.contains("input_layernorm") || name.contains("post_attention_layernorm") || name == "model.norm.weight" {
        return Shape::from((config.hidden_size,));
    }
    // Handle attention projection BIASES first (1D tensors)
    if name.ends_with(".bias") {
        if name.contains("q_proj") || name.contains("o_proj") {
            return Shape::from((config.hidden_size,));
        }
        if name.contains("k_proj") || name.contains("v_proj") {
            let kv_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
            let head_dim = config.hidden_size / config.num_attention_heads;
            return Shape::from((kv_heads * head_dim,));
        }
    }
    // Handle attention projection WEIGHTS (2D tensors)
    // PyTorch convention: [out_features, in_features]
    if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj") || name.contains("o_proj") {
        if name.contains("k_proj") || name.contains("v_proj") {
            let kv_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
            let head_dim = config.hidden_size / config.num_attention_heads;
            // PyTorch: [kv_dim, hidden_size] = [512, 3584]
            return Shape::from((kv_heads * head_dim, config.hidden_size));
        }
        // PyTorch: [hidden_size, hidden_size] = [3584, 3584]
        return Shape::from((config.hidden_size, config.hidden_size));
    }
    if name.contains("gate_proj") || name.contains("up_proj") {
        // PyTorch: [intermediate_size, hidden_size] = [18944, 3584]
        return Shape::from((config.intermediate_size, config.hidden_size));
    }
    if name.contains("down_proj") {
        // PyTorch: [hidden_size, intermediate_size] = [3584, 18944]
        return Shape::from((config.hidden_size, config.intermediate_size));
    }
    Shape::from((0,)) // Fallback
}

impl Drop for GhostLlama {
    fn drop(&mut self) {
        if let Some(id) = self.allocation_id {
            let _ = self.betti.release(id);
        }
    }
}
