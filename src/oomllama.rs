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

        // Debug: Print bias availability once per unique layer type
        static PRINTED_BIASES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !PRINTED_BIASES.swap(true, std::sync::atomic::Ordering::Relaxed) {
            println!("🔍 BIAS CHECK: {} -> {}", bias_name, if has_bias { "FOUND" } else { "NOT FOUND" });
        }

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

        // Debug weight values (once)
        static DEBUG_WEIGHTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_WEIGHTS.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let w_flat = w.flatten_all()?.to_vec1::<f32>()?;
            let w_min = w_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let w_max = w_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔍 LINEAR WEIGHT: shape={:?}, min={:.4}, max={:.4}", w.shape(), w_min, w_max);
            println!("🔍 LINEAR INPUT X: shape={:?}, min={:.4}, max={:.4}", x.shape(), x_min, x_max);
        }

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

        // Debug: check RMSNorm weight (gamma) values once
        static DEBUG_RMSNORM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_RMSNORM.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let w_flat = w.flatten_all()?.to_vec1::<f32>()?;
            let w_min = w_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let w_max = w_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let w_mean: f32 = w_flat.iter().sum::<f32>() / w_flat.len() as f32;
            println!("🔧 RMSNORM WEIGHT (gamma): min={:.4}, max={:.4}, mean={:.4}", w_min, w_max, w_mean);
        }

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
        // Debug first few calls
        static DEBUG_MLP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mlp_call = DEBUG_MLP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let do_debug = mlp_call < 3;  // Debug first 3 MLP calls (layers 0, 1, 2)

        if do_debug {
            // Check gate_proj weight stats
            let gate_w = self.gate_proj.weight.materialize()?;
            let w_flat = gate_w.flatten_all()?.to_vec1::<f32>()?;
            let w_min = w_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let w_max = w_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let w_mean: f32 = w_flat.iter().sum::<f32>() / w_flat.len() as f32;
            let w_var: f32 = w_flat.iter().map(|v| (v - w_mean).powi(2)).sum::<f32>() / w_flat.len() as f32;
            println!("🔧 MLP GATE_PROJ.weight: shape={:?}, min={:.6}, max={:.6}, mean={:.6}, var={:.6}",
                gate_w.shape(), w_min, w_max, w_mean, w_var);
        }

        let gate = self.gate_proj.forward(x)?;
        if do_debug {
            let g_flat = gate.flatten_all()?.to_vec1::<f32>()?;
            let g_min = g_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let g_max = g_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔧 MLP[{}] GATE output: min={:.4}, max={:.4}", mlp_call, g_min, g_max);
            // Show last position first 5 values
            let sq = gate.dim(1).unwrap_or(1);
            let hd = gate.dim(2).unwrap_or(1);
            let ls = (sq - 1) * hd;
            let le = (ls + 5).min(g_flat.len());
            println!("MLP_DEBUG[{}] gate last_pos[:5]: {:?}", mlp_call, &g_flat[ls..le]);
        }

        let up = self.up_proj.forward(x)?;
        if do_debug {
            let u_flat = up.flatten_all()?.to_vec1::<f32>()?;
            let u_min = u_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let u_max = u_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔧 MLP[{}] UP: min={:.4}, max={:.4}", mlp_call, u_min, u_max);
            let sq = up.dim(1).unwrap_or(1);
            let hd = up.dim(2).unwrap_or(1);
            let ls = (sq - 1) * hd;
            let le = (ls + 5).min(u_flat.len());
            println!("MLP_DEBUG[{}] up last_pos[:5]: {:?}", mlp_call, &u_flat[ls..le]);
        }

        // Simple approximation of SiLU: x * sigmoid(x)
        let activated_gate = (gate.clone() * candle_nn::ops::sigmoid(&gate)?)?;
        let intermediate = (activated_gate * up)?;
        if do_debug {
            let i_flat = intermediate.flatten_all()?.to_vec1::<f32>()?;
            let i_min = i_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let i_max = i_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔧 MLP INTERMEDIATE: min={:.4}, max={:.4}", i_min, i_max);
        }

        let result = self.down_proj.forward(&intermediate)?;
        if do_debug {
            let r_flat = result.flatten_all()?.to_vec1::<f32>()?;
            let r_min = r_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let r_max = r_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔧 MLP[{}] DOWN (output): min={:.4}, max={:.4}", mlp_call, r_min, r_max);
            let sq = result.dim(1).unwrap_or(1);
            let hd = result.dim(2).unwrap_or(1);
            let ls = (sq - 1) * hd;
            let le = (ls + 5).min(r_flat.len());
            println!("MLP_DEBUG[{}] down last_pos[:5]: {:?}", mlp_call, &r_flat[ls..le]);
        }
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

        // Debug Q/K/V projections (once)
        static DEBUG_QKV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_QKV.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let q_flat = q.flatten_all()?.to_vec1::<f32>()?;
            let k_flat = k.flatten_all()?.to_vec1::<f32>()?;
            let q_min = q_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let q_max = q_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let k_min = k_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let k_max = k_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔍 GHOST Q PROJ: min={:.4}, max={:.4}, shape={:?}", q_min, q_max, q.shape());
            let q_first5: Vec<f32> = q_flat[..5.min(q_flat.len())].to_vec();
            println!("PYTHON_CMP Q-PROJ q[0,:5]: {:?}", q_first5);
            println!("🔍 GHOST K PROJ: min={:.4}, max={:.4}, shape={:?}", k_min, k_max, k.shape());
        }

        let (batch, seq_len, _) = x.dims3()?;

        // Reshape for multi-head attention [batch, seq, hidden] -> [batch, n_heads, seq, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE using pre-computed frequencies (INTERLEAVED format for GGUF weights)
        let q = crate::oomllama_turbo::apply_rope(&q, &self.rope_sin, &self.rope_cos, 0)?;
        let k = crate::oomllama_turbo::apply_rope(&k, &self.rope_sin, &self.rope_cos, 0)?;

        // Debug Q/K after RoPE (once)
        static DEBUG_ROPE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_ROPE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let q_flat = q.flatten_all()?.to_vec1::<f32>()?;
            let k_flat = k.flatten_all()?.to_vec1::<f32>()?;
            let q_min = q_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let q_max = q_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let k_min = k_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let k_max = k_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔍 GHOST Q AFTER ROPE: min={:.4}, max={:.4}", q_min, q_max);
            println!("🔍 GHOST K AFTER ROPE: min={:.4}, max={:.4}", k_min, k_max);
        }

        // Flash Attention
        let att_out = flash_attention_forward(&q, &k, &v, &FlashAttentionConfig::default())?;

        // Reshape back [batch, n_heads, seq, head_dim] -> [batch, seq, hidden]
        let att_out = att_out.transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // Debug: pre-O-proj attention output (once)
        static DEBUG_PRE_OPROJ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_PRE_OPROJ.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let flat = att_out.flatten_all()?.to_vec1::<f32>()?;
            let min_v = flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_v = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("PRE_OPROJ: shape={:?}, min={:.6}, max={:.6}", att_out.shape(), min_v, max_v);
            println!("PRE_OPROJ [0,:10]: {:?}", &flat[..10.min(flat.len())]);
            // Also show o_proj weight info
            let ow = self.o_proj.weight.materialize()?;
            let ow_flat = ow.flatten_all()?.to_vec1::<f32>()?;
            println!("O_PROJ.weight shape={:?}, first 5={:?}", ow.shape(), &ow_flat[..5]);
        }

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
        // Debug: Check for NaN at layer entry (only first few calls)
        static DEBUG_LAYER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let debug_count = DEBUG_LAYER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let do_debug = debug_count < 3;

        if do_debug {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let has_nan = x_flat.iter().any(|v| v.is_nan());
            println!("🔧 LAYER {} INPUT: has_nan={}, first 3: {:?}", self.layer_idx, has_nan, &x_flat[..3.min(x_flat.len())]);
        }

        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;

        // Debug: check values AFTER layernorm, BEFORE attention
        if do_debug {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let x_mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let x_var: f32 = x_flat.iter().map(|v| (v - x_mean).powi(2)).sum::<f32>() / x_flat.len() as f32;
            println!("🔧 LAYER {} AFTER LAYERNORM: min={:.4}, max={:.4}, mean={:.4}, var={:.4}", self.layer_idx, x_min, x_max, x_mean, x_var);
        }

        // Turbo attention with KV-cache
        let x = self.self_attn.forward_turbo(&x, turbo)?;

        if do_debug {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔧 LAYER {} POST-ATTN: min={:.4}, max={:.4}", self.layer_idx, x_min, x_max);
        }

        let x = (x + residual)?;

        if do_debug {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔧 LAYER {} POST-RESIDUAL1: min={:.4}, max={:.4}", self.layer_idx, x_min, x_max);
        }

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;

        if do_debug {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔧 LAYER {} POST-MLP: min={:.4}, max={:.4}", self.layer_idx, x_min, x_max);
        }

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

        // DEBUG: Print first tensor's info
        static FIRST_PRINT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !FIRST_PRINT.swap(true, std::sync::atomic::Ordering::Relaxed) {
            println!("🔧 DEBUG: Target device = {:?}", self.device);
            println!("🔧 DEBUG: Tensor '{}', shape {:?}, {} values", self.name, shape, data.len());
            println!("🔧 DEBUG: First 5 values: {:?}", &data[..5.min(data.len())]);
        }

        // DEBUG: Show dequantized values for embed_tokens
        if self.name.contains("embed_tokens") {
            // println!("DEQUANT_DEBUG embed_tokens: total_values={}, first 10: {:?}", data.len(), &data[..10.min(data.len())]);
            // Row 1 starts at offset hidden_size (2048)
            if data.len() > 2058 {
                // println!("DEQUANT_DEBUG embed_tokens row 1 (offset 2048): {:?}", &data[2048..2058]);
            }
        }

        // 3. Create on CPU first, then transfer to target device
        let cpu_tensor = Tensor::from_vec(data, shape, &Device::Cpu)?;

        // DEBUG: Verify tensor content matches dequant output
        if self.name.contains("embed_tokens") {
            // Read row 1 directly from the tensor
            let row1 = cpu_tensor.narrow(0, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;
            // println!("TENSOR_VERIFY embed_tokens[1] first 5: {:?}", &row1[..5.min(row1.len())]);
//            println!("TENSOR_VERIFY shape: {:?}", cpu_tensor.shape());
        }

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
            println!("🔀 DUAL GPU MODE: Even layers → GPU 0, Odd layers → GPU 1");
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
            // Debug: print token IDs and embedding shape
            let token_ids = tokens.flatten_all()?.to_vec1::<u32>()?;
            // println!("🔍 TOKEN IDS: {:?}", &token_ids[..5.min(token_ids.len())]);
            // println!("🔍 EMBED_W shape: {:?}", embed_w.shape());

            // Debug: Check specific rows of embedding matrix
            let embed_flat = embed_w.flatten_all()?.to_vec1::<f32>()?;
            let hidden = 3584;
            for &tok in &token_ids[..3.min(token_ids.len())] {
                let start = tok as usize * hidden;
                let end = start + 5.min(hidden);
                if end <= embed_flat.len() {
                    // println!("🔍 EMBED ROW {} first 5: {:?}", tok, &embed_flat[start..end]);
                }
            }

            // For turbo: tokens is just the new token(s), not the full sequence
            let emb = embed_w.embedding(&tokens.flatten_all()?)?;

            // Debug: print raw embedding output
            let emb_flat = emb.flatten_all()?.to_vec1::<f32>()?;
            let emb_min = emb_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let emb_max = emb_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // println!("🔍 RAW EMBEDDING: shape={:?}, min={:.6}, max={:.6}, first 5: {:?}",
                // emb.shape(), emb_min, emb_max, &emb_flat[..5.min(emb_flat.len())]);
            emb
        };

        // Need to add batch dimension if missing [seq] -> [batch, seq, hidden]
        if x.dims().len() == 2 {
            x = x.unsqueeze(0)?;
        }

        // 2. Decoder Layers with Turbo
        static DEBUG_LAYERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let do_debug_layers = !DEBUG_LAYERS.swap(true, std::sync::atomic::Ordering::Relaxed);

        for (i, layer) in self.layers.iter().enumerate() {
            if i % 10 == 0 {
                let cached = turbo.seq_len();
                println!("🚀 Turbo layer {}/{} (KV-cache: {} tokens)...", i, self.layers.len(), cached);
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

            // Debug: track per-layer magnitude
            if do_debug_layers {
                let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
                let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
                let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let x_absmax = x_min.abs().max(x_max.abs());
                if i < 5 || x_absmax > 1000.0 {
                    println!("🔍 LAYER {} OUTPUT: min={:.2}, max={:.2}, |max|={:.2}", i, x_min, x_max, x_absmax);
                }
            }
        }

        // 3. Final Norm
        if dual_gpu {
            x = x.to_device(&self.devices[0])?;
        }

        // Debug: Check hidden state before and after final norm (once)
        static DEBUG_FINAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let do_debug_final = !DEBUG_FINAL.swap(true, std::sync::atomic::Ordering::Relaxed);
        if do_debug_final {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let x_mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            println!("🔍 PRE-FINAL_NORM: min={:.4}, max={:.4}, mean={:.4}", x_min, x_max, x_mean);
        }

        x = self.norm.forward(&x)?;

        if do_debug_final {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let x_mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            println!("🔍 POST-FINAL_NORM: min={:.4}, max={:.4}, mean={:.4}", x_min, x_max, x_mean);
        }

        // Debug: Check hidden state before lm_head (once)
        static DEBUG_LM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_LM.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let x_mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            println!("🔍 PRE-LM_HEAD: shape={:?}, min={:.4}, max={:.4}, mean={:.4}",
                x.shape(), x_min, x_max, x_mean);

            // Check lm_head weight
            let lm_w = self.lm_head.weight.materialize()?;
            let lm_w_flat = lm_w.flatten_all()?.to_vec1::<f32>()?;
            let w_min = lm_w_flat[..1000].iter().cloned().fold(f32::INFINITY, f32::min);
            let w_max = lm_w_flat[..1000].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("🔍 LM_HEAD.weight: shape={:?}, first 1000 min={:.4}, max={:.4}",
                lm_w.shape(), w_min, w_max);
        }

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

        // Debug: check embeddings
        {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let has_inf = x_flat.iter().any(|v| v.is_infinite());
            let has_nan = x_flat.iter().any(|v| v.is_nan());
            println!("🔍 POST-EMBED: has_inf={}, has_nan={}, first 3: {:?}", has_inf, has_nan, &x_flat[..3.min(x_flat.len())]);
            println!("PYTHON_CMP POST-EMBED x[0,:5]: {:?}", &x_flat[..5.min(x_flat.len())]);
        }

        // 2. Decoder Layers (The Ghost Loop)
        for (i, layer) in self.layers.iter().enumerate() {
            if i % 10 == 0 {
                println!("👻 Ghost processing layer {}/{}...", i, self.layers.len());
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

            // Debug after layernorm in layer 0 and 1
            if i <= 1 {
                let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
                let has_nan = x_flat.iter().any(|v| v.is_nan());
                let seq_len = x.dim(1).unwrap_or(1);
                let hidden = x.dim(2).unwrap_or(1);
                let last_start = (seq_len - 1) * hidden;
                let last_end = (last_start + 5).min(x_flat.len());
                println!("L{}_DEBUG post-norm last_pos[:5]: {:?}", i, &x_flat[last_start..last_end]);
                if i == 0 {
                    println!("🔍 L0 POST-NORM: has_nan={}", has_nan);
                    println!("PYTHON_CMP L0-POST-NORM norm_x[0,:5]: {:?}", &x_flat[..5.min(x_flat.len())]);
                }
            }

            // --- GHOST ATTENTION with Flash Attention ---
            x = layer.self_attn.forward(&x)?;

            // Debug after attention in layers 0 and 1
            if i <= 1 {
                let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
                let has_nan = x_flat.iter().any(|v| v.is_nan());
                let seq_len_tmp = x.dim(1).unwrap_or(1);
                let hidden_tmp = x.dim(2).unwrap_or(1);
                let last_s = (seq_len_tmp - 1) * hidden_tmp;
                let last_e = (last_s + 5).min(x_flat.len());
                println!("L{}_DEBUG post-attn(o_proj) last_pos[:5]: {:?}", i, &x_flat[last_s..last_e]);
                if i == 0 {
                    println!("🔍 L0 POST-ATTN: has_nan={}", has_nan);
                    println!("PYTHON_CMP L0-POST-ATTN attn_out[0,:5]: {:?}", &x_flat[..5.min(x_flat.len())]);
                }
            }

            x = (x + residual)?;

            if i <= 1 {
                let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
                let sq = x.dim(1).unwrap_or(1);
                let hd = x.dim(2).unwrap_or(1);
                let ls = (sq - 1) * hd;
                let le = (ls + 5).min(x_flat.len());
                println!("L{}_DEBUG post-residual1 last_pos[:5]: {:?}", i, &x_flat[ls..le]);
            }

            // --- GHOST MLP ---
            let residual = x.clone();
            x = layer.post_attention_layernorm.forward(&x)?;
            x = layer.mlp.forward(&x)?;
            x = (x + residual)?;

            // Debug EVERY layer: hidden state stats
            {
                let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
                let has_inf = x_flat.iter().any(|v| v.is_infinite());
                let has_nan = x_flat.iter().any(|v| v.is_nan());
                let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
                let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let x_mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
                // Last token position stats (this is what matters for generation)
                let seq_len = x.dim(1).unwrap_or(1);
                let hidden = x.dim(2).unwrap_or(1);
                let last_start = (seq_len - 1) * hidden;
                let last_end = (last_start + 10).min(x_flat.len());
                let last_slice = &x_flat[last_start..last_end];
                println!("LAYER_DEBUG L{}: min={:.4}, max={:.4}, mean={:.6}, nan={}, inf={}, last_pos[:10]={:?}",
                    i, x_min, x_max, x_mean, has_nan, has_inf, last_slice);
                if i == 0 {
                    println!("PYTHON_CMP L0-COMPLETE layer_out[0,:5]: {:?}", &x_flat[..5.min(x_flat.len())]);
                }
            }
        }

        // 3. Final Norm - Transfer back to primary GPU
        if dual_gpu {
            x = x.to_device(&self.devices[0])?;
        }

        // DEBUG: Pre-final-norm hidden state (first call only)
        static DEBUG_STD_FINAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let do_debug_std = !DEBUG_STD_FINAL.swap(true, std::sync::atomic::Ordering::Relaxed);
        if do_debug_std {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let x_mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            let has_nan = x_flat.iter().any(|v| v.is_nan());
            let has_inf = x_flat.iter().any(|v| v.is_infinite());
            println!("STD_DEBUG PRE-FINAL-NORM: shape={:?}, min={:.4}, max={:.4}, mean={:.4}, nan={}, inf={}",
                x.shape(), x_min, x_max, x_mean, has_nan, has_inf);
            // Show last token position hidden state (this is what lm_head uses for generation)
            let seq_len = x.dim(1).unwrap_or(1);
            let last_pos = x.narrow(1, seq_len - 1, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            println!("STD_DEBUG PRE-FINAL-NORM last_pos[:10]: {:?}", &last_pos[..10.min(last_pos.len())]);
        }

        x = self.norm.forward(&x)?;

        if do_debug_std {
            let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
            let x_min = x_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let x_max = x_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let x_mean: f32 = x_flat.iter().sum::<f32>() / x_flat.len() as f32;
            println!("STD_DEBUG POST-FINAL-NORM: shape={:?}, min={:.4}, max={:.4}, mean={:.4}", x.shape(), x_min, x_max, x_mean);
            // Last position
            let seq_len = x.dim(1).unwrap_or(1);
            let last_pos = x.narrow(1, seq_len - 1, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            println!("STD_DEBUG POST-FINAL-NORM last_pos[:10]: {:?}", &last_pos[..10.min(last_pos.len())]);

            // Check lm_head weight shape
            let lm_w = self.lm_head.weight.materialize().unwrap();
            println!("STD_DEBUG LM_HEAD.weight shape: {:?}", lm_w.shape());
            let lm_flat = lm_w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            println!("STD_DEBUG LM_HEAD.weight first 10: {:?}", &lm_flat[..10.min(lm_flat.len())]);
            let w_min = lm_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let w_max = lm_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("STD_DEBUG LM_HEAD.weight min={:.4}, max={:.4}", w_min, w_max);
        }

        // 4. LM Head (Final Ghost)
        // Dump pre-lm_head hidden state for last position (first call only)
        {
            static DUMP_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !DUMP_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let seq_len = x.dim(1)?;
                let last_hidden = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
                let last_flat = last_hidden.flatten_all()?.to_vec1::<f32>()?;
                println!("HIDDEN_DUMP last_pos first 20: {:?}", &last_flat[..20.min(last_flat.len())]);
                // Write to file for Python comparison
                use std::io::Write;
                if let Ok(mut outf) = std::fs::File::create("/tmp/oom_hidden_state.bin") {
                    for v in &last_flat {
                        let _ = outf.write_all(&v.to_le_bytes());
                    }
                    println!("HIDDEN_DUMP: Wrote {} values to /tmp/oom_hidden_state.bin", last_flat.len());
                }
                // Also dump lm_head weight first row
                let lm_w = self.lm_head.weight.materialize()?;
                let lm_row0 = lm_w.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
                println!("LM_HEAD row[0] first 20: {:?}", &lm_row0[..20.min(lm_row0.len())]);
                if let Ok(mut outf) = std::fs::File::create("/tmp/oom_lm_head_row0.bin") {
                    for v in &lm_row0 {
                        let _ = outf.write_all(&v.to_le_bytes());
                    }
                }
            }
        }
        let logits = self.lm_head.forward(&x)?;

        if do_debug_std {
            println!("STD_DEBUG LOGITS shape: {:?}", logits.shape());
        }

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
                        println!("📌 CUDA_VISIBLE_DEVICES={} - Single GPU mode", val);
                        false
                    } else {
                        println!("📌 CUDA_VISIBLE_DEVICES={} - Multi-GPU available", val);
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
                                println!("🎮 Secondary GPU {} detected + cross-device transfer OK!", secondary_idx);
                                Some(secondary_idx)
                            }
                            Err(e) => {
                                println!("⚠️ Secondary GPU {} detected but cross-device transfer failed: {:?}", secondary_idx, e);
                                println!("   → Falling back to single GPU mode. Upgrade NVIDIA driver for dual GPU support.");
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
                        println!("✅ CUDA device {} initialized successfully!", idx);
                        println!("🔧 PRIMARY DEVICE DEBUG: {:?}", d);
                        d
                    }
                    Err(e) => {
                        println!("⚠️ CUDA device {} failed: {:?}, falling back to CPU", idx, e);
                        Device::Cpu
                    }
                }
            }
            None => {
                println!("ℹ️ No GPU index specified, using CPU");
                Device::Cpu
            }
        };

        let secondary_device = secondary_gpu.and_then(|idx| Device::new_cuda(idx).ok());
        println!("🔧 SECONDARY_GPU: {:?}, SECONDARY_DEVICE: {:?}", secondary_gpu, secondary_device);

        // Load Tokenizer - detect model type from path
        let tokenizer = if let Some(path) = tokenizer_path {
            Tokenizer::from_file(path).map_err(|e| e.to_string())?
        } else {
            let api = Api::new()?;
            // Select tokenizer based on model name
            let model_name = model_path.unwrap_or("");
            let repo_name = if model_name.contains("qwen") || model_name.contains("humotica") {
                println!("📝 Loading Qwen tokenizer...");
                "Qwen/Qwen2.5-7B-Instruct"
            } else if model_name.contains("llama3") || model_name.contains("llama-3") {
                println!("📝 Loading Llama 3 tokenizer...");
                "meta-llama/Llama-3.2-1B"
            } else {
                println!("📝 Loading TinyLlama tokenizer (default)...");
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
                 println!("🦙 LLAMA 3.3 70B DETECTED: Using Llama config (Hidden: 8192, Vocab: 128256).");
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
                 println!("🦙 QWEN 32B DETECTED: OomLlama's brain! (Hidden: 5120, 64 layers, Vocab: 152064)");
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
                 println!("🦙 QWEN 0.5B DETECTED: Using Qwen 0.5B config (Hidden: 896, 24 layers, Vocab: 151936).");
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
                 println!("🦙 QWEN 3B DETECTED: Using Qwen 3B config (Hidden: 2048, 36 layers, Vocab: 151936).");
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
                 println!("🦙 QWEN 7B DETECTED: Using Qwen 7B config (Hidden: 3584, 28 layers, Vocab: 152064).");
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
                 println!("🐘 QWEN 72B DETECTED: Using Qwen config (Hidden: 8192, Vocab: 152064).");
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
                 println!("📜 Found local config: {:?}", config_path);
                 let l_config: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
                 Config::from(l_config)
             } else {
                 println!("🐑 Falling back to TinyLlama default config.");
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
                println!("🚀 TURBO MODE: Qwen 32B config detected - enabling KV-Cache + Flash Attention!");
                TurboConfig::qwen32b_dual3060()
            } else {
                // Generic config based on detected model
                println!("🚀 TURBO MODE: Generic config - enabling KV-Cache + Flash Attention!");
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
            println!("⚠️ TURBO disabled (CPU mode) - KV-Cache requires GPU");
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
        println!("💉 Brain: Absorbing new context ({} bytes)...", context.len());
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
        println!("🔍 FULL_PROMPT: {:?}", &full_prompt);
        let tokens = self.tokenizer.encode(full_prompt, true).map_err(|e| e.to_string())?.get_ids().to_vec();
        let prompt_len = tokens.len();
        // println!("🔍 ALL TOKEN IDS ({}): {:?}", tokens.len(), &tokens);

        // Check if turbo mode is available
        if let Some(ref mut turbo) = self.turbo {
            // 🚀 TURBO MODE: KV-Cache enabled inference
            println!("🚀 TURBO Inference starting ({} prompt tokens)...", prompt_len);

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

            // Debug: Show first token's logits with TOP 10 + decoded text
            {
                let logits_flat = last_logits.flatten_all()?;
                let logits_vec = logits_flat.to_vec1::<f32>()?;
                let vocab_size = logits_vec.len();
                let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_val = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let mean_val: f32 = logits_vec.iter().sum::<f32>() / vocab_size as f32;
                let variance: f32 = logits_vec.iter().map(|v| (v - mean_val).powi(2)).sum::<f32>() / vocab_size as f32;
                let std_val = variance.sqrt();
                let nan_count = logits_vec.iter().filter(|v| v.is_nan()).count();
                let inf_count = logits_vec.iter().filter(|v| v.is_infinite()).count();
                println!("LOGIT_STATS [prompt]: vocab={}, min={:.4}, max={:.4}, mean={:.4}, std={:.4}, nan={}, inf={}",
                    vocab_size, min_val, max_val, mean_val, std_val, nan_count, inf_count);
                // Show top 10 by value with decoded text
                let mut indexed: Vec<(usize, &f32)> = logits_vec.iter().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                println!("TOP_10_LOGITS [prompt] (argmax token_id={}):", next_token);
                for (rank, (tid, logit_val)) in indexed.iter().take(10).enumerate() {
                    let decoded = self.tokenizer.decode(&[*tid as u32], false)
                        .unwrap_or_else(|_| format!("<decode_err:{}>", tid));
                    println!("  TOP_TOKEN rank={} id={:>6} logit={:>10.4} text={:?}",
                        rank, tid, logit_val, decoded);
                }
                // Also show bottom 5
                println!("BOTTOM_5_LOGITS [prompt]:");
                for (rank, (tid, logit_val)) in indexed.iter().rev().take(5).enumerate() {
                    let decoded = self.tokenizer.decode(&[*tid as u32], false)
                        .unwrap_or_else(|_| format!("<decode_err:{}>", tid));
                    println!("  BOT_TOKEN rank={} id={:>6} logit={:>10.4} text={:?}",
                        rank, tid, logit_val, decoded);
                }
            }

            // Autoregressive generation with KV-cache
            for i in 0..max_tokens {
                // Check EOS
                if next_token == 2 || next_token == 151643 || next_token == 128001 || next_token == 151645 {
                    println!("🛑 EOS token reached at step {}", i);
                    break;
                }

                generated_tokens.push(next_token);

                if i % 5 == 0 {
                    let cached = turbo.seq_len();
                    println!("🚀 Generated {} tokens (KV-cache: {} entries)...", i + 1, cached);
                }

                // TURBO: Only pass the NEW token - KV-cache handles history!
                let new_token_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward_turbo(&new_token_tensor, turbo)?;

                // Get next token
                let seq_len = logits.dim(1)?;
                let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
                let next_token_idx = last_logits.argmax(candle_core::D::Minus1)?;
                next_token = next_token_idx.squeeze(0)?.to_scalar::<u32>()?;

                // Per-token logit debug (every 10 tokens + first 3)
                if i < 3 || i % 10 == 0 {
                    let logits_flat = last_logits.flatten_all()?;
                    let logits_vec = logits_flat.to_vec1::<f32>()?;
                    let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min_val = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mean_val: f32 = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
                    let decoded = self.tokenizer.decode(&[next_token], false)
                        .unwrap_or_else(|_| format!("<err:{}>", next_token));
                    println!("LOGIT_STATS [step {}]: min={:.4}, max={:.4}, mean={:.4}, token_id={}, text={:?}",
                        i, min_val, max_val, mean_val, next_token, decoded);
                    // Top 3 for each step
                    let mut indexed: Vec<(usize, &f32)> = logits_vec.iter().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                    for (rank, (tid, logit_val)) in indexed.iter().take(3).enumerate() {
                        let dec = self.tokenizer.decode(&[*tid as u32], false)
                            .unwrap_or_else(|_| format!("<err:{}>", tid));
                        println!("  TOP_TOKEN [step {}] rank={} id={} logit={:.4} text={:?}",
                            i, rank, tid, logit_val, dec);
                    }
                }
            }

            println!("🚀 TURBO complete! {} tokens generated (prompt: {}, KV-cache: {} entries)",
                     generated_tokens.len(), prompt_len, turbo.seq_len());

            let output = self.tokenizer.decode(&generated_tokens, true).map_err(|e| e.to_string())?;
            Ok(output)
        } else {
            // Standard mode (no KV-cache)
            println!("🤖 Ghost Inference starting (no turbo)...");

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

                // Debug: print logits info for first 3 tokens + every 10th
                if i < 3 || i % 10 == 0 {
                    let logits_flat = last_logits.flatten_all()?;
                    let logits_vec = logits_flat.to_vec1::<f32>()?;
                    let vocab_size = logits_vec.len();
                    let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min_val = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mean_val: f32 = logits_vec.iter().sum::<f32>() / vocab_size as f32;
                    let variance: f32 = logits_vec.iter().map(|v| (v - mean_val).powi(2)).sum::<f32>() / vocab_size as f32;
                    let std_val = variance.sqrt();
                    let nan_count = logits_vec.iter().filter(|v| v.is_nan()).count();
                    let inf_count = logits_vec.iter().filter(|v| v.is_infinite()).count();
                    let decoded = self.tokenizer.decode(&[next_token], false)
                        .unwrap_or_else(|_| format!("<decode_err:{}>", next_token));
                    println!("LOGIT_STATS [step {}]: vocab={}, min={:.4}, max={:.4}, mean={:.4}, std={:.4}, nan={}, inf={}, argmax={}, text={:?}",
                        i, vocab_size, min_val, max_val, mean_val, std_val, nan_count, inf_count, next_token, decoded);
                    // Show logit for token 17 ("2") - reference says this should be ~34.67
                    if i == 0 && logits_vec.len() > 17 {
                        println!("TOKEN17_DEBUG: logit[17]={:.6} (reference: ~34.67)", logits_vec[17]);
                        // Also show logits for related tokens
                        let check_tokens = vec![(17, "2"), (19, "4"), (220, " "), (785, "The"), (151645, "<|im_end|>")];
                        for (tid, name) in &check_tokens {
                            if *tid < logits_vec.len() {
                                println!("  logit[{}] ({}) = {:.6}", tid, name, logits_vec[*tid]);
                            }
                        }
                    }
                    // Show top 10 by value with decoded text
                    let mut indexed: Vec<(usize, &f32)> = logits_vec.iter().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                    println!("TOP_10_LOGITS [step {}]:", i);
                    for (rank, (tid, logit_val)) in indexed.iter().take(10).enumerate() {
                        let dec = self.tokenizer.decode(&[*tid as u32], false)
                            .unwrap_or_else(|_| format!("<decode_err:{}>", tid));
                        println!("  TOP_TOKEN [step {}] rank={} id={:>6} logit={:>10.4} text={:?}",
                            i, rank, tid, logit_val, dec);
                    }
                    if i == 0 {
                        // Also show bottom 5 on first token
                        println!("BOTTOM_5_LOGITS [step 0]:");
                        for (rank, (tid, logit_val)) in indexed.iter().rev().take(5).enumerate() {
                            let dec = self.tokenizer.decode(&[*tid as u32], false)
                                .unwrap_or_else(|_| format!("<decode_err:{}>", tid));
                            println!("  BOT_TOKEN rank={} id={:>6} logit={:>10.4} text={:?}",
                                rank, tid, logit_val, dec);
                        }
                    }
                }

                if next_token == 2 || next_token == 151643 || next_token == 128001 || next_token == 151645 {
                    println!("🛑 EOS token reached at step {}", i);
                    break;
                }

                generated_tokens.push(next_token);
                println!("🔢 Token {}: {} (seq_len={})", i, next_token, current_input.dim(1)?);

                // Without KV-cache: must pass full sequence each time
                let new_token = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
                current_input = Tensor::cat(&[&current_input, &new_token], 1)?;

                if i % 5 == 0 {
                    println!("🔤 Generated {} tokens...", i + 1);
                }
            }

            println!("✅ Generation complete ({} tokens).", generated_tokens.len());
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
