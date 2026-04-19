//! TIBET Bridge — Provenance for every inference
//!
//! Every OomLlama inference generates a TIBET token with:
//! - ERIN: model name + prompt hash
//! - ERAAN: user.aint identity + JIS clearance
//! - EROMHEEN: device + GPU + layer distribution
//! - ERACHTER: output tokens + latency + cache hits
//!
//! Uses the internal `tibet.rs` TibetFactory (proven, >300 token types)
//! augmented with inference-specific semantics.

use crate::tibet::{TibetToken, TibetFactory, TokenType};
use serde_json::json;
use sha2::{Digest, Sha256};

/// Inference provenance — creates TIBET tokens per inference call
pub struct InferenceProvenance {
    factory: TibetFactory,
    model_name: String,
    device_info: DeviceContext,
}

/// Device context for EROMHEEN
#[derive(Debug, Clone)]
pub struct DeviceContext {
    pub hostname: String,
    pub gpu_id: Option<u8>,
    pub layer_distribution: LayerDistribution,
}

/// How layers are distributed across machines
#[derive(Debug, Clone)]
pub enum LayerDistribution {
    /// All layers local
    Local,
    /// RAM-RAID: even layers local, odd layers remote
    RamRaid {
        local_host: String,
        remote_host: String,
    },
    /// Multi-machine: layer ranges per host
    Cluster(Vec<(String, u32, u32)>), // (host, start_layer, end_layer)
}

impl InferenceProvenance {
    pub fn new(model_name: &str, device_info: DeviceContext) -> Self {
        let factory = TibetFactory::new("oomllama")
            .with_hwid(device_info.hostname.clone());
        Self {
            factory,
            model_name: model_name.to_string(),
            device_info,
        }
    }

    /// Mint a TIBET token for an inference call
    pub fn mint_inference_token(
        &self,
        prompt: &str,
        user_aint: &str,
        clearance: &str,
        output_tokens: usize,
        latency_ms: f64,
        cache_hits: u64,
    ) -> TibetToken {
        // ERIN: what went in
        let prompt_hash = hex::encode(Sha256::digest(prompt.as_bytes()));

        let erin = json!({
            "model": self.model_name,
            "prompt_hash": &prompt_hash[..16],
            "prompt_tokens": prompt.split_whitespace().count(),
        });

        // Build the token
        let mut token = TibetToken::new(
            TokenType::Action,
            "oomllama",
            erin,
            format!("Inference: {} → {} tokens", self.model_name, output_tokens),
        );

        // ERAAN: who + clearance
        token.eraan = vec![
            format!("user:{}", user_aint),
            format!("clearance:{}", clearance),
        ];

        // EROMHEEN: device context
        token.eromheen = json!({
            "device": self.device_info.hostname,
            "gpu": self.device_info.gpu_id,
            "layer_distribution": match &self.device_info.layer_distribution {
                LayerDistribution::Local => "local".to_string(),
                LayerDistribution::RamRaid { local_host, remote_host } =>
                    format!("ram-raid:{}+{}", local_host, remote_host),
                LayerDistribution::Cluster(hosts) =>
                    format!("cluster:{}", hosts.len()),
            },
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cache_hits": cache_hits,
        });

        token
    }
}
