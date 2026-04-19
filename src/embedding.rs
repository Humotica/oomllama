//! NEURAL EMBEDDING ENGINE - GPU 1 (RTX 3060)
//!
//! Converts pure knowledge into mathematical vectors using Candle.
//! Dedicated to GPU 1 for background indexing.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use std::sync::Arc;

pub struct EmbeddingEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingEngine {
    /// Initialize the engine on a specific GPU (default 1)
    pub fn new(gpu_index: usize) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Try CUDA first, fall back to CPU if no GPU found
        let device = match Device::new_cuda(gpu_index) {
            Ok(d) => {
                println!("🚀 Embedding Engine active on GPU {}", gpu_index);
                d
            },
            Err(_) => {
                println!("⚠️ GPU {} not available, falling back to CPU", gpu_index);
                Device::Cpu
            }
        };

        let model_id = "sentence-transformers/all-MiniLM-L6-v2";
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let config: BertConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)?
        };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Convert text to a vector (384 dimensions)
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        
        let tokens = encoding.get_ids().to_vec();
        let token_ids = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;

        // Mean pooling
        let (_, seq_len, _) = embeddings.dims3()?;
        let summed = embeddings.sum(1)?;
        let mean = (summed / (seq_len as f64))?;

        // Normalize
        let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = mean.broadcast_div(&norm)?;

        Ok(normalized.squeeze(0)?.to_vec1::<f32>()?)
    }
}
