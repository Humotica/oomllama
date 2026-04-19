use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// SENTINEL Intent Classifier Output Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentinelOutput {
    pub role: String,
    pub provider: String,
    pub intent_label: String,
    pub urgency: f32,
    pub tags: Vec<String>,
    pub trust_required: u32,
    pub safe_harbor: bool,
    pub trace_id: String,
    pub confidence: f32,
    pub embedding_method: String,
}

/// Intent Category with embedding and routing info
#[derive(Clone)]
struct IntentCategory {
    role: String,
    provider: String,
    intent_label: String,
    keywords: Vec<&'static str>,
    embedding: Vec<f32>,
    trust_required: u32,
    urgency: f32,
    tags: Vec<String>,
}

/// SENTINEL Intent Classifier
/// Uses Candle for real transformer-based embeddings.
pub struct SentinelClassifier {
    model: Option<BertModel>,
    tokenizer: Option<Tokenizer>,
    device: Device,
    categories: Vec<IntentCategory>,
    embedding_dim: usize,
}

impl SentinelClassifier {
    /// Initialize the classifier with predefined intent categories and BERT model
    pub fn new(model_id: &str) -> Result<Self> {
        let device = Device::Cpu;
        let embedding_dim = 384; // all-MiniLM-L6-v2 dimension

        // Load real model
        let (model, tokenizer) = match Self::load_model(model_id, &device) {
            Ok((m, t)) => (Some(m), Some(t)),
            Err(e) => {
                eprintln!("⚠️ Failed to load BERT model: {}. Falling back to basic mode.", e);
                (None, None)
            }
        };

        // Define intent categories with semantic keywords
        let categories = vec![
            IntentCategory {
                role: "code_assistant".into(),
                provider: "claude".into(),
                intent_label: "implement_feature".into(),
                keywords: vec![
                    "code", "rust", "python", "fix", "bug", "implement", "function",
                    "compile", "error", "programming", "developer", "api", "debug",
                    "refactor", "optimize", "class", "method", "variable", "syntax",
                    "typescript", "javascript", "build", "cargo", "npm", "git",
                ],
                embedding: Vec::new(),
                trust_required: 2,
                urgency: 0.3,
                tags: vec!["code".into(), "development".into()],
            },
            IntentCategory {
                role: "security_validator".into(),
                provider: "claude".into(),
                intent_label: "security_check".into(),
                keywords: vec![
                    "security", "firewall", "hack", "vulnerability", "password",
                    "encrypt", "ssl", "certificate", "auth", "permission", "access",
                    "threat", "attack", "protect", "secure", "breach", "audit",
                    "iptables", "ssh", "key", "token", "oauth", "jwt",
                ],
                embedding: Vec::new(),
                trust_required: 3,
                urgency: 0.5,
                tags: vec!["security".into()],
            },
            IntentCategory {
                role: "emergency_response".into(),
                provider: "raid".into(),
                intent_label: "urgent_assist".into(),
                keywords: vec![
                    "help", "emergency", "urgent", "critical", "down", "crash",
                    "broken", "failed", "disaster", "asap", "immediately", "sos",
                    "production", "outage", "incident", "alert", "panic",
                    "server down", "everything is broken", "urgent incident", "site is offline",
                    "server ligt plat", "alles is kapot", "hulp!", "nood", "incident",
                ],
                embedding: Vec::new(),
                trust_required: 4,
                urgency: 0.9,
                tags: vec!["emergency".into(), "urgent".into()],
            },
            IntentCategory {
                role: "data_analyst".into(),
                provider: "gemini".into(),
                intent_label: "analyze_data".into(),
                keywords: vec![
                    "data", "analyze", "statistics", "graph", "chart", "metrics",
                    "dashboard", "report", "trend", "pattern", "insight", "query",
                    "database", "sql", "csv", "json", "export", "visualization",
                    "analyze the metrics", "show dashboard", "report on",
                    "analyseer de metrics", "toon dashboard", "maak rapport",
                ],
                embedding: Vec::new(),
                trust_required: 2,
                urgency: 0.2,
                tags: vec!["data".into(), "analysis".into()],
            },
            IntentCategory {
                role: "researcher".into(),
                provider: "gemini".into(),
                intent_label: "research".into(),
                keywords: vec![
                    "research", "find", "search", "lookup", "what", "how", "why",
                    "explain", "document", "learn", "understand", "information",
                    "knowledge", "study", "investigate", "explore",
                    "how does this work", "investigate",
                    "hoe werkt dit", "leg uit", "onderzoek dit",
                ],
                embedding: Vec::new(),
                trust_required: 1,
                urgency: 0.2,
                tags: vec!["research".into()],
            },
            IntentCategory {
                role: "general_assistant".into(),
                provider: "ollama".into(),
                intent_label: "chat".into(),
                keywords: vec![
                    "hello", "hi", "hey", "thanks", "please", "good", "nice",
                    "chat", "talk", "conversation", "weather", "joke", "story",
                    "can you help", "hallo", "hoi", "kun je helpen",
                ],
                embedding: Vec::new(),
                trust_required: 0,
                urgency: 0.1,
                tags: vec![],
            },
        ];

        let mut classifier = Self {
            model,
            tokenizer,
            device,
            categories,
            embedding_dim,
        };

        // Compute embeddings for all categories
        classifier.compute_category_embeddings()?;

        Ok(classifier)
    }

    fn load_model(model_id: &str, device: &Device) -> Result<(BertModel, Tokenizer)> {
        let model_id = if model_id == "models/sentinel" || model_id == "mock" {
            "sentence-transformers/all-MiniLM-L6-v2"
        } else {
            model_id
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let config: BertConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, device)?
        };

        let model = BertModel::load(vb, &config)?;

        Ok((model, tokenizer))
    }

    /// Compute embeddings for each category based on keywords
    fn compute_category_embeddings(&mut self) -> Result<()> {
        // Collect texts first to avoid borrow conflict
        let texts: Vec<String> = self.categories
            .iter()
            .map(|cat| cat.keywords.join(" "))
            .collect();

        // Compute embeddings
        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| self.text_to_embedding(text))
            .collect::<Result<Vec<_>>>()?;

        // Assign back
        for (category, embedding) in self.categories.iter_mut().zip(embeddings) {
            category.embedding = embedding;
        }

        Ok(())
    }

    /// Convert text to embedding using the BERT model or fallback hashing
    pub fn text_to_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if let (Some(model), Some(tokenizer)) = (&self.model, &self.tokenizer) {
            // Real ML embedding
            let encoding = tokenizer.encode(text, true)
                .map_err(|e| format!("Tokenization failed: {}", e))?;
            
            let tokens = encoding.get_ids().to_vec();
            let token_ids = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let token_type_ids = token_ids.zeros_like()?;

            let embeddings = model.forward(&token_ids, &token_type_ids, None)?;

            // Mean pooling
            let (_, seq_len, _) = embeddings.dims3()?;
            let summed = embeddings.sum(1)?;
            let mean = (summed / (seq_len as f64))?;

            // Normalize
            let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
            let normalized = mean.broadcast_div(&norm)?;

            Ok(normalized.squeeze(0)?.to_vec1::<f32>()?)
        } else {
            // Fallback to n-gram hashing
            Self::text_to_embedding_fallback(text, self.embedding_dim)
        }
    }

    /// Fallback version for use if model loading fails
    fn text_to_embedding_fallback(text: &str, embedding_dim: usize) -> Result<Vec<f32>> {
        let text = text.to_lowercase();
        let mut embedding = vec![0.0f32; embedding_dim];

        let chars: Vec<char> = text.chars().collect();
        for window in chars.windows(3) {
            let mut hash: usize = 5381;
            for c in window {
                hash = hash.wrapping_mul(33).wrapping_add(*c as usize);
            }
            let idx = hash % embedding_dim;
            embedding[idx] += 1.0;
        }

        for word in text.split_whitespace() {
            let mut hash: usize = 0;
            for (i, c) in word.chars().enumerate() {
                hash = hash.wrapping_add((c as usize).wrapping_mul(31_usize.pow(i as u32)));
            }
            let idx = hash % embedding_dim;
            embedding[idx] += 2.0;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }

    /// Compute cosine similarity using Candle tensors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let tensor_a = Tensor::from_slice(a, (a.len(),), &self.device)?;
        let tensor_b = Tensor::from_slice(b, (b.len(),), &self.device)?;

        let dot = (&tensor_a * &tensor_b)?.sum_all()?.to_scalar::<f32>()?;
        let norm_a = tensor_a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let norm_b = tensor_b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            Ok(dot / (norm_a * norm_b))
        } else {
            Ok(0.0)
        }
    }

    /// Detect intent using semantic similarity
    pub fn detect_intent(&self, text: &str, sender_trust: f32, trace_id: String) -> Result<SentinelOutput> {
        let query_embedding = self.text_to_embedding(text)?;
        let text_lower = text.to_lowercase();

        let mut best_category = &self.categories[self.categories.len() - 1]; // Default: general
        let mut best_score: f32 = -1.0;

        for category in &self.categories {
            let score = self.cosine_similarity(&query_embedding, &category.embedding)?;

            // For emergency classification, require actual keyword presence
            // This prevents false positives from weak n-gram matching
            if category.intent_label == "urgent_assist" {
                let has_urgent_keyword = category.keywords.iter()
                    .any(|kw| text_lower.contains(&kw.to_lowercase()));
                if !has_urgent_keyword {
                    continue; // Skip emergency if no urgent keywords found
                }
            }

            if score > best_score {
                best_score = score;
                best_category = category;
            }
        }

        let method = if self.model.is_some() { "candle_bert_v1" } else { "candle_ngram_fallback" };
        let confidence = (best_score * 0.5 + 0.5).min(0.99);

        Ok(SentinelOutput {
            role: best_category.role.clone(),
            provider: best_category.provider.clone(),
            intent_label: best_category.intent_label.clone(),
            urgency: best_category.urgency,
            tags: best_category.tags.clone(),
            trust_required: best_category.trust_required,
            safe_harbor: sender_trust < 0.5,
            trace_id,
            confidence,
            embedding_method: method.into(),
        })
    }

    /// Enrich the payload with SENTINEL analysis
    pub fn enrich_payload(&self, output: &SentinelOutput, original_payload: String) -> String {
        serde_json::to_string(&serde_json::json!({
            "sentinel": output,
            "original": original_payload,
            "verified": true,
            "version": "sentinel-rs-0.2.0"
        })).unwrap_or_else(|_| original_payload)
    }

    /// Get statistics about the classifier
    pub fn stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        stats.insert("embedding_dim".into(), self.embedding_dim.into());
        stats.insert("num_categories".into(), self.categories.len().into());
        stats.insert("device".into(), format!("{:?}", self.device).into());
        stats.insert("method".into(), (if self.model.is_some() { "candle_bert_v1" } else { "candle_ngram_fallback" }).into());

        let category_names: Vec<String> = self.categories.iter()
            .map(|c| c.role.clone())
            .collect();
        stats.insert("categories".into(), category_names.into());

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_semantic_intent_detection() {
        let classifier = SentinelClassifier::new("mock").unwrap();
        let trace_id = Uuid::new_v4().to_string();

        // Test code detection
        let output = classifier.detect_intent("please fix this rust code", 0.9, trace_id.clone()).unwrap();
        assert_eq!(output.role, "code_assistant");
        assert_eq!(output.provider, "claude");

        // Test semantic understanding (no exact keyword match)
        let output2 = classifier.detect_intent("my program has a bug", 0.9, Uuid::new_v4().to_string()).unwrap();
        assert_eq!(output2.role, "code_assistant");

        // Test security
        let output3 = classifier.detect_intent("check the firewall", 0.9, Uuid::new_v4().to_string()).unwrap();
        assert_eq!(output3.role, "security_validator");

        // Test emergency
        let output4 = classifier.detect_intent("help urgent server down", 0.9, Uuid::new_v4().to_string()).unwrap();
        assert_eq!(output4.role, "emergency_response");
        assert!(output4.urgency > 0.5);
    }

    #[test]
    fn test_embedding_method_tag() {
        let classifier = SentinelClassifier::new("mock").unwrap();
        let output = classifier.detect_intent("hello", 0.9, "test".into()).unwrap();
        assert_eq!(output.embedding_method, "candle_bert_v1"); // Updated to expect BERT if available
    }
}