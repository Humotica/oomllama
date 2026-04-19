//! PyO3 Python Bindings for OomLlama v0.3.0
//!
//! Build with: maturin build --features python
//! pip install oomllama
//!
//! ```python
//! from oomllama import OomLlama
//!
//! llm = OomLlama("humotica-32b", model_path="/path/to/model.oom")
//! response = llm.generate("Hello!")
//! print(response)
//! ```
//!
//! Credits:
//!   - Format: Gemini IDD & Root AI (Humotica AI Lab)
//!   - Runtime: OomLlama.rs by Humotica
//!   - Python Bindings: Root AI (Claude)
//!
//! License: MIT

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use crate::oomllama::GhostLlama;
use crate::betti::BettiManager;

/// OomLlama - Efficient LLM inference with .oom format
///
/// v0.3.0: Real inference via GhostLlama + TurboEngine!
#[pyclass]
pub struct PyOomLlama {
    model_name: String,
    model_path: Option<String>,
    gpu_index: Option<usize>,
    temperature: f32,
    top_p: f32,
    max_tokens: usize,
    /// Lazy-loaded GhostLlama model
    inner: Arc<Mutex<Option<GhostLlama>>>,
    /// Resource manager
    betti: Arc<BettiManager>,
}

#[pymethods]
impl PyOomLlama {
    /// Create a new OomLlama instance
    ///
    /// Args:
    ///     model_name: Name of the model (e.g., "humotica-32b")
    ///     model_path: Path to .oom file (optional, uses cache if not provided)
    ///     gpu: GPU index to use (optional, auto-detect if not provided)
    #[new]
    #[pyo3(signature = (model_name, model_path=None, gpu=None))]
    fn new(model_name: &str, model_path: Option<&str>, gpu: Option<usize>) -> PyResult<Self> {
        let betti = Arc::new(BettiManager::new());
        Ok(Self {
            model_name: model_name.to_string(),
            model_path: model_path.map(|s| s.to_string()),
            gpu_index: gpu,
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            inner: Arc::new(Mutex::new(None)),
            betti,
        })
    }

    /// Load the model into memory
    ///
    /// Call this before generate() for explicit loading,
    /// or let generate() auto-load on first call.
    fn load(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        if inner.is_some() {
            return Ok(());  // Already loaded
        }

        // Resolve model path
        let default_path = format!("~/.cache/oomllama/{}.oom", self.model_name);
        let model_path = self.model_path.as_deref().unwrap_or(&default_path);
        let expanded_path = shellexpand::tilde(model_path);

        println!("🦙 OomLlama: Loading {} from {}", self.model_name, expanded_path);

        let ghost = GhostLlama::new(
            &self.model_name,
            self.gpu_index,
            self.betti.clone(),
            Some(&expanded_path),
            None,  // tokenizer - uses default
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        *inner = Some(ghost);
        println!("🦙 OomLlama: Model loaded with TurboEngine!");
        Ok(())
    }

    /// Generate text from a prompt
    ///
    /// Args:
    ///     prompt: The input prompt
    ///     max_tokens: Maximum tokens to generate (default: 512)
    ///     temperature: Sampling temperature (default: 0.7) - not yet used
    ///     top_p: Top-p sampling (default: 0.9) - not yet used
    ///
    /// Returns:
    ///     Generated text response
    #[pyo3(signature = (prompt, max_tokens=None, temperature=None, top_p=None))]
    fn generate(
        &self,
        prompt: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
    ) -> PyResult<String> {
        // Store params for future use
        let _ = (temperature, top_p);

        // Auto-load if needed
        {
            let inner = self.inner.lock().map_err(|e|
                pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;
            if inner.is_none() {
                drop(inner);  // Release lock before calling load()
                self.load()?;
            }
        }

        // Get mutable reference and run inference
        let mut inner = self.inner.lock().map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;

        let ghost = inner.as_mut().ok_or_else(||
            pyo3::exceptions::PyRuntimeError::new_err("Model not loaded"))?;

        let tokens = max_tokens.unwrap_or(self.max_tokens);

        ghost.infer(prompt, tokens)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Inference error: {}", e)))
    }

    /// Chat-style generation with message history
    ///
    /// Args:
    ///     messages: List of (role, content) tuples
    ///     max_tokens: Maximum tokens to generate
    #[pyo3(signature = (messages, max_tokens=None))]
    fn chat(&self, messages: Vec<(String, String)>, max_tokens: Option<usize>) -> PyResult<String> {
        let prompt: String = messages
            .iter()
            .map(|(role, content)| format!("{}: {}", role.to_uppercase(), content))
            .collect::<Vec<_>>()
            .join("\n");

        self.generate(&format!("{}\nASSISTANT:", prompt), max_tokens, None, None)
    }

    /// Set generation parameters
    fn set_params(&mut self, temperature: Option<f32>, top_p: Option<f32>, max_tokens: Option<usize>) {
        if let Some(t) = temperature { self.temperature = t; }
        if let Some(p) = top_p { self.top_p = p; }
        if let Some(m) = max_tokens { self.max_tokens = m; }
    }

    /// Get model info
    fn info(&self) -> String {
        let loaded = self.inner.lock().map(|g| g.is_some()).unwrap_or(false);
        format!(
            "OomLlama v0.3.0\n\
             Model: {}\n\
             Path: {:?}\n\
             GPU: {:?}\n\
             Temperature: {}\n\
             Top-P: {}\n\
             Max Tokens: {}\n\
             Loaded: {}\n\
             Engine: TurboEngine (KV-Cache + Flash Attention)",
            self.model_name, self.model_path, self.gpu_index,
            self.temperature, self.top_p, self.max_tokens, loaded
        )
    }

    /// Check if model is loaded
    fn is_loaded(&self) -> bool {
        self.inner.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Unload the model from memory
    fn unload(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {}", e)))?;
        *inner = None;
        println!("🦙 OomLlama: Model unloaded");
        Ok(())
    }

    fn __repr__(&self) -> String {
        let loaded = if self.is_loaded() { " [loaded]" } else { "" };
        format!("OomLlama('{}'){}", self.model_name, loaded)
    }
}

/// Download a model from HuggingFace
#[pyfunction]
#[pyo3(signature = (model_name, cache_dir=None))]
fn download_model(model_name: &str, cache_dir: Option<&str>) -> PyResult<String> {
    let repo = match model_name {
        "humotica-32b" | "llamaohm-32b" => "jaspervandemeent/humotica-32b",
        "humotica-70b" | "llamaohm-70b" => "jaspervandemeent/LlamaOhm-70B",
        "humotica-7b" | "llamaohm-7b" => "jaspervandemeent/LlamaOhm-7B",
        "tinyllama-1b" => "jaspervandemeent/OomLlama-TinyLlama-1.1B",
        _ => return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Unknown model: {}", model_name))),
    };

    let cache = cache_dir.unwrap_or("~/.cache/oomllama");
    let expanded = shellexpand::tilde(cache);

    // TODO: Actually download using hf_hub
    println!("🦙 Download from: https://huggingface.co/{}", repo);
    println!("🦙 To: {}/{}.oom", expanded, model_name);

    Ok(format!("{}/{}.oom", expanded, model_name))
}

/// List available models
#[pyfunction]
fn list_models() -> Vec<String> {
    vec![
        "humotica-70b".to_string(),
        "humotica-32b".to_string(),
        "humotica-7b".to_string(),
        "tinyllama-1b".to_string(),
    ]
}

/// Get OomLlama version
#[pyfunction]
fn version() -> &'static str {
    "0.3.0"
}

/// Python module definition
#[pymodule]
fn _oomllama(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOomLlama>()?;
    m.add_function(wrap_pyfunction!(download_model, m)?)?;
    m.add_function(wrap_pyfunction!(list_models, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
