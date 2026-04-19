//! Multi-GPU Support for OomLlama
//!
//! Splits layers across multiple GPUs for larger models.
//! P520: Dual RTX 3060 (24GB total VRAM)
//!
//! Strategy:
//! - 7B model: Single GPU (fits in 12GB)
//! - 32B model: Split layers 50/50 across GPUs
//! - 70B model: Split layers + lazy loading per GPU
//!
//! One love, one fAmIly! 🦙🦙

use candle_core::{Device, Tensor};
use std::sync::Arc;

/// Multi-GPU configuration
#[derive(Debug, Clone)]
pub struct MultiGPUConfig {
    /// Number of available GPUs
    pub num_gpus: usize,
    /// VRAM per GPU in bytes
    pub vram_per_gpu: Vec<usize>,
    /// Layer assignment: layer_idx -> gpu_idx
    pub layer_to_gpu: Vec<usize>,
    /// Whether to use async transfers
    pub async_transfer: bool,
}

impl MultiGPUConfig {
    /// Auto-detect GPUs using nvidia-smi
    pub fn auto_detect() -> Self {
        // Try nvidia-smi for GPU count
        let num_gpus = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).lines().count())
            .unwrap_or(1)
            .max(1);

        let vram_per_gpu = (0..num_gpus)
            .map(|_| 12 * 1024 * 1024 * 1024) // 12GB default per RTX 3060
            .collect();

        Self {
            num_gpus,
            vram_per_gpu,
            layer_to_gpu: Vec::new(),
            async_transfer: cfg!(feature = "cuda"),
        }
    }

    /// Create config with explicit GPU count
    pub fn with_gpus(num_gpus: usize) -> Self {
        let vram_per_gpu = (0..num_gpus)
            .map(|_| 12 * 1024 * 1024 * 1024)
            .collect();

        Self {
            num_gpus,
            vram_per_gpu,
            layer_to_gpu: Vec::new(),
            async_transfer: cfg!(feature = "cuda"),
        }
    }

    /// Create layer assignment for a model
    pub fn assign_layers(&mut self, num_layers: usize) {
        if self.num_gpus <= 1 {
            // Single GPU: all layers on GPU 0
            self.layer_to_gpu = vec![0; num_layers];
        } else {
            // Multi-GPU: round-robin or 50/50 split
            let layers_per_gpu = num_layers / self.num_gpus;
            self.layer_to_gpu = (0..num_layers)
                .map(|i| (i / layers_per_gpu).min(self.num_gpus - 1))
                .collect();
        }
    }

    /// Get device for a specific layer
    pub fn device_for_layer(&self, layer_idx: usize) -> Device {
        let gpu_idx = self.layer_to_gpu.get(layer_idx).copied().unwrap_or(0);
        Device::new_cuda(gpu_idx).unwrap_or(Device::Cpu)
    }
}

/// Multi-GPU tensor manager
pub struct MultiGPUManager {
    config: MultiGPUConfig,
    /// Pinned tensors per GPU (for hot layers)
    pinned: Vec<Vec<Arc<Tensor>>>,
}

impl MultiGPUManager {
    pub fn new(config: MultiGPUConfig) -> Self {
        let pinned = (0..config.num_gpus).map(|_| Vec::new()).collect();
        Self { config, pinned }
    }

    /// Transfer tensor to appropriate GPU for layer
    pub fn transfer_for_layer(&self, tensor: &Tensor, layer_idx: usize) -> candle_core::Result<Tensor> {
        let device = self.config.device_for_layer(layer_idx);
        tensor.to_device(&device)
    }

    /// Pin a tensor on a specific GPU (keeps it in VRAM)
    pub fn pin_tensor(&mut self, tensor: Tensor, gpu_idx: usize) -> Arc<Tensor> {
        let arc = Arc::new(tensor);
        if gpu_idx < self.pinned.len() {
            self.pinned[gpu_idx].push(arc.clone());
        }
        arc
    }

    /// Get config
    pub fn config(&self) -> &MultiGPUConfig {
        &self.config
    }

    /// Report VRAM usage
    pub fn report_usage(&self) {
        println!("🖥️ Multi-GPU Status:");
        println!("   GPUs: {}", self.config.num_gpus);
        for (i, pinned) in self.pinned.iter().enumerate() {
            println!("   GPU {}: {} pinned tensors", i, pinned.len());
        }
    }
}

/// Layer distribution strategy
#[derive(Debug, Clone, Copy)]
pub enum LayerStrategy {
    /// All layers on single GPU
    SingleGPU(usize),
    /// Split layers evenly across GPUs
    EvenSplit,
    /// First half on GPU 0, second half on GPU 1
    HalfSplit,
    /// Custom assignment
    Custom,
}

impl LayerStrategy {
    /// Apply strategy to create layer->GPU mapping
    pub fn apply(&self, num_layers: usize, num_gpus: usize) -> Vec<usize> {
        match self {
            LayerStrategy::SingleGPU(gpu) => vec![*gpu; num_layers],
            LayerStrategy::EvenSplit => {
                let per_gpu = num_layers / num_gpus;
                (0..num_layers)
                    .map(|i| (i / per_gpu).min(num_gpus - 1))
                    .collect()
            }
            LayerStrategy::HalfSplit => {
                let half = num_layers / 2;
                (0..num_layers)
                    .map(|i| if i < half { 0 } else { 1.min(num_gpus - 1) })
                    .collect()
            }
            LayerStrategy::Custom => vec![0; num_layers],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_strategy() {
        // 32 layers, 2 GPUs
        let strategy = LayerStrategy::HalfSplit;
        let mapping = strategy.apply(32, 2);

        assert_eq!(mapping.len(), 32);
        assert_eq!(mapping[0], 0);  // First half on GPU 0
        assert_eq!(mapping[16], 1); // Second half on GPU 1
    }
}
