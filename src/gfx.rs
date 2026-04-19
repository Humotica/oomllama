//! BETTI-GFX - GPU Resource Visualization
//!
//! Real-time GPU monitoring and visualization for AETHER.
//! Shows what's happening on the P520 (dual RTX 3060).
//!
//! One love, one fAmIly!

use serde::{Deserialize, Serialize};
use std::process::Command;
use chrono::{DateTime, Utc};

/// GPU information from nvidia-smi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_free_mb: u64,
    pub utilization_gpu: u32,
    pub utilization_memory: u32,
    pub temperature_c: u32,
    pub power_draw_w: f32,
    pub power_limit_w: f32,
}

impl GpuInfo {
    /// Memory utilization percentage
    pub fn memory_percent(&self) -> f32 {
        if self.memory_total_mb > 0 {
            (self.memory_used_mb as f32 / self.memory_total_mb as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Generate ASCII bar for memory usage
    pub fn memory_bar(&self, width: usize) -> String {
        let filled = ((self.memory_percent() / 100.0) * width as f32) as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }

    /// Generate ASCII bar for GPU utilization
    pub fn gpu_bar(&self, width: usize) -> String {
        let filled = ((self.utilization_gpu as f32 / 100.0) * width as f32) as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }
}

/// Complete BETTI-GFX status
#[derive(Debug, Clone, Serialize)]
pub struct GfxStatus {
    pub timestamp: DateTime<Utc>,
    pub node: String,
    pub gpus: Vec<GpuInfo>,
    pub total_vram_gb: f32,
    pub used_vram_gb: f32,
    pub avg_utilization: f32,
    pub avg_temperature: f32,
    pub total_power_w: f32,
    pub ascii_display: String,
}

/// BETTI-GFX Monitor
pub struct GfxMonitor {
    pub node: String,
    pub ssh_host: Option<String>,
}

impl GfxMonitor {
    /// Create monitor for local GPUs
    pub fn local() -> Self {
        Self {
            node: "local".to_string(),
            ssh_host: None,
        }
    }

    /// Create monitor for remote node via SSH
    pub fn remote(node: &str, ssh_host: &str) -> Self {
        Self {
            node: node.to_string(),
            ssh_host: Some(ssh_host.to_string()),
        }
    }

    /// Query GPU status
    pub fn query(&self) -> Result<GfxStatus, String> {
        let nvidia_output = self.run_nvidia_smi()?;
        let gpus = self.parse_nvidia_smi(&nvidia_output)?;

        let total_vram: u64 = gpus.iter().map(|g| g.memory_total_mb).sum();
        let used_vram: u64 = gpus.iter().map(|g| g.memory_used_mb).sum();
        let total_power: f32 = gpus.iter().map(|g| g.power_draw_w).sum();

        let avg_util = if gpus.is_empty() {
            0.0
        } else {
            gpus.iter().map(|g| g.utilization_gpu as f32).sum::<f32>() / gpus.len() as f32
        };

        let avg_temp = if gpus.is_empty() {
            0.0
        } else {
            gpus.iter().map(|g| g.temperature_c as f32).sum::<f32>() / gpus.len() as f32
        };

        let ascii_display = self.generate_ascii(&gpus);

        Ok(GfxStatus {
            timestamp: Utc::now(),
            node: self.node.clone(),
            gpus,
            total_vram_gb: total_vram as f32 / 1024.0,
            used_vram_gb: used_vram as f32 / 1024.0,
            avg_utilization: avg_util,
            avg_temperature: avg_temp,
            total_power_w: total_power,
            ascii_display,
        })
    }

    /// Run nvidia-smi (local or remote)
    fn run_nvidia_smi(&self) -> Result<String, String> {
        let query = "index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit";

        let output = if let Some(host) = &self.ssh_host {
            Command::new("ssh")
                .arg("-o").arg("ConnectTimeout=2")
                .arg("-o").arg("StrictHostKeyChecking=no")
                .arg(host)
                .arg(format!("nvidia-smi --query-gpu={} --format=csv,noheader,nounits", query))
                .output()
        } else {
            Command::new("nvidia-smi")
                .arg(format!("--query-gpu={}", query))
                .arg("--format=csv,noheader,nounits")
                .output()
        };

        match output {
            Ok(out) if out.status.success() => {
                String::from_utf8(out.stdout)
                    .map_err(|e| format!("UTF-8 error: {}", e))
            }
            Ok(out) => {
                Err(format!("nvidia-smi failed: {}", String::from_utf8_lossy(&out.stderr)))
            }
            Err(e) => Err(format!("Failed to run nvidia-smi: {}", e))
        }
    }

    /// Parse nvidia-smi CSV output
    fn parse_nvidia_smi(&self, output: &str) -> Result<Vec<GpuInfo>, String> {
        let mut gpus = Vec::new();

        for line in output.trim().lines() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 10 {
                continue;
            }

            gpus.push(GpuInfo {
                index: parts[0].parse().unwrap_or(0),
                name: parts[1].to_string(),
                memory_total_mb: parts[2].parse().unwrap_or(0),
                memory_used_mb: parts[3].parse().unwrap_or(0),
                memory_free_mb: parts[4].parse().unwrap_or(0),
                utilization_gpu: parts[5].parse().unwrap_or(0),
                utilization_memory: parts[6].parse().unwrap_or(0),
                temperature_c: parts[7].parse().unwrap_or(0),
                power_draw_w: parts[8].parse().unwrap_or(0.0),
                power_limit_w: parts[9].parse().unwrap_or(0.0),
            });
        }

        Ok(gpus)
    }

    /// Generate ASCII visualization
    fn generate_ascii(&self, gpus: &[GpuInfo]) -> String {
        let mut output = String::new();

        output.push_str("┌─────────────────────────────────────────────────────┐\n");
        output.push_str("│              BETTI-GFX Resource Monitor             │\n");
        output.push_str("├─────────────────────────────────────────────────────┤\n");

        if gpus.is_empty() {
            output.push_str("│  No GPUs detected or connection failed              │\n");
        } else {
            for gpu in gpus {
                output.push_str(&format!("│  GPU {} │ {} │\n",
                    gpu.index,
                    &gpu.name[..gpu.name.len().min(38)]
                ));
                output.push_str(&format!("│  VRAM: {} {:>5}/{:>5} MB ({:.1}%) │\n",
                    gpu.memory_bar(20),
                    gpu.memory_used_mb,
                    gpu.memory_total_mb,
                    gpu.memory_percent()
                ));
                output.push_str(&format!("│  GPU:  {} {:>3}% │ {:>3}°C │ {:>3.0}W │\n",
                    gpu.gpu_bar(20),
                    gpu.utilization_gpu,
                    gpu.temperature_c,
                    gpu.power_draw_w
                ));
                output.push_str("├─────────────────────────────────────────────────────┤\n");
            }

            // Summary
            let total: u64 = gpus.iter().map(|g| g.memory_total_mb).sum();
            let used: u64 = gpus.iter().map(|g| g.memory_used_mb).sum();
            let util_avg = gpus.iter().map(|g| g.utilization_gpu).sum::<u32>() as f32 / gpus.len() as f32;

            output.push_str(&format!("│  TOTAL: {:.1} / {:.1} GB  │  AVG: {:.0}%  │\n",
                used as f32 / 1024.0,
                total as f32 / 1024.0,
                util_avg
            ));
        }

        output.push_str("└─────────────────────────────────────────────────────┘\n");
        output
    }
}

/// Simulated GPU data for when real GPUs aren't available
pub fn simulated_status() -> GfxStatus {
    let gpus = vec![
        GpuInfo {
            index: 0,
            name: "NVIDIA GeForce RTX 3060".to_string(),
            memory_total_mb: 12288,
            memory_used_mb: 4096,
            memory_free_mb: 8192,
            utilization_gpu: 35,
            utilization_memory: 33,
            temperature_c: 52,
            power_draw_w: 85.0,
            power_limit_w: 170.0,
        },
        GpuInfo {
            index: 1,
            name: "NVIDIA GeForce RTX 3060".to_string(),
            memory_total_mb: 12288,
            memory_used_mb: 2048,
            memory_free_mb: 10240,
            utilization_gpu: 15,
            utilization_memory: 17,
            temperature_c: 48,
            power_draw_w: 45.0,
            power_limit_w: 170.0,
        },
    ];

    let total_vram: u64 = gpus.iter().map(|g| g.memory_total_mb).sum();
    let used_vram: u64 = gpus.iter().map(|g| g.memory_used_mb).sum();
    let total_power: f32 = gpus.iter().map(|g| g.power_draw_w).sum();
    let avg_util = gpus.iter().map(|g| g.utilization_gpu as f32).sum::<f32>() / gpus.len() as f32;
    let avg_temp = gpus.iter().map(|g| g.temperature_c as f32).sum::<f32>() / gpus.len() as f32;

    let monitor = GfxMonitor::local();
    let ascii_display = monitor.generate_ascii(&gpus);

    GfxStatus {
        timestamp: Utc::now(),
        node: "P520-simulated".to_string(),
        gpus,
        total_vram_gb: total_vram as f32 / 1024.0,
        used_vram_gb: used_vram as f32 / 1024.0,
        avg_utilization: avg_util,
        avg_temperature: avg_temp,
        total_power_w: total_power,
        ascii_display,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_bar() {
        let gpu = GpuInfo {
            index: 0,
            name: "Test GPU".to_string(),
            memory_total_mb: 12288,
            memory_used_mb: 6144,
            memory_free_mb: 6144,
            utilization_gpu: 50,
            utilization_memory: 50,
            temperature_c: 60,
            power_draw_w: 100.0,
            power_limit_w: 200.0,
        };

        assert_eq!(gpu.memory_percent(), 50.0);
        let bar = gpu.memory_bar(10);
        assert!(bar.contains("█████")); // Half filled
    }

    #[test]
    fn test_simulated_status() {
        let status = simulated_status();
        assert_eq!(status.gpus.len(), 2);
        assert!(status.total_vram_gb > 20.0); // 24GB total
    }
}
