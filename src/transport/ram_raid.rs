//! RAM-RAID — 70B Always Fits
//!
//! Distributes model layers across machines via the Genesis Highway (10Gbps).
//! Even layers on local machine, odd layers on remote — transparent via userfaultfd.
//!
//! Uses trust-kernel's RamRaidController + ClusterMuxClient for the heavy lifting.
//! This module provides the OomLlama-specific layer distribution logic.
//!
//! ```text
//! Laptop (32GB RAM):
//!   Layer 0, 2, 4... → local RAM (Spaceshuttle)
//!   Layer 1, 3, 5... → page fault → ClusterMux → P520 → decrypt → inject
//!
//! P520 (dual RTX 3060):
//!   All layers local (GPU inference)
//!   Acts as remote RAM for laptop via ClusterMuxServer
//! ```

use std::sync::Arc;
use std::collections::HashMap;

use tibet_trust_kernel::ram_raid::{RaidConfig, RamRaidController, RaidStripe, FaultResult};
use tibet_trust_kernel::cluster_mux::ClusterMuxClient;
use tibet_trust_kernel::cluster_transport::BlockStore;

/// Layer distribution strategy
#[derive(Debug, Clone)]
pub enum DistributionStrategy {
    /// All layers local (single machine, e.g., P520 with GPU)
    AllLocal,
    /// Even/odd split across two machines
    EvenOdd {
        local_host: String,
        remote_host: String,
        remote_endpoint: String,
    },
    /// Custom layer ranges per host
    Custom(Vec<HostRange>),
}

/// A range of layers assigned to a specific host
#[derive(Debug, Clone)]
pub struct HostRange {
    pub host: String,
    pub endpoint: String,
    pub start_layer: u32,
    pub end_layer: u32,
}

/// Model RAID — distributes a model's layers across the cluster
pub struct ModelRaid {
    /// Distribution strategy
    strategy: DistributionStrategy,
    /// Layer → host mapping
    layer_hosts: HashMap<u32, String>,
    /// Total layers in the model
    total_layers: u32,
    /// .aint identity for cluster auth
    identity: String,
    /// Block size for RAID (typically = page size = 4096)
    block_size: usize,
}

impl ModelRaid {
    /// Create RAID config for a model
    pub fn new(
        total_layers: u32,
        identity: &str,
        strategy: DistributionStrategy,
    ) -> Self {
        let mut layer_hosts = HashMap::new();

        match &strategy {
            DistributionStrategy::AllLocal => {
                // All layers local — no RAID needed
            }
            DistributionStrategy::EvenOdd { local_host, remote_host, .. } => {
                for layer in 0..total_layers {
                    if layer % 2 == 0 {
                        layer_hosts.insert(layer, local_host.clone());
                    } else {
                        layer_hosts.insert(layer, remote_host.clone());
                    }
                }
            }
            DistributionStrategy::Custom(ranges) => {
                for range in ranges {
                    for layer in range.start_layer..=range.end_layer {
                        layer_hosts.insert(layer, range.host.clone());
                    }
                }
            }
        }

        Self {
            strategy,
            layer_hosts,
            total_layers,
            identity: identity.to_string(),
            block_size: 4096,
        }
    }

    /// Check if a layer is local
    pub fn is_local(&self, layer: u32) -> bool {
        match &self.strategy {
            DistributionStrategy::AllLocal => true,
            DistributionStrategy::EvenOdd { .. } => layer % 2 == 0,
            DistributionStrategy::Custom(ranges) => {
                // Check if any range with "localhost" contains this layer
                ranges.iter().any(|r|
                    layer >= r.start_layer && layer <= r.end_layer
                    && r.host == "localhost"
                )
            }
        }
    }

    /// Check if a layer is remote
    pub fn is_remote(&self, layer: u32) -> bool {
        !self.is_local(layer)
    }

    /// Get the host for a specific layer
    pub fn host_for_layer(&self, layer: u32) -> Option<&str> {
        self.layer_hosts.get(&layer).map(|s| s.as_str())
    }

    /// Get the remote endpoint for a specific layer
    pub fn endpoint_for_layer(&self, layer: u32) -> Option<&str> {
        match &self.strategy {
            DistributionStrategy::EvenOdd { remote_endpoint, .. } if layer % 2 != 0 => {
                Some(remote_endpoint.as_str())
            }
            DistributionStrategy::Custom(ranges) => {
                ranges.iter()
                    .find(|r| layer >= r.start_layer && layer <= r.end_layer)
                    .map(|r| r.endpoint.as_str())
            }
            _ => None,
        }
    }

    /// Count local vs remote layers
    pub fn distribution_summary(&self) -> (u32, u32) {
        let local = (0..self.total_layers).filter(|l| self.is_local(*l)).count() as u32;
        let remote = self.total_layers - local;
        (local, remote)
    }

    /// Create a RamRaidController for this distribution
    ///
    /// The controller manages the actual page fault handling and
    /// remote block fetching via ClusterMux.
    pub fn create_controller(
        &self,
        arena_size: usize,
    ) -> RamRaidController {
        let mut config = RaidConfig::new(
            arena_size,
            "inference",
            &self.identity,
        );

        // Configure remote RAM B if we have a remote host
        match &self.strategy {
            DistributionStrategy::EvenOdd { remote_host, remote_endpoint, .. } => {
                config = config.with_remote_ram_b(remote_host, remote_endpoint);
            }
            DistributionStrategy::Custom(ranges) => {
                // Use first remote range as RAM B
                if let Some(remote) = ranges.iter().find(|r| r.host != "localhost") {
                    config = config.with_remote_ram_b(&remote.host, &remote.endpoint);
                }
            }
            _ => {}
        }

        RamRaidController::new(config)
    }

    /// Estimate memory usage per machine
    pub fn memory_estimate(&self, model_size_bytes: u64) -> MemoryEstimate {
        let bytes_per_layer = model_size_bytes / self.total_layers as u64;
        let (local_layers, remote_layers) = self.distribution_summary();

        MemoryEstimate {
            total_model_bytes: model_size_bytes,
            local_bytes: bytes_per_layer * local_layers as u64,
            remote_bytes: bytes_per_layer * remote_layers as u64,
            local_layers,
            remote_layers,
            bytes_per_layer,
        }
    }

    /// Print distribution report
    pub fn report(&self, model_name: &str, model_size_bytes: u64) -> String {
        let est = self.memory_estimate(model_size_bytes);
        let (local, remote) = self.distribution_summary();

        format!(
            r#"
=== RAM-RAID: {} ===
  Strategy:       {:?}
  Total layers:   {}
  Local layers:   {} ({:.1} GB)
  Remote layers:  {} ({:.1} GB)
  Savings:        {:.0}% RAM reduction on local machine
  Block size:     {} bytes
  Identity:       {}
"#,
            model_name,
            match &self.strategy {
                DistributionStrategy::AllLocal => "All Local".to_string(),
                DistributionStrategy::EvenOdd { local_host, remote_host, .. } =>
                    format!("Even/Odd ({} + {})", local_host, remote_host),
                DistributionStrategy::Custom(ranges) =>
                    format!("Custom ({} ranges)", ranges.len()),
            },
            self.total_layers,
            local, est.local_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            remote, est.remote_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            if est.total_model_bytes > 0 {
                (1.0 - est.local_bytes as f64 / est.total_model_bytes as f64) * 100.0
            } else { 0.0 },
            self.block_size,
            self.identity,
        )
    }
}

/// Memory distribution estimate
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub total_model_bytes: u64,
    pub local_bytes: u64,
    pub remote_bytes: u64,
    pub local_layers: u32,
    pub remote_layers: u32,
    pub bytes_per_layer: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_even_odd_distribution() {
        let raid = ModelRaid::new(
            80, // 70B model has ~80 layers
            "root_idd.aint",
            DistributionStrategy::EvenOdd {
                local_host: "p520".to_string(),
                remote_host: "dl360".to_string(),
                remote_endpoint: "10.0.100.1:4430".to_string(),
            },
        );

        assert!(raid.is_local(0));
        assert!(raid.is_remote(1));
        assert!(raid.is_local(2));
        assert!(raid.is_remote(79));

        let (local, remote) = raid.distribution_summary();
        assert_eq!(local, 40);
        assert_eq!(remote, 40);
    }

    #[test]
    fn test_all_local() {
        let raid = ModelRaid::new(32, "test.aint", DistributionStrategy::AllLocal);

        for layer in 0..32 {
            assert!(raid.is_local(layer));
        }

        let (local, remote) = raid.distribution_summary();
        assert_eq!(local, 32);
        assert_eq!(remote, 0);
    }

    #[test]
    fn test_memory_estimate_70b() {
        let raid = ModelRaid::new(
            80,
            "root_idd.aint",
            DistributionStrategy::EvenOdd {
                local_host: "p520".to_string(),
                remote_host: "dl360".to_string(),
                remote_endpoint: "10.0.100.1:4430".to_string(),
            },
        );

        // 70B model ≈ 35GB quantized
        let est = raid.memory_estimate(35 * 1024 * 1024 * 1024);
        // Each machine holds ~17.5GB
        assert!(est.local_bytes < 20 * 1024 * 1024 * 1024);
        assert!(est.remote_bytes < 20 * 1024 * 1024 * 1024);
        assert_eq!(est.local_layers, 40);
        assert_eq!(est.remote_layers, 40);
    }
}
