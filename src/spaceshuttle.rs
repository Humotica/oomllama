//! Spaceshuttle — Identity-Gated Lazy Model Loading
//!
//! The core innovation: mmap 20GB .oom file, physically loaded: 0 bytes.
//! Page fault on first access → decrypt → decompress → inject.
//! After 10 prompts: hot layers in RAM, cold layers on disk.
//!
//! Uses tibet-store-mmu's MmuArena (userfaultfd) for transparent
//! page fault handling with AES-256-GCM encrypted pages.
//!
//! ```text
//! .oom file (encrypted + zstd + JIS clearance per layer)
//!   → Spaceshuttle mmap (virtual: 20GB, physical: 0 bytes)
//!   → First inference: page fault on layer 0
//!     → userfaultfd catches fault
//!     → SessionKey decrypt (5µs)
//!     → zstd decompress
//!     → inject into page
//!     → app resumes — layer is now in RAM
//!   → Second inference: layer 0 = cache hit, layer 1 = fault
//!   → After warmup: hot path entirely in RAM
//! ```
//!
//! Result: instant startup, progressive loading, encrypted at rest.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bridge::jis::{LayerPolicy, Clearance};

// Import from tibet-store-mmu only — it re-uses trust-kernel types internally.
// We avoid importing trust-kernel types directly to prevent dual-version conflicts.
use tibet_store_mmu::{
    MmuArena, MmuConfig, MmuStats, FillMode,
    seal_pages_compressed, CompressedSealResult, mmu_claim,
};

/// Spaceshuttle model loader — lazy encrypted model loading via userfaultfd
pub struct Spaceshuttle {
    /// Layer metadata: name → mapping
    layer_map: HashMap<String, LayerMapping>,
    /// Cache hit tracking per layer
    access_counts: HashMap<String, AtomicU64>,
    /// Total page faults served
    total_faults: AtomicU64,
    /// Total cache hits (pages already in RAM)
    total_cache_hits: AtomicU64,
    /// JIS layer policy
    layer_policy: LayerPolicy,
    /// Model name
    model_name: String,
    /// Total model size in bytes
    model_size_bytes: u64,
}

/// Mapping of a model layer to pages in the MmuArena
#[derive(Debug, Clone)]
pub struct LayerMapping {
    /// Layer index in the transformer
    pub layer_index: u32,
    /// Starting page in the arena
    pub page_start: usize,
    /// Number of pages this layer spans
    pub page_count: usize,
    /// JIS clearance required for this layer
    pub clearance: Clearance,
    /// Size in bytes
    pub size_bytes: u64,
    /// Whether this layer has been accessed (warm)
    pub warm: bool,
}

/// Stats for the Spaceshuttle loader
#[derive(Debug, Clone)]
pub struct SpaceshuttleStats {
    /// Total page faults handled
    pub total_faults: u64,
    /// Cache hits (pages already resident)
    pub cache_hits: u64,
    /// Number of warm layers (accessed at least once)
    pub warm_layers: usize,
    /// Number of cold layers (never accessed)
    pub cold_layers: usize,
    /// Total model size
    pub model_size_bytes: u64,
    /// Estimated resident size (warm layers only)
    pub resident_bytes: u64,
    /// Cache hit rate
    pub hit_rate: f64,
}

/// Result of sealing layer data (compression + encryption stats)
#[derive(Debug)]
pub struct SealedLayerStats {
    /// Number of pages sealed
    pub page_count: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Total encrypted size in bytes
    pub encrypted_bytes: usize,
    /// Total original size in bytes
    pub original_bytes: usize,
}

impl Spaceshuttle {
    /// Create a new Spaceshuttle for a model
    pub fn new(model_name: &str, total_layers: u32) -> Self {
        Self {
            layer_map: HashMap::new(),
            access_counts: HashMap::new(),
            total_faults: AtomicU64::new(0),
            total_cache_hits: AtomicU64::new(0),
            layer_policy: LayerPolicy::default_for_layers(total_layers),
            model_name: model_name.to_string(),
            model_size_bytes: 0,
        }
    }

    /// Register a layer's page mapping
    pub fn register_layer(
        &mut self,
        name: &str,
        layer_index: u32,
        page_start: usize,
        page_count: usize,
        size_bytes: u64,
    ) {
        let clearance = self.layer_policy.required_clearance(layer_index);
        self.layer_map.insert(name.to_string(), LayerMapping {
            layer_index,
            page_start,
            page_count,
            clearance,
            size_bytes,
            warm: false,
        });
        self.access_counts.insert(name.to_string(), AtomicU64::new(0));
        self.model_size_bytes += size_bytes;
    }

    /// Seal raw layer data into encrypted + compressed pages
    ///
    /// Returns the sealed result (from tibet-store-mmu) and stats.
    /// The CompressedSealResult contains the encrypted blocks ready
    /// for FillMode::CompressedEncryptedRestore.
    pub fn seal_layer_data(
        data: &[u8],
        page_size: usize,
        clearance_name: &str,
        source: &str,
        zstd_level: i32,
    ) -> (CompressedSealResult, SealedLayerStats) {
        // Split data into page-sized chunks
        let pages: Vec<Vec<u8>> = data
            .chunks(page_size)
            .map(|chunk| {
                let mut page = vec![0u8; page_size];
                page[..chunk.len()].copy_from_slice(chunk);
                page
            })
            .collect();

        // Map clearance name to tibet-store-mmu's ClearanceLevel
        let clearance = Self::parse_clearance(clearance_name);

        // Compress + encrypt via tibet-store-mmu
        let result = seal_pages_compressed(&pages, clearance, source, zstd_level);

        let stats = SealedLayerStats {
            page_count: result.blocks.len(),
            compression_ratio: result.compression_ratio,
            encrypted_bytes: result.total_encrypted,
            original_bytes: result.total_original,
        };

        (result, stats)
    }

    /// Record a layer access (for cache tracking)
    pub fn record_access(&mut self, layer_name: &str) {
        if let Some(mapping) = self.layer_map.get_mut(layer_name) {
            if mapping.warm {
                self.total_cache_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                mapping.warm = true;
                self.total_faults.fetch_add(1, Ordering::Relaxed);
            }
        }
        if let Some(counter) = self.access_counts.get(layer_name) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Check if caller has clearance for a specific layer
    pub fn check_layer_access(&self, layer_name: &str, caller_clearance: &Clearance) -> bool {
        if let Some(mapping) = self.layer_map.get(layer_name) {
            self.layer_policy.can_access_layer(caller_clearance, mapping.layer_index)
        } else {
            false // Unknown layer = denied
        }
    }

    /// Get layers sorted by access frequency (hottest first)
    pub fn hot_layers(&self) -> Vec<(&str, u64)> {
        let mut layers: Vec<(&str, u64)> = self.access_counts
            .iter()
            .map(|(name, count)| (name.as_str(), count.load(Ordering::Relaxed)))
            .collect();
        layers.sort_by(|a, b| b.1.cmp(&a.1));
        layers
    }

    /// Get cold layers (never accessed)
    pub fn cold_layers(&self) -> Vec<&str> {
        self.layer_map
            .iter()
            .filter(|(_, mapping)| !mapping.warm)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get comprehensive stats
    pub fn stats(&self) -> SpaceshuttleStats {
        let warm_layers = self.layer_map.values().filter(|m| m.warm).count();
        let cold_layers = self.layer_map.values().filter(|m| !m.warm).count();
        let resident_bytes: u64 = self.layer_map.values()
            .filter(|m| m.warm)
            .map(|m| m.size_bytes)
            .sum();
        let total_faults = self.total_faults.load(Ordering::Relaxed);
        let cache_hits = self.total_cache_hits.load(Ordering::Relaxed);
        let total_accesses = total_faults + cache_hits;

        SpaceshuttleStats {
            total_faults,
            cache_hits,
            warm_layers,
            cold_layers,
            model_size_bytes: self.model_size_bytes,
            resident_bytes,
            hit_rate: if total_accesses > 0 {
                cache_hits as f64 / total_accesses as f64
            } else {
                0.0
            },
        }
    }

    /// Create an MmuArena with encrypted model data
    ///
    /// This is the production entry point: creates a userfaultfd-backed
    /// arena where page faults trigger decrypt → decompress → inject.
    ///
    /// The `sealed` parameter comes from `seal_layer_data()`.
    ///
    /// Returns None if userfaultfd is not available (needs root or CAP_SYS_PTRACE).
    pub fn create_arena(
        arena_size: usize,
        sealed: CompressedSealResult,
        identity: &str,
        clearance_name: &str,
        use_hugepages: bool,
    ) -> Option<MmuArena> {
        let clearance = Self::parse_clearance(clearance_name);
        let claim = mmu_claim(identity, clearance.clone());

        let fill_mode = FillMode::CompressedEncryptedRestore {
            sealed_pages: sealed.blocks,
            original_sizes: sealed.original_sizes,
            claim,
            clearance,
        };

        let config = if use_hugepages {
            MmuConfig::hugepages(arena_size, fill_mode)
        } else {
            MmuConfig::normal(arena_size, fill_mode)
        };

        MmuArena::new(config)
    }

    /// Map clearance name string to tibet-store-mmu's ClearanceLevel
    fn parse_clearance(name: &str) -> tibet_trust_kernel::bifurcation::ClearanceLevel {
        // tibet-store-mmu re-exports ClearanceLevel from trust-kernel
        // We use string-based mapping to avoid cross-crate type conflicts
        match name {
            "unclassified" | "public" => tibet_trust_kernel::bifurcation::ClearanceLevel::Unclassified,
            "restricted" => tibet_trust_kernel::bifurcation::ClearanceLevel::Restricted,
            "confidential" => tibet_trust_kernel::bifurcation::ClearanceLevel::Confidential,
            "secret" => tibet_trust_kernel::bifurcation::ClearanceLevel::Secret,
            "top-secret" | "topsecret" => tibet_trust_kernel::bifurcation::ClearanceLevel::TopSecret,
            _ => tibet_trust_kernel::bifurcation::ClearanceLevel::Unclassified,
        }
    }

    /// Print a human-readable status
    pub fn status_report(&self) -> String {
        let stats = self.stats();
        format!(
            r#"
=== SPACESHUTTLE STATUS: {} ===
  Model size:     {:.1} GB
  Resident:       {:.1} GB ({:.0}%)
  Warm layers:    {} / {}
  Cold layers:    {}
  Page faults:    {}
  Cache hits:     {}
  Hit rate:       {:.1}%
  Status:         {}
"#,
            self.model_name,
            stats.model_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.resident_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            if stats.model_size_bytes > 0 {
                stats.resident_bytes as f64 / stats.model_size_bytes as f64 * 100.0
            } else { 0.0 },
            stats.warm_layers,
            stats.warm_layers + stats.cold_layers,
            stats.cold_layers,
            stats.total_faults,
            stats.cache_hits,
            stats.hit_rate * 100.0,
            if stats.hit_rate > 0.9 { "WARMED UP" }
            else if stats.hit_rate > 0.5 { "WARMING" }
            else { "COLD START" },
        )
    }
}

// Re-export useful types — access via oomllama::spaceshuttle::*
pub use tibet_store_mmu::userfaultfd_available;
pub use tibet_store_mmu::format_ns;
pub use tibet_store_mmu::percentile;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spaceshuttle_layer_registration() {
        let mut shuttle = Spaceshuttle::new("test-7b", 32);

        // Register some layers
        shuttle.register_layer("model.layers.0.self_attn.q_proj", 0, 0, 10, 4096);
        shuttle.register_layer("model.layers.0.self_attn.k_proj", 0, 10, 5, 2048);
        shuttle.register_layer("model.layers.16.self_attn.q_proj", 16, 15, 10, 4096);

        let stats = shuttle.stats();
        assert_eq!(stats.warm_layers, 0);
        assert_eq!(stats.cold_layers, 3);
        assert_eq!(stats.model_size_bytes, 10240);
    }

    #[test]
    fn test_spaceshuttle_access_tracking() {
        let mut shuttle = Spaceshuttle::new("test-7b", 32);
        shuttle.register_layer("layer.0", 0, 0, 10, 4096);

        // First access = fault (cold)
        shuttle.record_access("layer.0");
        let stats = shuttle.stats();
        assert_eq!(stats.total_faults, 1);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.warm_layers, 1);

        // Second access = cache hit (warm)
        shuttle.record_access("layer.0");
        let stats = shuttle.stats();
        assert_eq!(stats.total_faults, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_spaceshuttle_clearance_check() {
        let mut shuttle = Spaceshuttle::new("test-64layer", 64);
        shuttle.register_layer("base.layer.0", 0, 0, 10, 4096);
        shuttle.register_layer("secret.layer.50", 50, 100, 10, 4096);

        // Unclassified can access base layers
        assert!(shuttle.check_layer_access("base.layer.0", &Clearance::Unclassified));
        // Unclassified cannot access secret layers
        assert!(!shuttle.check_layer_access("secret.layer.50", &Clearance::Unclassified));
        // Secret can access secret layers
        assert!(shuttle.check_layer_access("secret.layer.50", &Clearance::Secret));
    }

    #[test]
    fn test_seal_layer_data() {
        let page_size = 4096;
        let raw_data = vec![42u8; page_size * 3]; // 3 pages of data

        let (result, stats) = Spaceshuttle::seal_layer_data(
            &raw_data,
            page_size,
            "confidential",
            "test-model",
            3,
        );

        assert_eq!(result.blocks.len(), 3);
        assert_eq!(stats.page_count, 3);
        assert!(stats.compression_ratio > 1.0); // Repetitive data compresses well
        assert!(stats.encrypted_bytes < stats.original_bytes);
    }
}
