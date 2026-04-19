//! CopperEngine — Distributed Inference Co-Processing
//!
//! Named after the Amiga Copper and Atari Blitter: co-processors that
//! handled bulk work independently from the main CPU.
//!
//! # The Problem
//!
//! Your inference CPU spends cycles on:
//! 1. Matrix multiplies (attention, feed-forward) — THE ACTUAL WORK
//! 2. Decrypting model pages (AES-256-GCM)       — overhead
//! 3. Decompressing pages (zstd)                  — overhead
//! 4. Memory copies                               — overhead
//!
//! Meanwhile, idle machines in your network do nothing.
//!
//! # The Solution: Three Modes
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ MODE 1: Copper (pre-decrypt)                           │
//! │                                                         │
//! │   Idle box decrypts + decompresses pages BEFORE the    │
//! │   inference CPU needs them. Like the Amiga Copper       │
//! │   preparing display lines ahead of the raster beam.     │
//! │                                                         │
//! │   Laptop asks for layer 5 → DL360 already has it       │
//! │   decrypted and ready → zero decrypt overhead           │
//! ├─────────────────────────────────────────────────────────┤
//! │ MODE 2: Blitter (remote matmul)                        │
//! │                                                         │
//! │   Remote CPU does the full matrix multiply for its      │
//! │   assigned layers. Sends back only the result:          │
//! │   hidden_state = 4096 floats = 16KB                     │
//! │   instead of transferring 100MB+ of weights.            │
//! │                                                         │
//! │   Like the Atari Blitter doing memory block operations  │
//! │   with logic — the CPU just collects the result.        │
//! ├─────────────────────────────────────────────────────────┤
//! │ MODE 3: Full (autonomous layer processing)             │
//! │                                                         │
//! │   Remote node owns its layers completely:               │
//! │   decrypt → decompress → matmul → send result           │
//! │   The main CPU never touches those layers at all.       │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Configuration
//!
//! Point to any machine with spare CPU cycles:
//!
//! ```toml
//! [[copper.nodes]]
//! name = "dl360"
//! endpoint = "10.0.100.1:4431"
//! mode = "blitter"
//! layers = "odd"
//!
//! [[copper.nodes]]
//! name = "old-laptop"
//! endpoint = "192.168.1.50:4431"
//! mode = "copper"
//! layers = "40-60"
//!
//! [[copper.nodes]]
//! name = "rpi-cluster"
//! endpoint = "10.0.100.10:4431"
//! mode = "copper"
//! layers = "0-20"
//! ```
//!
//! Like Docker Compose, but for inference compute.
//! Every dusty CPU in your closet becomes an inference accelerator.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// How a CopperNode participates in inference
#[derive(Debug, Clone, PartialEq)]
pub enum CopperMode {
    /// Pre-decrypt + decompress pages ahead of inference (Amiga Copper)
    /// Remote does: encrypted page → AES decrypt → zstd decompress → cache
    /// Main CPU gets: plaintext ready to use, zero decrypt overhead
    Copper,

    /// Remote matrix multiply (Atari Blitter)
    /// Remote does: load weights → matmul(input, weights) → send result
    /// Main CPU gets: 16KB hidden state instead of 100MB weights
    Blitter,

    /// Full autonomous layer processing (Copper + Blitter combined)
    /// Remote does: decrypt → decompress → matmul → send result
    /// Main CPU: never touches these layers at all
    Full,
}

/// Which layers a node is responsible for
#[derive(Debug, Clone)]
pub enum LayerAssignment {
    /// All even-numbered layers (0, 2, 4, ...)
    Even,
    /// All odd-numbered layers (1, 3, 5, ...)
    Odd,
    /// Specific range (inclusive)
    Range(u32, u32),
    /// Explicit list
    List(Vec<u32>),
}

impl LayerAssignment {
    /// Parse from config string: "odd", "even", "0-31", "1,3,5,7"
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "odd" => Self::Odd,
            "even" => Self::Even,
            s if s.contains('-') => {
                let parts: Vec<&str> = s.split('-').collect();
                if parts.len() == 2 {
                    let start = parts[0].trim().parse().unwrap_or(0);
                    let end = parts[1].trim().parse().unwrap_or(0);
                    Self::Range(start, end)
                } else {
                    Self::List(vec![])
                }
            }
            s if s.contains(',') => {
                let layers: Vec<u32> = s.split(',')
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                Self::List(layers)
            }
            _ => Self::List(vec![]),
        }
    }

    /// Check if this assignment includes a specific layer
    pub fn includes(&self, layer: u32) -> bool {
        match self {
            Self::Even => layer % 2 == 0,
            Self::Odd => layer % 2 != 0,
            Self::Range(start, end) => layer >= *start && layer <= *end,
            Self::List(layers) => layers.contains(&layer),
        }
    }

    /// Get all layer indices for a model with total_layers
    pub fn resolve(&self, total_layers: u32) -> Vec<u32> {
        (0..total_layers).filter(|l| self.includes(*l)).collect()
    }
}

/// Configuration for a single CopperNode (one remote machine)
#[derive(Debug, Clone)]
pub struct CopperNodeConfig {
    /// Human-readable name (e.g., "dl360", "old-laptop", "rpi-cluster")
    pub name: String,
    /// Network endpoint (host:port)
    pub endpoint: String,
    /// Co-processing mode
    pub mode: CopperMode,
    /// Which layers this node handles
    pub layers: LayerAssignment,
    /// .aint identity for cluster auth
    pub aint_domain: Option<String>,
    /// Number of CPU cores available on this node
    pub cores: Option<usize>,
    /// Whether this node has GPU
    pub has_gpu: bool,
}

/// Full CopperEngine configuration
#[derive(Debug, Clone)]
pub struct CopperConfig {
    /// All registered co-processor nodes
    pub nodes: Vec<CopperNodeConfig>,
    /// Total layers in the model
    pub total_layers: u32,
    /// Hidden dimension (for result sizing)
    pub hidden_dim: usize,
    /// Prefetch depth: how many layers ahead to pre-compute
    pub prefetch_depth: u32,
    /// Timeout for node responses
    pub timeout: Duration,
}

impl CopperConfig {
    /// Create empty config (local-only inference)
    pub fn local_only(total_layers: u32, hidden_dim: usize) -> Self {
        Self {
            nodes: vec![],
            total_layers,
            hidden_dim,
            prefetch_depth: 2,
            timeout: Duration::from_millis(100),
        }
    }

    /// Add a co-processor node
    pub fn add_node(&mut self, node: CopperNodeConfig) -> &mut Self {
        self.nodes.push(node);
        self
    }

    /// Quick setup: single remote node in Blitter mode for odd layers
    pub fn with_blitter(total_layers: u32, hidden_dim: usize, name: &str, endpoint: &str) -> Self {
        Self {
            nodes: vec![CopperNodeConfig {
                name: name.to_string(),
                endpoint: endpoint.to_string(),
                mode: CopperMode::Blitter,
                layers: LayerAssignment::Odd,
                aint_domain: None,
                cores: None,
                has_gpu: false,
            }],
            total_layers,
            hidden_dim,
            prefetch_depth: 2,
            timeout: Duration::from_millis(100),
        }
    }

    /// Validate that all layers are covered
    pub fn coverage_report(&self) -> CoverageReport {
        let mut covered = vec![false; self.total_layers as usize];
        let mut node_map: HashMap<u32, Vec<String>> = HashMap::new();

        for node in &self.nodes {
            for layer in node.layers.resolve(self.total_layers) {
                if (layer as usize) < covered.len() {
                    covered[layer as usize] = true;
                    node_map.entry(layer)
                        .or_default()
                        .push(node.name.clone());
                }
            }
        }

        let remote_layers: Vec<u32> = covered.iter().enumerate()
            .filter(|(_, &c)| c)
            .map(|(i, _)| i as u32)
            .collect();
        let local_layers: Vec<u32> = covered.iter().enumerate()
            .filter(|(_, &c)| !c)
            .map(|(i, _)| i as u32)
            .collect();

        CoverageReport {
            total_layers: self.total_layers,
            remote_layers,
            local_layers,
            node_map,
        }
    }
}

/// Report on which layers are handled where
#[derive(Debug)]
pub struct CoverageReport {
    pub total_layers: u32,
    pub remote_layers: Vec<u32>,
    pub local_layers: Vec<u32>,
    pub node_map: HashMap<u32, Vec<String>>,
}

impl std::fmt::Display for CoverageReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔═══════════════════════════════════════╗")?;
        writeln!(f, "║  CopperEngine Coverage Report         ║")?;
        writeln!(f, "╠═══════════════════════════════════════╣")?;
        writeln!(f, "║  Total layers:  {:>4}                  ║", self.total_layers)?;
        writeln!(f, "║  Local layers:  {:>4} (main CPU)       ║", self.local_layers.len())?;
        writeln!(f, "║  Remote layers: {:>4} (co-processors)  ║", self.remote_layers.len())?;
        writeln!(f, "╠═══════════════════════════════════════╣")?;

        // Group by node
        let mut node_layers: HashMap<String, Vec<u32>> = HashMap::new();
        for (layer, nodes) in &self.node_map {
            for node in nodes {
                node_layers.entry(node.clone()).or_default().push(*layer);
            }
        }
        for (node, mut layers) in node_layers {
            layers.sort();
            let range_str = if layers.len() > 6 {
                format!("{}-{} ({} layers)", layers[0], layers[layers.len()-1], layers.len())
            } else {
                layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(",")
            };
            writeln!(f, "║  {:<12} → {}",  node, range_str)?;
        }

        writeln!(f, "╚═══════════════════════════════════════╝")
    }
}

// ============================================================================
// ENGINE
// ============================================================================

/// Result from a remote layer computation
#[derive(Debug, Clone)]
pub struct LayerResult {
    /// Layer index that was computed
    pub layer: u32,
    /// The hidden state output (hidden_dim floats)
    pub hidden_state: Vec<f32>,
    /// Which node computed this
    pub node_name: String,
    /// How long it took on the remote side
    pub remote_compute_ms: f64,
    /// Network round-trip time
    pub network_ms: f64,
}

/// State of a copper node
#[derive(Debug, Clone)]
pub struct NodeState {
    /// Node config
    pub config: CopperNodeConfig,
    /// Whether the node is reachable
    pub online: bool,
    /// Layers computed so far
    pub layers_computed: u64,
    /// Average compute time per layer
    pub avg_compute_ms: f64,
    /// Pages pre-decrypted (Copper mode)
    pub pages_pre_decrypted: u64,
}

/// CopperEngine — The co-processor orchestrator
///
/// Manages remote nodes, dispatches layer computations,
/// and prefetches results ahead of the inference cursor.
pub struct CopperEngine {
    config: CopperConfig,
    /// Layer → node index mapping (resolved at init)
    layer_routing: HashMap<u32, usize>,
    /// Node states
    node_states: Vec<NodeState>,
    /// Prefetch cache: layer → pre-computed result
    prefetch_cache: HashMap<u32, LayerResult>,
    /// Stats
    total_remote_layers: AtomicU64,
    total_local_layers: AtomicU64,
    total_saved_bytes: AtomicU64,
    active: AtomicBool,
}

impl CopperEngine {
    /// Initialize the CopperEngine and resolve layer routing
    pub fn new(config: CopperConfig) -> Self {
        let mut layer_routing = HashMap::new();
        let mut node_states = Vec::new();

        for (idx, node_config) in config.nodes.iter().enumerate() {
            // Map each layer to its node
            for layer in node_config.layers.resolve(config.total_layers) {
                layer_routing.insert(layer, idx);
            }

            node_states.push(NodeState {
                config: node_config.clone(),
                online: false, // Will be checked on first use
                layers_computed: 0,
                avg_compute_ms: 0.0,
                pages_pre_decrypted: 0,
            });
        }

        Self {
            config,
            layer_routing,
            node_states,
            prefetch_cache: HashMap::new(),
            total_remote_layers: AtomicU64::new(0),
            total_local_layers: AtomicU64::new(0),
            total_saved_bytes: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Check if a layer should be computed remotely
    pub fn is_remote(&self, layer: u32) -> bool {
        self.layer_routing.contains_key(&layer)
    }

    /// Check if a layer should be computed locally
    pub fn is_local(&self, layer: u32) -> bool {
        !self.is_remote(layer)
    }

    /// Get the node responsible for a layer
    pub fn node_for_layer(&self, layer: u32) -> Option<&CopperNodeConfig> {
        self.layer_routing.get(&layer)
            .and_then(|&idx| self.config.nodes.get(idx))
    }

    /// Get the mode for a specific layer
    pub fn mode_for_layer(&self, layer: u32) -> Option<&CopperMode> {
        self.node_for_layer(layer).map(|n| &n.mode)
    }

    /// Dispatch a layer computation to a remote node
    ///
    /// For Blitter/Full mode: sends input hidden state, receives output hidden state.
    /// For Copper mode: sends pre-decrypt request, returns None (data cached on node).
    ///
    /// In production this goes over the network via ClusterMux.
    /// For now, simulates the dispatch and returns timing estimates.
    pub fn dispatch_layer(
        &mut self,
        layer: u32,
        input: &[f32],
    ) -> Option<LayerResult> {
        let node_idx = *self.layer_routing.get(&layer)?;
        let node = &self.config.nodes[node_idx];

        // Check prefetch cache first
        if let Some(cached) = self.prefetch_cache.remove(&layer) {
            self.total_remote_layers.fetch_add(1, Ordering::Relaxed);
            // Saved: full layer weights (hidden_dim * hidden_dim * 4 bytes for f32)
            let saved = (self.config.hidden_dim * self.config.hidden_dim * 4) as u64;
            self.total_saved_bytes.fetch_add(saved, Ordering::Relaxed);
            return Some(cached);
        }

        let t0 = Instant::now();

        // Simulate remote computation based on mode
        let result = match &node.mode {
            CopperMode::Copper => {
                // Copper mode: pre-decrypt only, no compute result
                // The decrypted page will be available in local RAM-RAID cache
                self.node_states[node_idx].pages_pre_decrypted += 1;
                None
            }
            CopperMode::Blitter | CopperMode::Full => {
                // Blitter/Full: remote does the matmul
                // In production: send input via ClusterMux, receive result
                // Here: simulate with pass-through (real impl connects via network)
                let hidden_state = vec![0.0f32; self.config.hidden_dim];

                self.node_states[node_idx].layers_computed += 1;
                self.total_remote_layers.fetch_add(1, Ordering::Relaxed);

                let saved = (self.config.hidden_dim * self.config.hidden_dim * 4) as u64;
                self.total_saved_bytes.fetch_add(saved, Ordering::Relaxed);

                Some(LayerResult {
                    layer,
                    hidden_state,
                    node_name: node.name.clone(),
                    remote_compute_ms: 0.0, // Will be filled by actual network call
                    network_ms: t0.elapsed().as_secs_f64() * 1000.0,
                })
            }
        };

        result
    }

    /// Trigger prefetch for upcoming layers
    ///
    /// Call this when processing layer N to prefetch layers N+1..N+depth.
    /// Like the Amiga Copper running ahead of the raster beam.
    pub fn prefetch_ahead(&mut self, current_layer: u32) {
        let depth = self.config.prefetch_depth;
        for ahead in 1..=depth {
            let target = current_layer + ahead;
            if target >= self.config.total_layers {
                break;
            }
            if self.is_remote(target) && !self.prefetch_cache.contains_key(&target) {
                // In production: async dispatch to remote node
                // The result will be in prefetch_cache when we need it
                let node_idx = self.layer_routing[&target];
                let node = &self.config.nodes[node_idx];

                if node.mode != CopperMode::Copper {
                    // Pre-compute: simulate putting result in cache
                    self.prefetch_cache.insert(target, LayerResult {
                        layer: target,
                        hidden_state: vec![0.0f32; self.config.hidden_dim],
                        node_name: node.name.clone(),
                        remote_compute_ms: 0.0,
                        network_ms: 0.0,
                    });
                }
            }
        }
    }

    /// Run a full inference pass through all layers with Copper dispatch
    ///
    /// Takes a closure for local layer computation.
    /// Remote layers are dispatched to co-processor nodes.
    pub fn run_inference<F>(
        &mut self,
        initial_hidden: Vec<f32>,
        mut local_compute: F,
    ) -> Vec<f32>
    where
        F: FnMut(u32, &[f32]) -> Vec<f32>,
    {
        let mut hidden = initial_hidden;

        for layer in 0..self.config.total_layers {
            // Prefetch upcoming layers (Copper running ahead)
            self.prefetch_ahead(layer);

            if self.is_remote(layer) {
                // Dispatch to co-processor
                if let Some(result) = self.dispatch_layer(layer, &hidden) {
                    hidden = result.hidden_state;
                } else {
                    // Copper mode or node offline: fall back to local
                    self.total_local_layers.fetch_add(1, Ordering::Relaxed);
                    hidden = local_compute(layer, &hidden);
                }
            } else {
                // Local computation
                self.total_local_layers.fetch_add(1, Ordering::Relaxed);
                hidden = local_compute(layer, &hidden);
            }
        }

        hidden
    }

    /// Get engine statistics
    pub fn stats(&self) -> CopperStats {
        let remote = self.total_remote_layers.load(Ordering::Relaxed);
        let local = self.total_local_layers.load(Ordering::Relaxed);
        let saved = self.total_saved_bytes.load(Ordering::Relaxed);

        CopperStats {
            total_nodes: self.config.nodes.len(),
            remote_layers: remote,
            local_layers: local,
            offload_ratio: if remote + local > 0 {
                remote as f64 / (remote + local) as f64
            } else { 0.0 },
            bandwidth_saved_bytes: saved,
            node_states: self.node_states.clone(),
        }
    }

    /// Print a status report
    pub fn report(&self) -> String {
        let stats = self.stats();
        let coverage = self.config.coverage_report();

        let mut s = format!("{}", coverage);
        s.push_str(&format!("\n  Remote layers computed: {}\n", stats.remote_layers));
        s.push_str(&format!("  Local layers computed:  {}\n", stats.local_layers));
        s.push_str(&format!("  Offload ratio:         {:.0}%\n", stats.offload_ratio * 100.0));
        s.push_str(&format!("  Bandwidth saved:       {:.1} MB\n",
            stats.bandwidth_saved_bytes as f64 / (1024.0 * 1024.0)));

        for node in &stats.node_states {
            s.push_str(&format!("\n  {} ({:?}):\n", node.config.name, node.config.mode));
            s.push_str(&format!("    Layers computed:     {}\n", node.layers_computed));
            s.push_str(&format!("    Pages pre-decrypted: {}\n", node.pages_pre_decrypted));
        }

        s
    }
}

/// Engine statistics
#[derive(Debug, Clone)]
pub struct CopperStats {
    pub total_nodes: usize,
    pub remote_layers: u64,
    pub local_layers: u64,
    pub offload_ratio: f64,
    pub bandwidth_saved_bytes: u64,
    pub node_states: Vec<NodeState>,
}

// ============================================================================
// TOML CONFIG PARSER
// ============================================================================

impl CopperConfig {
    /// Parse from a TOML-style config string
    ///
    /// ```toml
    /// [copper]
    /// total_layers = 80
    /// hidden_dim = 4096
    /// prefetch_depth = 3
    ///
    /// [[copper.nodes]]
    /// name = "dl360"
    /// endpoint = "10.0.100.1:4431"
    /// mode = "blitter"
    /// layers = "odd"
    /// cores = 24
    ///
    /// [[copper.nodes]]
    /// name = "old-laptop"
    /// endpoint = "192.168.1.50:4431"
    /// mode = "copper"
    /// layers = "40-60"
    /// ```
    pub fn from_node_specs(
        total_layers: u32,
        hidden_dim: usize,
        specs: &[(&str, &str, &str, &str)], // (name, endpoint, mode, layers)
    ) -> Self {
        let mut config = Self::local_only(total_layers, hidden_dim);

        for (name, endpoint, mode, layers) in specs {
            let copper_mode = match *mode {
                "copper" => CopperMode::Copper,
                "blitter" => CopperMode::Blitter,
                "full" => CopperMode::Full,
                _ => CopperMode::Copper,
            };

            config.add_node(CopperNodeConfig {
                name: name.to_string(),
                endpoint: endpoint.to_string(),
                mode: copper_mode,
                layers: LayerAssignment::parse(layers),
                aint_domain: Some(format!("{}.aint", name)),
                cores: None,
                has_gpu: false,
            });
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_assignment_parse() {
        assert!(LayerAssignment::parse("odd").includes(1));
        assert!(!LayerAssignment::parse("odd").includes(0));
        assert!(LayerAssignment::parse("even").includes(0));
        assert!(!LayerAssignment::parse("even").includes(1));
        assert!(LayerAssignment::parse("10-20").includes(15));
        assert!(!LayerAssignment::parse("10-20").includes(21));
        assert!(LayerAssignment::parse("1,3,5").includes(3));
        assert!(!LayerAssignment::parse("1,3,5").includes(4));
    }

    #[test]
    fn test_layer_assignment_resolve() {
        let odd = LayerAssignment::Odd;
        let layers = odd.resolve(8);
        assert_eq!(layers, vec![1, 3, 5, 7]);

        let range = LayerAssignment::Range(2, 5);
        let layers = range.resolve(8);
        assert_eq!(layers, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_coverage_report() {
        let config = CopperConfig::from_node_specs(
            32, 4096,
            &[
                ("dl360", "10.0.100.1:4431", "blitter", "odd"),
                ("rpi", "10.0.100.10:4431", "copper", "0-7"),
            ],
        );

        let report = config.coverage_report();
        assert_eq!(report.remote_layers.len(), 16 + 4); // 16 odd + 4 even in 0-7 not already odd
        // Local = layers not covered by any node
        assert!(report.local_layers.len() > 0);
        let _ = format!("{}", report); // Shouldn't panic
    }

    #[test]
    fn test_copper_engine_routing() {
        let config = CopperConfig::with_blitter(80, 4096, "dl360", "10.0.100.1:4431");
        let engine = CopperEngine::new(config);

        // Odd layers → remote (Blitter)
        assert!(engine.is_remote(1));
        assert!(engine.is_remote(79));
        // Even layers → local
        assert!(engine.is_local(0));
        assert!(engine.is_local(78));

        assert_eq!(engine.node_for_layer(1).unwrap().name, "dl360");
        assert!(engine.node_for_layer(0).is_none());
    }

    #[test]
    fn test_copper_engine_dispatch() {
        let config = CopperConfig::with_blitter(8, 128, "dl360", "10.0.100.1:4431");
        let mut engine = CopperEngine::new(config);

        let input = vec![1.0f32; 128];

        // Dispatch odd layer → should return result
        let result = engine.dispatch_layer(1, &input);
        assert!(result.is_some());
        assert_eq!(result.unwrap().hidden_state.len(), 128);

        // Dispatch even layer → no node assigned
        let result = engine.dispatch_layer(0, &input);
        assert!(result.is_none());

        let stats = engine.stats();
        assert_eq!(stats.remote_layers, 1);
    }

    #[test]
    fn test_copper_engine_full_inference() {
        let config = CopperConfig::with_blitter(8, 64, "dl360", "10.0.100.1:4431");
        let mut engine = CopperEngine::new(config);

        let initial = vec![1.0f32; 64];

        let output = engine.run_inference(initial, |_layer, input| {
            // Simple local compute: just pass through
            input.to_vec()
        });

        assert_eq!(output.len(), 64);

        let stats = engine.stats();
        // 4 odd layers remote, 4 even layers local
        assert_eq!(stats.remote_layers, 4);
        assert_eq!(stats.local_layers, 4);
        assert!((stats.offload_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_copper_engine_prefetch() {
        let config = CopperConfig::with_blitter(8, 64, "dl360", "10.0.100.1:4431");
        let mut engine = CopperEngine::new(config);

        // Prefetch from layer 0: should pre-compute layers 1 and 2
        engine.prefetch_ahead(0);

        // Layer 1 (odd) should be in prefetch cache
        assert!(engine.prefetch_cache.contains_key(&1));
        // Layer 2 (even) is local, no prefetch needed
        assert!(!engine.prefetch_cache.contains_key(&2));
    }

    #[test]
    fn test_multi_node_config() {
        // Non-overlapping assignment: each node gets unique layers
        let config = CopperConfig::from_node_specs(
            80, 4096,
            &[
                ("dl360", "10.0.100.1:4431", "blitter", "16-39"),
                ("old-laptop", "192.168.1.50:4431", "copper", "0-15"),
                ("rpi", "10.0.100.10:4431", "full", "60-79"),
            ],
        );

        let engine = CopperEngine::new(config);

        // Layer 20: dl360 (Blitter, 16-39)
        assert_eq!(engine.node_for_layer(20).unwrap().name, "dl360");
        assert_eq!(engine.mode_for_layer(20), Some(&CopperMode::Blitter));

        // Layer 4: old-laptop (Copper, 0-15)
        assert_eq!(engine.node_for_layer(4).unwrap().name, "old-laptop");
        assert_eq!(engine.mode_for_layer(4), Some(&CopperMode::Copper));

        // Layer 65: rpi (Full, 60-79)
        assert_eq!(engine.node_for_layer(65).unwrap().name, "rpi");
        assert_eq!(engine.mode_for_layer(65), Some(&CopperMode::Full));

        // Layer 45: not assigned to any node → local
        assert!(engine.is_local(45));
    }

    #[test]
    fn test_bandwidth_savings() {
        let hidden_dim = 4096;
        let config = CopperConfig::with_blitter(8, hidden_dim, "dl360", "10.0.100.1:4431");
        let mut engine = CopperEngine::new(config);

        let initial = vec![1.0f32; hidden_dim];
        let _output = engine.run_inference(initial, |_l, input| input.to_vec());

        let stats = engine.stats();
        // 4 remote layers, each saves hidden_dim^2 * 4 bytes
        let expected_saved = 4 * (hidden_dim * hidden_dim * 4) as u64;
        assert_eq!(stats.bandwidth_saved_bytes, expected_saved);

        // That's 4 * 4096 * 4096 * 4 = 256MB saved for just 8 layers!
        let saved_mb = stats.bandwidth_saved_bytes as f64 / (1024.0 * 1024.0);
        assert!(saved_mb > 200.0, "Should save >200MB, saved {:.0}MB", saved_mb);
    }
}
