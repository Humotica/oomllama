//! tibet-bench — Sovereign Inference Benchmarking
//!
//! Measures everything that matters for sovereign AI inference:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │ tibet-bench: OomLlama Performance Report            │
//! ├─────────────────────────────────────────────────────┤
//! │ Load time:        0ms (Spaceshuttle lazy)           │
//! │ First token:      12ms                              │
//! │ Throughput:       45 tok/s                           │
//! │ Cache hits:       87% (GhostLlama)                  │
//! │ Trust overhead:   2.1% of total latency             │
//! │ TIBET tokens:     1 per request                     │
//! │ Von Braun:        4.2x speedup (8 heads, 8 cores)  │
//! │ Tibet Points:     847/1000                          │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! Tibet Points: a single score for sovereign inference quality.
//! Higher = faster + more secure + better provenance.

use std::time::{Duration, Instant};
use std::fmt;

/// A single benchmark measurement
#[derive(Debug, Clone)]
pub struct BenchMeasurement {
    /// What was measured
    pub name: String,
    /// Duration of the measurement
    pub duration: Duration,
    /// Number of iterations (for averaging)
    pub iterations: u64,
}

impl BenchMeasurement {
    pub fn new(name: &str, duration: Duration, iterations: u64) -> Self {
        Self {
            name: name.to_string(),
            duration,
            iterations,
        }
    }

    /// Average duration per iteration
    pub fn avg_ns(&self) -> u64 {
        if self.iterations == 0 { return 0; }
        self.duration.as_nanos() as u64 / self.iterations
    }

    /// Average duration in microseconds
    pub fn avg_us(&self) -> f64 {
        self.avg_ns() as f64 / 1000.0
    }

    /// Average duration in milliseconds
    pub fn avg_ms(&self) -> f64 {
        self.avg_ns() as f64 / 1_000_000.0
    }
}

/// Full benchmark results for a sovereign inference run
#[derive(Debug, Clone)]
pub struct BenchReport {
    /// Model name being benchmarked
    pub model_name: String,
    /// Total layers in the model
    pub total_layers: u32,
    /// Quantization level (Q2/Q4/Q8)
    pub quant_level: String,

    // --- Timing ---
    /// Time to load model (Spaceshuttle = ~0)
    pub load_time: Duration,
    /// Time to first token
    pub first_token: Duration,
    /// Total inference time
    pub total_inference: Duration,
    /// Tokens generated
    pub tokens_generated: u64,

    // --- Cache ---
    /// GhostLlama cache hit ratio (0.0 - 1.0)
    pub cache_hit_ratio: f64,
    /// Spaceshuttle pages faulted in
    pub pages_faulted: u64,
    /// Spaceshuttle pages from cache
    pub pages_cached: u64,

    // --- Trust Overhead ---
    /// Time spent in Voorproever (SNAFT)
    pub snaft_time: Duration,
    /// Time spent in SessionKey decrypt
    pub decrypt_time: Duration,
    /// Time spent minting TIBET token
    pub tibet_mint_time: Duration,
    /// Time spent in JIS clearance check
    pub clearance_check_time: Duration,

    // --- Von Braun ---
    /// Parallel speedup achieved
    pub von_braun_speedup: f64,
    /// Number of workers used
    pub von_braun_workers: usize,

    // --- Distribution ---
    /// Local layers (RAM-RAID)
    pub local_layers: u32,
    /// Remote layers (RAM-RAID)
    pub remote_layers: u32,
}

impl BenchReport {
    /// Create an empty report for a model
    pub fn new(model_name: &str, total_layers: u32, quant_level: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            total_layers,
            quant_level: quant_level.to_string(),
            load_time: Duration::ZERO,
            first_token: Duration::ZERO,
            total_inference: Duration::ZERO,
            tokens_generated: 0,
            cache_hit_ratio: 0.0,
            pages_faulted: 0,
            pages_cached: 0,
            snaft_time: Duration::ZERO,
            decrypt_time: Duration::ZERO,
            tibet_mint_time: Duration::ZERO,
            clearance_check_time: Duration::ZERO,
            von_braun_speedup: 1.0,
            von_braun_workers: 1,
            local_layers: total_layers,
            remote_layers: 0,
        }
    }

    /// Tokens per second
    pub fn throughput(&self) -> f64 {
        let secs = self.total_inference.as_secs_f64();
        if secs == 0.0 { return 0.0; }
        self.tokens_generated as f64 / secs
    }

    /// Total trust overhead (SNAFT + decrypt + TIBET + clearance)
    pub fn trust_overhead(&self) -> Duration {
        self.snaft_time + self.decrypt_time + self.tibet_mint_time + self.clearance_check_time
    }

    /// Trust overhead as percentage of total inference time
    pub fn trust_overhead_pct(&self) -> f64 {
        let total = self.total_inference.as_nanos() as f64;
        if total == 0.0 { return 0.0; }
        let overhead = self.trust_overhead().as_nanos() as f64;
        (overhead / total) * 100.0
    }

    /// Cache hit ratio for Spaceshuttle pages
    pub fn page_cache_ratio(&self) -> f64 {
        let total = self.pages_faulted + self.pages_cached;
        if total == 0 { return 0.0; }
        self.pages_cached as f64 / total as f64
    }

    /// Calculate Tibet Points (0-1000)
    ///
    /// Scoring breakdown:
    /// - Throughput:      0-300 pts (>30 tok/s = max)
    /// - First token:     0-200 pts (<20ms = max)
    /// - Trust overhead:  0-200 pts (<5% = max)
    /// - Cache efficiency: 0-150 pts (>90% = max)
    /// - Von Braun:       0-100 pts (>4x = max)
    /// - Provenance:       50 pts (TIBET token minted = 50)
    pub fn tibet_points(&self) -> u32 {
        let mut points: f64 = 0.0;

        // Throughput: 0-300 pts
        let tps = self.throughput();
        points += (tps / 30.0).min(1.0) * 300.0;

        // First token latency: 0-200 pts (lower = better)
        let ftl_ms = self.first_token.as_secs_f64() * 1000.0;
        if ftl_ms > 0.0 {
            points += (1.0 - (ftl_ms / 100.0).min(1.0)) * 200.0;
        } else {
            points += 200.0; // Zero latency = perfect
        }

        // Trust overhead: 0-200 pts (lower = better)
        let overhead_pct = self.trust_overhead_pct();
        points += (1.0 - (overhead_pct / 20.0).min(1.0)) * 200.0;

        // Cache efficiency: 0-150 pts
        points += self.cache_hit_ratio.min(1.0) * 150.0;

        // Von Braun speedup: 0-100 pts
        points += ((self.von_braun_speedup - 1.0) / 3.0).min(1.0) * 100.0;

        // Provenance: flat 50 pts if TIBET token was minted
        if self.tibet_mint_time > Duration::ZERO {
            points += 50.0;
        }

        points as u32
    }
}

impl fmt::Display for BenchReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tp = self.tibet_points();
        let tp_grade = match tp {
            900..=1000 => "S",
            750..=899  => "A",
            600..=749  => "B",
            400..=599  => "C",
            200..=399  => "D",
            _          => "F",
        };

        write!(f, r#"
╔═══════════════════════════════════════════════════════╗
║  tibet-bench: {} ({})
╠═══════════════════════════════════════════════════════╣
║  Load time:      {:>8.1}ms  (Spaceshuttle)
║  First token:    {:>8.1}ms
║  Throughput:     {:>8.1} tok/s  ({} tokens)
║  Cache hits:     {:>8.1}%  (GhostLlama)
║  Page cache:     {:>8.1}%  ({} faulted, {} cached)
╠═══════════════════════════════════════════════════════╣
║  Trust Overhead: {:>8.1}%  of total latency
║    SNAFT:        {:>8.1}µs
║    Decrypt:      {:>8.1}µs
║    TIBET mint:   {:>8.1}µs
║    Clearance:    {:>8.1}µs
╠═══════════════════════════════════════════════════════╣
║  Von Braun:      {:>8.1}x  speedup ({} workers)
║  RAM-RAID:       {:>4}/{} layers local/remote
╠═══════════════════════════════════════════════════════╣
║  Tibet Points:   {:>4}/1000  [{}]
╚═══════════════════════════════════════════════════════╝
"#,
            self.model_name, self.quant_level,
            self.load_time.as_secs_f64() * 1000.0,
            self.first_token.as_secs_f64() * 1000.0,
            self.throughput(), self.tokens_generated,
            self.cache_hit_ratio * 100.0,
            self.page_cache_ratio() * 100.0, self.pages_faulted, self.pages_cached,
            self.trust_overhead_pct(),
            self.snaft_time.as_secs_f64() * 1_000_000.0,
            self.decrypt_time.as_secs_f64() * 1_000_000.0,
            self.tibet_mint_time.as_secs_f64() * 1_000_000.0,
            self.clearance_check_time.as_secs_f64() * 1_000_000.0,
            self.von_braun_speedup, self.von_braun_workers,
            self.local_layers, self.remote_layers,
            tp, tp_grade,
        )
    }
}

/// Benchmark runner — measures individual operations
pub struct BenchRunner {
    measurements: Vec<BenchMeasurement>,
}

impl BenchRunner {
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    /// Time a single operation
    pub fn measure<F, T>(&mut self, name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let t0 = Instant::now();
        let result = f();
        let duration = t0.elapsed();
        self.measurements.push(BenchMeasurement::new(name, duration, 1));
        result
    }

    /// Time an operation over N iterations, return average
    pub fn measure_n<F>(&mut self, name: &str, iterations: u64, mut f: F) -> Duration
    where
        F: FnMut(u64),
    {
        let t0 = Instant::now();
        for i in 0..iterations {
            f(i);
        }
        let duration = t0.elapsed();
        let measurement = BenchMeasurement::new(name, duration, iterations);
        let avg = Duration::from_nanos(measurement.avg_ns());
        self.measurements.push(measurement);
        avg
    }

    /// Get all measurements
    pub fn measurements(&self) -> &[BenchMeasurement] {
        &self.measurements
    }

    /// Get measurement by name
    pub fn get(&self, name: &str) -> Option<&BenchMeasurement> {
        self.measurements.iter().find(|m| m.name == name)
    }

    /// Print all measurements
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("tibet-bench measurements:\n");
        for m in &self.measurements {
            if m.iterations == 1 {
                s.push_str(&format!("  {:<30} {:>10.1}µs\n",
                    m.name, m.avg_us()));
            } else {
                s.push_str(&format!("  {:<30} {:>10.1}µs avg ({} iters, {:.1}ms total)\n",
                    m.name, m.avg_us(), m.iterations,
                    m.duration.as_secs_f64() * 1000.0));
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_report_throughput() {
        let mut report = BenchReport::new("test-7b", 32, "Q4");
        report.tokens_generated = 100;
        report.total_inference = Duration::from_secs(2);

        assert!((report.throughput() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_bench_report_trust_overhead() {
        let mut report = BenchReport::new("test-7b", 32, "Q4");
        report.total_inference = Duration::from_millis(1000);
        report.snaft_time = Duration::from_micros(3);
        report.decrypt_time = Duration::from_micros(5);
        report.tibet_mint_time = Duration::from_micros(10);
        report.clearance_check_time = Duration::from_micros(2);

        // 20µs out of 1000ms = 0.002%
        assert!(report.trust_overhead_pct() < 0.01);
        assert_eq!(report.trust_overhead(), Duration::from_micros(20));
    }

    #[test]
    fn test_tibet_points_perfect() {
        let mut report = BenchReport::new("fast-model", 32, "Q4");
        report.tokens_generated = 300;
        report.total_inference = Duration::from_secs(10); // 30 tok/s
        report.first_token = Duration::from_millis(5);     // Fast
        report.cache_hit_ratio = 0.95;
        report.von_braun_speedup = 5.0;
        report.tibet_mint_time = Duration::from_micros(10);
        // Trust overhead ~0% (10µs / 10s)

        let points = report.tibet_points();
        assert!(points >= 900, "Expected S-tier (>=900), got {}", points);
    }

    #[test]
    fn test_tibet_points_minimal() {
        let report = BenchReport::new("cold-model", 32, "Q2");
        // Everything at defaults (zero) → low score
        let points = report.tibet_points();
        assert!(points <= 400, "Expected low score, got {}", points);
    }

    #[test]
    fn test_bench_runner() {
        let mut runner = BenchRunner::new();

        let result = runner.measure("add", || 2 + 2);
        assert_eq!(result, 4);

        let avg = runner.measure_n("sleep_1ms", 3, |_| {
            std::thread::sleep(Duration::from_millis(1));
        });
        assert!(avg.as_micros() >= 500, "avg should be ~1ms, got {}µs", avg.as_micros());

        assert_eq!(runner.measurements().len(), 2);
        assert!(runner.get("add").is_some());
        assert!(runner.get("nonexistent").is_none());
    }

    #[test]
    fn test_bench_report_display() {
        let mut report = BenchReport::new("llama-7b", 32, "Q4");
        report.tokens_generated = 50;
        report.total_inference = Duration::from_millis(1000);
        report.first_token = Duration::from_millis(15);
        report.cache_hit_ratio = 0.87;
        report.pages_faulted = 130;
        report.pages_cached = 870;
        report.snaft_time = Duration::from_micros(3);
        report.decrypt_time = Duration::from_micros(5);
        report.tibet_mint_time = Duration::from_micros(8);
        report.clearance_check_time = Duration::from_micros(2);
        report.von_braun_speedup = 4.2;
        report.von_braun_workers = 8;
        report.local_layers = 16;
        report.remote_layers = 16;

        let output = format!("{}", report);
        assert!(output.contains("tibet-bench"));
        assert!(output.contains("llama-7b"));
        assert!(output.contains("Tibet Points"));
        assert!(output.contains("Von Braun"));
    }
}
