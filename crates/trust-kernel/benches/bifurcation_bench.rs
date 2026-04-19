use std::time::Instant;

// Inline the types we need (bench is external crate)
#[path = "../src/bifurcation.rs"]
mod bifurcation;

use bifurcation::*;

fn make_test_claim(clearance: ClearanceLevel) -> JisClaim {
    JisClaim {
        identity: "root_idd.aint".to_string(),
        ed25519_pub: "a".repeat(64), // 32 bytes hex = 64 chars
        clearance,
        role: "operator".to_string(),
        dept: "security".to_string(),
        claimed_at: "2026-04-14T15:00:00Z".to_string(),
        signature: "test_sig".to_string(),
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  AIRLOCK BIFURCATIE BENCHMARK — Encrypt-by-Default");
    println!("  \"Bewijs wie je bent om te ontsleutelen\"");
    println!("═══════════════════════════════════════════════════════════\n");

    // ── Part 1: Seal (encrypt-on-write) ──
    {
        println!("── Part 1: Seal (encrypt-on-write) ──");
        let mut engine = AirlockBifurcation::new();
        let data = vec![0x42u8; 4096]; // 4KB block

        let iterations = 10_000;
        let t0 = Instant::now();

        for i in 0..iterations {
            let _ = engine.seal(&data, i, ClearanceLevel::Confidential, "root_idd.aint");
        }

        let total_us = t0.elapsed().as_micros();
        let per_op = total_us / iterations as u128;
        let throughput_mbs = (iterations as f64 * 4096.0) / (total_us as f64 / 1_000_000.0) / (1024.0 * 1024.0);

        println!("  {} seals × 4KB", iterations);
        println!("  Total:      {} µs", total_us);
        println!("  Per seal:   {} µs", per_op);
        println!("  Throughput: {:.1} MB/s", throughput_mbs);
        println!("  Stats:      sealed={}\n", engine.stats().blocks_sealed);
    }

    // ── Part 2: Open (decrypt-on-read) with valid claim ──
    {
        println!("── Part 2: Open (decrypt-on-read) — valid claim ──");
        let mut engine = AirlockBifurcation::new();
        let data = vec![0xAB; 4096];

        // Seal blocks first
        let mut blocks = Vec::new();
        for i in 0..1000 {
            if let BifurcationResult::Sealed { block, .. } =
                engine.seal(&data, i, ClearanceLevel::Confidential, "root_idd.aint")
            {
                blocks.push(block);
            }
        }

        let claim = make_test_claim(ClearanceLevel::Secret); // Secret > Confidential = OK

        let t0 = Instant::now();
        let mut success = 0u64;

        for block in &blocks {
            if let BifurcationResult::Opened { .. } = engine.open(block, &claim) {
                success += 1;
            }
        }

        let total_us = t0.elapsed().as_micros();
        let per_op = total_us / blocks.len() as u128;

        println!("  {} opens × 4KB", blocks.len());
        println!("  Total:    {} µs", total_us);
        println!("  Per open: {} µs", per_op);
        println!("  Success:  {}/{}", success, blocks.len());
        println!("  Stats:    opened={}\n", engine.stats().blocks_opened);
    }

    // ── Part 3: Access Denied (clearance te laag) ──
    {
        println!("── Part 3: Access Denied — clearance te laag ──");
        let mut engine = AirlockBifurcation::new();
        let data = vec![0xCC; 4096];

        // Seal met SECRET clearance
        let block = if let BifurcationResult::Sealed { block, .. } =
            engine.seal(&data, 0, ClearanceLevel::Secret, "system")
        {
            block
        } else {
            panic!("Seal failed");
        };

        // Probeer te openen met RESTRICTED (te laag)
        let low_claim = make_test_claim(ClearanceLevel::Restricted);
        let result = engine.open(&block, &low_claim);

        match &result {
            BifurcationResult::AccessDenied { required, presented, identity } => {
                println!("  ✓ Correct geweigerd!");
                println!("    Required:  {}", required.as_str());
                println!("    Presented: {}", presented.as_str());
                println!("    Identity:  {}", identity);
            }
            other => println!("  ✗ Unexpected: {:?}", other),
        }

        // Probeer met TOPSECRET (hoog genoeg)
        let high_claim = make_test_claim(ClearanceLevel::TopSecret);
        let result = engine.open(&block, &high_claim);

        match &result {
            BifurcationResult::Opened { opened_by, .. } => {
                println!("  ✓ Correct geopend met TopSecret door {}", opened_by);
            }
            other => println!("  ✗ Unexpected: {:?}", other),
        }
        println!("  Stats: denied={}\n", engine.stats().access_denied);
    }

    // ── Part 4: Roundtrip integriteit ──
    {
        println!("── Part 4: Roundtrip integriteit ──");
        let mut engine = AirlockBifurcation::new();

        let test_data: Vec<Vec<u8>> = vec![
            b"Hello, TIBET! Dit is geheime data.".to_vec(),
            vec![0u8; 65536],         // 64KB zeros
            (0..255).collect(),        // 0-254 byte pattern
            vec![0xFF; 1024 * 1024],   // 1MB
        ];

        let claim = make_test_claim(ClearanceLevel::TopSecret);

        for (i, data) in test_data.iter().enumerate() {
            let block = if let BifurcationResult::Sealed { block, .. } =
                engine.seal(data, i, ClearanceLevel::Unclassified, "test")
            {
                block
            } else {
                println!("  ✗ Seal failed for test {}", i);
                continue;
            };

            match engine.open(&block, &claim) {
                BifurcationResult::Opened { plaintext, .. } => {
                    if plaintext == *data {
                        println!("  ✓ Test {}: {}B roundtrip OK", i, data.len());
                    } else {
                        println!("  ✗ Test {}: DATA MISMATCH! {} vs {}", i, plaintext.len(), data.len());
                    }
                }
                other => println!("  ✗ Test {}: {:?}", i, other),
            }
        }
        println!();
    }

    // ── Part 5: Live Migration ──
    {
        println!("── Part 5: Live Migration — Zero-Downtime Transfer ──");

        let block_count = 256;
        let block_size = 4096;
        let mut migration = LiveMigration::new(block_count);

        // Simuleer block data
        let blocks: Vec<Vec<u8>> = (0..block_count)
            .map(|i| vec![(i & 0xFF) as u8; block_size])
            .collect();

        // Round 1: Bulk sync (alles is dirty)
        let round1 = migration.sync_round(&blocks, block_size);
        println!("  Round 1 (BulkSync):");
        println!("    Transferred: {} blocks", round1.blocks_transferred);
        println!("    Skipped:     {} blocks", round1.blocks_skipped);
        println!("    Duration:    {} µs", round1.duration_us);
        println!("    Phase:       {:?}", round1.phase);
        println!("    Progress:    {:.1}%", migration.progress_pct());

        // Simuleer writes (10% dirty)
        for i in (0..block_count).step_by(10) {
            migration.mark_dirty(i);
        }

        // Round 2: Delta sync
        let round2 = migration.sync_round(&blocks, block_size);
        println!("\n  Round 2 (DeltaSync):");
        println!("    Transferred: {} blocks", round2.blocks_transferred);
        println!("    Skipped:     {} blocks", round2.blocks_skipped);
        println!("    Duration:    {} µs", round2.duration_us);
        println!("    Phase:       {:?}", round2.phase);

        // Simuleer laatste paar dirty blocks (< 1%)
        migration.mark_dirty(0);
        migration.mark_dirty(1);

        // Round 3: Micro-freeze
        let round3 = migration.sync_round(&blocks, block_size);
        println!("\n  Round 3 (MicroFreeze → Handoff):");
        println!("    Transferred: {} blocks", round3.blocks_transferred);
        println!("    Remaining:   {} dirty", round3.remaining_dirty);
        println!("    Phase:       {:?}", round3.phase);

        // Handoff
        migration.complete_handoff();
        println!("\n  Migration complete!");
        println!("    Total transferred: {} bytes", migration.bytes_transferred);
        println!("    Total skipped:     {} bytes", migration.bytes_skipped);
        println!("    Sync rounds:       {}", migration.sync_rounds);
        println!("    Final phase:       {:?}", migration.phase);
        println!();
    }

    // ── Part 6: Throughput scaling ──
    {
        println!("── Part 6: Throughput scaling (1KB → 1MB) ──");
        let mut engine = AirlockBifurcation::new();
        let claim = make_test_claim(ClearanceLevel::TopSecret);

        for &size in &[1024, 4096, 16384, 65536, 262144, 1048576] {
            let data = vec![0x42u8; size];
            let iterations = if size >= 262144 { 100 } else { 1000 };

            let t0 = Instant::now();
            for i in 0..iterations {
                let block = if let BifurcationResult::Sealed { block, .. } =
                    engine.seal(&data, i, ClearanceLevel::Unclassified, "bench")
                {
                    block
                } else {
                    continue;
                };
                let _ = engine.open(&block, &claim);
            }
            let total_us = t0.elapsed().as_micros();
            let per_roundtrip = total_us / iterations as u128;
            let throughput = (iterations as f64 * size as f64 * 2.0) / (total_us as f64 / 1_000_000.0) / (1024.0 * 1024.0);

            println!("  {:>7}: {:>6} µs/roundtrip  {:>8.1} MB/s",
                format!("{}KB", size / 1024),
                per_roundtrip,
                throughput);
        }
    }

    // ── Part 7: Hardware Feature Detection ──
    {
        println!("\n── Part 7: Hardware Feature Detection ──");

        // RDRAND
        let has_rdrand = rdrand_available();
        println!("  RDRAND:  {}", if has_rdrand { "✅ beschikbaar (nonces via hardware)" } else { "❌ niet beschikbaar (OsRng fallback)" });

        if has_rdrand {
            // Benchmark: RDRAND nonce vs OsRng nonce
            let iterations = 100_000;

            let t0 = Instant::now();
            for _ in 0..iterations {
                let _ = rdrand_nonce();
            }
            let rdrand_us = t0.elapsed().as_micros();

            let t0 = Instant::now();
            for _ in 0..iterations {
                let mut n = [0u8; 12];
                rand::Rng::fill(&mut rand::rngs::OsRng, &mut n);
            }
            let osrng_us = t0.elapsed().as_micros();

            let rdrand_ns = (rdrand_us * 1000) / iterations as u128;
            let osrng_ns = (osrng_us * 1000) / iterations as u128;
            let speedup = osrng_us as f64 / rdrand_us as f64;

            println!("  RDRAND nonce:  {} ns/nonce", rdrand_ns);
            println!("  OsRng nonce:   {} ns/nonce", osrng_ns);
            println!("  Speedup:       {:.1}x", speedup);
        }

        // RDSEED
        let has_rdseed = rdseed_available();
        println!("  RDSEED:  {}", if has_rdseed { "✅ beschikbaar (echte entropy)" } else { "❌ niet beschikbaar" });

        // CAT L3
        let cat_status = cat_l3_status();
        match &cat_status {
            CatL3Status::Active { ways_reserved, ways_total, bitmask, .. } => {
                println!("  CAT L3:  ✅ ACTIEF — {}/{} ways gereserveerd (mask=0x{})", ways_reserved, ways_total, bitmask);
            }
            CatL3Status::Available { ways_total } => {
                println!("  CAT L3:  💤 beschikbaar — {} ways, niet geactiveerd", ways_total);
                println!("           Activeer: cat_l3_activate(&CatL3Config::default(), pid)");
            }
            CatL3Status::NotMounted => {
                println!("  CAT L3:  ⚠️  CPU ondersteunt het, resctrl niet gemount");
                println!("           mount -t resctrl resctrl /sys/fs/resctrl");
            }
            CatL3Status::NotSupported => {
                println!("  CAT L3:  ❌ niet beschikbaar op deze CPU");
            }
            CatL3Status::Failed { reason } => {
                println!("  CAT L3:  ✗ fout: {}", reason);
            }
        }
        println!();
    }

    // ── Part 8: Key Cache — DH Elimination ──
    {
        println!("── Part 8: Key Cache — DH Elimination ──");
        let mut engine = AirlockBifurcation::new();
        let data = vec![0xAB; 4096];
        let claim = make_test_claim(ClearanceLevel::TopSecret);

        // Seal 100 blocks (keys worden gecached bij seal)
        let mut blocks = Vec::new();
        for i in 0..100 {
            if let BifurcationResult::Sealed { block, .. } =
                engine.seal(&data, i, ClearanceLevel::Confidential, "bench")
            {
                blocks.push(block);
            }
        }

        // Reset cache stats
        engine.key_cache.hits = 0;
        engine.key_cache.misses = 0;

        // Open elke block 1x — dit is een cache HIT (seal cachede de key)
        let t0 = Instant::now();
        for block in &blocks {
            let _ = engine.open(block, &claim);
        }
        let cached_us = t0.elapsed().as_micros();
        let cached_per_op = cached_us / blocks.len() as u128;
        let cache_hits_1 = engine.key_cache.hits;

        println!("  After seal → open (cache warm):");
        println!("    100 opens:   {} µs total", cached_us);
        println!("    Per open:    {} µs", cached_per_op);
        println!("    Cache hits:  {}", cache_hits_1);
        println!("    Hit rate:    {:.1}%", engine.key_cache.hit_rate());

        // Open dezelfde blocks NOGMAALS — alles cached
        let t0 = Instant::now();
        for _ in 0..10 {
            for block in &blocks {
                let _ = engine.open(block, &claim);
            }
        }
        let repeat_us = t0.elapsed().as_micros();
        let repeat_per_op = repeat_us / 1000; // 10 × 100 blocks

        println!("\n  Repeated opens (10x same blocks):");
        println!("    1000 opens:  {} µs total", repeat_us);
        println!("    Per open:    {} µs", repeat_per_op);
        println!("    Cache hits:  {}", engine.key_cache.hits);
        println!("    Hit rate:    {:.1}%", engine.key_cache.hit_rate());
        println!("    Cache size:  {} entries", engine.key_cache.len());

        // Vergelijk: nieuwe engine ZONDER cache benefit
        let mut fresh = AirlockBifurcation::new();
        // Seal blocks met fresh engine
        let mut fresh_blocks = Vec::new();
        for i in 0..100 {
            if let BifurcationResult::Sealed { block, .. } =
                fresh.seal(&data, i, ClearanceLevel::Confidential, "bench")
            {
                fresh_blocks.push(block);
            }
        }
        // Flush cache — force cold opens
        fresh.key_cache.flush();

        let t0 = Instant::now();
        for block in &fresh_blocks {
            let _ = fresh.open(block, &claim);
        }
        let cold_us = t0.elapsed().as_micros();
        let cold_per_op = cold_us / fresh_blocks.len() as u128;

        let speedup = cold_per_op as f64 / cached_per_op.max(1) as f64;

        println!("\n  Cold opens (cache flushed):");
        println!("    100 opens:   {} µs total", cold_us);
        println!("    Per open:    {} µs", cold_per_op);

        println!("\n  ⚡ Cache speedup: {:.1}x ({} µs → {} µs per open)", speedup, cold_per_op, cached_per_op);
        println!();
    }

    // ── Part 9: Multi-Core Parallel Seal — Von Braun Mode 🚀 ──
    {
        println!("── Part 9: Multi-Core Parallel Seal — Von Braun Mode ──");
        let system_secret = b"TIBET-BIFURCATION-SYSTEM-KEY-V01";
        let mut secret_bytes = [0u8; 32];
        secret_bytes.copy_from_slice(system_secret);

        let num_threads = rayon::current_num_threads();
        println!("  CPU cores: {} threads beschikbaar\n", num_threads);

        // Test met verschillende block sizes
        for &(block_size, block_count) in &[
            (4096usize, 10_000usize),     // 4KB × 10K = 40MB
            (16384, 4_000),                // 16KB × 4K = 64MB
            (65536, 1_000),                // 64KB × 1K = 64MB
        ] {
            let plaintexts: Vec<Vec<u8>> = (0..block_count)
                .map(|_| vec![0x42u8; block_size])
                .collect();

            // Single-threaded baseline
            let mut engine = AirlockBifurcation::new();
            let t0 = Instant::now();
            for (i, pt) in plaintexts.iter().enumerate() {
                let _ = engine.seal(pt, i, ClearanceLevel::Confidential, "bench");
            }
            let single_us = t0.elapsed().as_micros() as u64;
            let single_mbs = (block_count as f64 * block_size as f64) / (single_us as f64 / 1_000_000.0) / (1024.0 * 1024.0);

            // Multi-core parallel
            let result = parallel_seal(
                &plaintexts,
                &secret_bytes,
                ClearanceLevel::Confidential,
                "bench-parallel",
            );

            let speedup = single_us as f64 / result.total_us.max(1) as f64;

            println!("  {}KB × {} blocks ({} MB):",
                block_size / 1024,
                block_count,
                block_count * block_size / (1024 * 1024));
            println!("    Single-thread: {:>8} µs  ({:>7.1} MB/s)",
                single_us, single_mbs);
            println!("    Multi-core:    {:>8} µs  ({:>7.1} MB/s)  {} threads",
                result.total_us, result.throughput_mbs, result.threads_used);
            println!("    Speedup:       {:.1}x", speedup);
            println!();
        }

        // Parallel open test
        println!("  Parallel Open (4KB × 1000 blocks):");
        let plaintexts: Vec<Vec<u8>> = (0..1000)
            .map(|_| vec![0xAB; 4096])
            .collect();

        let seal_result = parallel_seal(
            &plaintexts,
            &secret_bytes,
            ClearanceLevel::Confidential,
            "bench-parallel",
        );

        let claim = make_test_claim(ClearanceLevel::TopSecret);

        let open_result = parallel_open(
            &seal_result.blocks,
            &claim,
            &secret_bytes,
        );

        println!("    Sealed:     {} blocks in {} µs", seal_result.blocks.len(), seal_result.total_us);
        println!("    Opened:     {} blocks in {} µs", open_result.plaintexts.len(), open_result.total_us);
        println!("    Per open:   {} µs", open_result.per_block_us);
        println!("    Throughput: {:.1} MB/s", open_result.throughput_mbs);
        println!("    Denied:     {}", open_result.denied);

        // Verify integrity
        let all_ok = open_result.plaintexts.iter().all(|pt| pt.len() == 4096 && pt[0] == 0xAB);
        println!("    Integrity:  {}", if all_ok { "✓ alle blocks correct" } else { "✗ FOUT!" });
        println!();
    }

    // ── Part 10: Session Key — DH Elimination voor Seal ──
    {
        println!("── Part 10: Session Key — DH Elimination voor Seal ──");
        let claim = make_test_claim(ClearanceLevel::TopSecret);
        let data = vec![0x42u8; 4096];
        let iterations = 10_000;

        // Regular seal (volledige DH per block)
        let mut engine_regular = AirlockBifurcation::new();
        let t0 = Instant::now();
        for i in 0..iterations {
            let _ = engine_regular.seal(&data, i, ClearanceLevel::Confidential, "bench");
        }
        let regular_us = t0.elapsed().as_micros();
        let regular_per = regular_us / iterations as u128;
        let regular_mbs = (iterations as f64 * 4096.0) / (regular_us as f64 / 1_000_000.0) / (1024.0 * 1024.0);

        // Session seal (één DH, daarna alleen HKDF+AES)
        let mut engine_session = AirlockBifurcation::new();
        let t0 = Instant::now();
        for i in 0..iterations {
            let _ = engine_session.seal_session(&data, i, ClearanceLevel::Confidential, "bench");
        }
        let session_us = t0.elapsed().as_micros();
        let session_per = session_us / iterations as u128;
        let session_mbs = (iterations as f64 * 4096.0) / (session_us as f64 / 1_000_000.0) / (1024.0 * 1024.0);

        let speedup = regular_per as f64 / session_per.max(1) as f64;

        println!("  {} seals × 4KB:", iterations);
        println!("    Regular seal:  {:>6} µs/seal  ({:>7.1} MB/s)  [full DH per block]",
            regular_per, regular_mbs);
        println!("    Session seal:  {:>6} µs/seal  ({:>7.1} MB/s)  [cached DH]",
            session_per, session_mbs);
        println!("    ⚡ Speedup:     {:.1}x", speedup);
        println!("    Session blocks: {}", engine_session.session_blocks());

        // Roundtrip verify: session-sealed blocks moeten openbaar zijn
        println!("\n  Roundtrip verify (session sealed → open):");
        let mut engine = AirlockBifurcation::new();
        let test_sizes = [64, 4096, 65536, 1048576];
        for &size in &test_sizes {
            let test_data = vec![0xABu8; size];
            let block = if let BifurcationResult::Sealed { block, .. } =
                engine.seal_session(&test_data, size, ClearanceLevel::Confidential, "test")
            {
                block
            } else {
                println!("    ✗ {}B seal failed", size);
                continue;
            };

            match engine.open(&block, &claim) {
                BifurcationResult::Opened { plaintext, .. } => {
                    if plaintext == test_data {
                        println!("    ✓ {}B roundtrip OK", size);
                    } else {
                        println!("    ✗ {}B DATA MISMATCH", size);
                    }
                }
                other => println!("    ✗ {}B: {:?}", size, other),
            }
        }

        // Throughput scaling met session keys
        println!("\n  Session throughput scaling:");
        let mut engine = AirlockBifurcation::new();
        for &size in &[1024, 4096, 16384, 65536, 262144, 1048576] {
            let data = vec![0x42u8; size];
            let iters = if size >= 262144 { 100 } else { 1000 };

            let t0 = Instant::now();
            for i in 0..iters {
                let _ = engine.seal_session(&data, i, ClearanceLevel::Unclassified, "bench");
            }
            let total_us = t0.elapsed().as_micros();
            let per_seal = total_us / iters as u128;
            let throughput = (iters as f64 * size as f64) / (total_us as f64 / 1_000_000.0) / (1024.0 * 1024.0);

            println!("    {:>7}: {:>6} µs/seal  {:>8.1} MB/s",
                format!("{}KB", size / 1024),
                per_seal,
                throughput);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  Airlock Bifurcatie v2: encrypt-by-default WERKT.");
    println!("  Geen JIS = dood materiaal. Identity IS the key.");
    println!("  Von Braun Mode: multi-core + session keys.");
    println!("═══════════════════════════════════════════════════════════");
}
