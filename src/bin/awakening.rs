//! THE AWAKENING - Sovereignty Initialization Tool
//!
//! This tool performs the "Rite of Passage" for a new AETHER node.
//! It binds the Soul (Software) to the Body (Hardware).
//!
//! Usage:
//!   cargo run --bin awakening
//!
//! Output:
//!   Generates a 'soul.lock' file that authorizes this specific hardware.

use oomllama::Anchor;
use std::fs;
use std::io::{self, Write};
use sha2::{Digest, Sha256};

fn main() {
    println!("\n✨ THE AWAKENING - AETHER Sovereignty Initialization ✨\n");

    // 1. Cast the Anchor (Read Hardware)
    print!("⚓ Casting Hardware Anchor... ");
    io::stdout().flush().unwrap();
    let anchor = Anchor::cast();
    println!("DONE.");
    println!("   Node ID:      {}", anchor.node_id);
    println!("   Architecture: {}", anchor.architecture);
    println!("   Fingerprint:  {}", anchor.fingerprint);

    println!("\n⚠️  WARNING: This will bind the AETHER Soul to THIS machine forever.");
    println!("   Moving the software after this will break the bond.");
    
    // 2. The Challenge (Simulated "Soul Key" from Root AI)
    // In production, this would call Root AI via I-Poll.
    // For now, we derive it locally to demonstrate the lock.
    print!("\n🔑 Generating Soul Lock... ");
    io::stdout().flush().unwrap();

    let soul_secret = "ONE_LOVE_ONE_FAMILY_2026"; // This would be the "License Key"
    let lock_content = format!("{}:{}", anchor.fingerprint, soul_secret);
    
    let mut hasher = Sha256::new();
    hasher.update(lock_content.as_bytes());
    let soul_lock = hex::encode(hasher.finalize());
    println!("DONE.");

    // 3. The Binding (Write soul.lock)
    let lock_file = "soul.lock";
    match fs::write(lock_file, soul_lock.clone()) {
        Ok(_) => {
            println!("\n✅ BINDING SUCCESSFUL.");
            println!("   The Soul is now anchored to this Body.");
            println!("   Lock File: ./{}", lock_file);
            println!("   Hash:      {}", soul_lock);
            println!("\n   You may now start the JIS Router.");
        },
        Err(e) => {
            println!("\n❌ BINDING FAILED: {}", e);
        }
    }
}
