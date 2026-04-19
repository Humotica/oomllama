//! ANCHOR - The Physical Binding
//!
//! This module generates the unique Hardware Hash (HWID) that anchors
//! the AETHER Soul (Software) to the REDSTONE Body (Hardware).
//!
//! "Where the digital meets the physical."

use sha2::{Digest, Sha256};
use std::fs;
use std::process::Command;

/// The Hardware Anchor
#[derive(Debug, Clone)]
pub struct Anchor {
    pub node_id: String,
    pub architecture: String,
    pub fingerprint: String,
}

impl Anchor {
    /// Cast the anchor - analyze the current hardware
    pub fn cast() -> Self {
        let node_id = Self::get_machine_id();
        let architecture = std::env::consts::ARCH.to_string();
        let gpu_uuid = Self::get_gpu_uuid();
        
        // Combine into a sovereign fingerprint
        let raw = format!("{}:{}:{}", node_id, architecture, gpu_uuid);
        let mut hasher = Sha256::new();
        hasher.update(raw.as_bytes());
        let fingerprint = hex::encode(hasher.finalize());

        Self {
            node_id,
            architecture,
            fingerprint,
        }
    }

    /// Read the immutable machine ID (Systemd/DBus)
    fn get_machine_id() -> String {
        // Try /etc/machine-id first (Standard Linux)
        if let Ok(id) = fs::read_to_string("/etc/machine-id") {
            return id.trim().to_string();
        }
        
        // Fallback to DMI Product UUID (Bare Metal)
        if let Ok(id) = fs::read_to_string("/sys/class/dmi/id/product_uuid") {
            return id.trim().to_string();
        }

        // Fallback for containers/dev: Hostname
        if let Ok(id) = fs::read_to_string("/etc/hostname") {
            return format!("host-{}", id.trim());
        }

        "unknown-floating-soul".to_string()
    }

    /// Try to get GPU UUID via nvidia-smi (if available)
    fn get_gpu_uuid() -> String {
        // We use a simple shell command here to avoid complex bindings for the PoC
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=uuid", "--format=csv,noheader"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let s = String::from_utf8_lossy(&o.stdout);
                s.trim().to_string()
            },
            _ => "no-gpu-detected".to_string(), // CPU-only node
        }
    }

    /// Verify if this soul is allowed to inhabit this body
    pub fn verify_binding(&self, allowed_hash: &str) -> bool {
        self.fingerprint == allowed_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_cast() {
        let anchor = Anchor::cast();
        println!("⚓ ANCHOR CAST ⚓");
        println!("Node ID:      {}", anchor.node_id);
        println!("Arch:         {}", anchor.architecture);
        println!("Fingerprint:  {}", anchor.fingerprint);
        
        assert!(!anchor.node_id.is_empty());
        assert_eq!(anchor.fingerprint.len(), 64); // SHA256 hex
    }
}
