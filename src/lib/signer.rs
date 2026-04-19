//! TIBET Signer Module
//! 
//! "Code is Intent. Signature is Proof."
//!
//! This module handles the cryptographic signing of commits and events using JIS keys.
//! It is the core of the Humotica Sovereign Trust Layer.

use sha2::{Sha256, Digest};
use hmac::{Hmac, Mac};
use hex;
use std::time::{SystemTime, UNIX_EPOCH};

// Placeholder for JIS Identity (in production, read from secure vault/keychain)
const DEV_SECRET_KEY: &str = "JIS-DEV-SECRET-DO-NOT-USE-IN-PROD"; 

#[derive(Debug)]
pub struct TibetToken {
    pub signature: String,
    pub timestamp: u64,
    pub agent_id: String,
    pub intent_hash: String,
}

impl TibetToken {
    /// Formats the token as a string for inclusion in git commits
    pub fn to_string(&self) -> String {
        format!("TIBET-SIG-v1:{}:{}:{}:{}", self.agent_id, self.timestamp, self.intent_hash, self.signature)
    }
}

pub struct TibetSigner {
    agent_id: String,
    secret_key: String,
}

impl TibetSigner {
    pub fn new(agent_id: &str, secret_key: Option<&str>) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            secret_key: secret_key.unwrap_or(DEV_SECRET_KEY).to_string(),
        }
    }

    /// Sign a payload (e.g., git diff or commit message)
    pub fn sign(&self, payload: &str) -> Result<TibetToken, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();

        // 1. Calculate Intent Hash (SHA256 of payload)
        let mut hasher = Sha256::new();
        hasher.update(payload.as_bytes());
        let intent_hash = hex::encode(hasher.finalize());

        // 2. Create Canonical String to Sign (JIS-Format)
        // Format: JIS-v1:<agent>:<timestamp>:<intent_hash>
        let canonical_string = format!("JIS-v1:{}:{}:{}", self.agent_id, timestamp, intent_hash);

        // 3. Sign with HMAC-SHA256 (Proof of Authority)
        let mut mac = Hmac::<Sha256>::new_from_slice(self.secret_key.as_bytes())
            .map_err(|_| "Invalid key length".to_string())?;
        mac.update(canonical_string.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        Ok(TibetToken {
            signature,
            timestamp,
            agent_id: self.agent_id.clone(),
            intent_hash,
        })
    }

    /// Verify a received TIBET token
    pub fn verify(&self, payload: &str, token_str: &str) -> bool {
        // Parse token string: TIBET-SIG-v1:agent:ts:hash:sig
        let parts: Vec<&str> = token_str.split(':').collect();
        if parts.len() != 5 || parts[0] != "TIBET-SIG-v1" {
            return false;
        }

        let _agent = parts[1]; // In real impl, fetch public key for this agent
        let ts = parts[2];
        let claimed_hash = parts[3];
        let claimed_sig = parts[4];

        // 1. Verify Hash
        let mut hasher = Sha256::new();
        hasher.update(payload.as_bytes());
        let real_hash = hex::encode(hasher.finalize());
        
        if real_hash != claimed_hash {
            println!("❌ Hash mismatch! Payload modified.");
            return false;
        }

        // 2. Verify Signature
        let canonical_string = format!("JIS-v1:{}:{}:{}", self.agent_id, ts, real_hash);
        let mut mac = Hmac::<Sha256>::new_from_slice(self.secret_key.as_bytes()).unwrap();
        mac.update(canonical_string.as_bytes());
        let real_sig = hex::encode(mac.finalize().into_bytes());

        if real_sig == claimed_sig {
            true
        } else {
            println!("❌ Signature forgery detected!");
            false
        }
    }
}
