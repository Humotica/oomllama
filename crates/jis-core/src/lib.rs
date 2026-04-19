//! JIS Core — JTel Identity Standard
//!
//! Cryptographic identity where intent comes first.
//! Ed25519 Rust kernel with Python, WASM, and C bindings.
//!
//! Part of the HumoticaOS / AInternet identity stack.
//! IETF Draft: draft-vandemeent-jis-identity
//!
//! # Identifiers
//!
//! JIS uses `jis:` URIs as the primary format. The `did:jis:` format
//! remains supported for W3C DID compatibility. `.aint` domains provide
//! human-readable resolution via AINS (AInternet Name Service).
//!
//! ```text
//! jis:alice                      — Person
//! jis:humotica:root_idd          — AI agent in org
//! jis:device:6G:sensor-001       — IoT device
//! alice.aint                     — AInternet domain (resolves to jis:alice)
//! ```
//!
//! # Example
//!
//! ```rust
//! use jis_core::{JISEngine, JISDocumentBuilder};
//!
//! let engine = JISEngine::new();
//! let jis_id = engine.create_jis("alice");       // jis:alice
//! let did = engine.create_did("alice");           // did:jis:alice (compat)
//! let sig = engine.sign("I agree to these terms");
//! assert!(engine.verify("I agree to these terms", &sig));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use alloc::format;
use core::fmt;

use ed25519_dalek::{SigningKey, VerifyingKey, Signer, Verifier, Signature};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

// ============================================
// JIS + DID Types
// ============================================

/// A parsed JIS or DID identifier
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedDID {
    pub method: String,
    pub id: String,
}

/// A parsed JIS URI (jis:alice, jis:humotica:root_idd)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JISIdentity {
    /// The full JIS URI (e.g. "jis:alice")
    pub uri: String,
    /// The identifier part (e.g. "alice")
    pub id: String,
    /// Optional .aint domain (e.g. "alice.aint")
    pub aint_domain: Option<String>,
    /// Equivalent DID form (e.g. "did:jis:alice")
    pub did_compat: String,
}

/// JIS clearance level — maps to NATO classification + AINS trust tiers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JISClearance {
    /// Public — no restrictions
    Unclassified,
    /// Free tier — basic identity, rate limited
    Restricted,
    /// Verified — challenge-response completed
    Confidential,
    /// Trusted — multi-channel verification
    Secret,
    /// Core — full cryptographic ceremony
    TopSecret,
}

impl fmt::Display for JISClearance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unclassified => write!(f, "unclassified"),
            Self::Restricted => write!(f, "restricted"),
            Self::Confidential => write!(f, "confidential"),
            Self::Secret => write!(f, "secret"),
            Self::TopSecret => write!(f, "top-secret"),
        }
    }
}

/// A JIS Claim — declares identity + intent + clearance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JISClaim {
    /// JIS URI of the claimant
    pub identity: String,
    /// Ed25519 public key (hex)
    pub ed25519_pub: String,
    /// Clearance level
    pub clearance: JISClearance,
    /// Role (operator, agent, device, service)
    pub role: String,
    /// Department / organization
    pub dept: String,
    /// Timestamp of claim
    pub claimed_at: String,
    /// Ed25519 signature over the claim fields
    pub signature: String,
}

/// Verification method types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationMethodType {
    Ed25519VerificationKey2020,
    JsonWebKey2020,
    EcdsaSecp256k1VerificationKey2019,
}

impl fmt::Display for VerificationMethodType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ed25519VerificationKey2020 => write!(f, "Ed25519VerificationKey2020"),
            Self::JsonWebKey2020 => write!(f, "JsonWebKey2020"),
            Self::EcdsaSecp256k1VerificationKey2019 => write!(f, "EcdsaSecp256k1VerificationKey2019"),
        }
    }
}

/// A verification method in a DID document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMethod {
    pub id: String,
    #[serde(rename = "type")]
    pub method_type: String,
    pub controller: String,
    #[serde(rename = "publicKeyMultibase", skip_serializing_if = "Option::is_none")]
    pub public_key_multibase: Option<String>,
    #[serde(rename = "publicKeyHex", skip_serializing_if = "Option::is_none")]
    pub public_key_hex: Option<String>,
}

/// A service endpoint in a DID document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub id: String,
    #[serde(rename = "type")]
    pub service_type: String,
    #[serde(rename = "serviceEndpoint")]
    pub endpoint: String,
}

/// A complete DID Document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DIDDocument {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub controller: Option<String>,
    #[serde(rename = "verificationMethod", skip_serializing_if = "Vec::is_empty", default)]
    pub verification_method: Vec<VerificationMethod>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub authentication: Vec<String>,
    #[serde(rename = "assertionMethod", skip_serializing_if = "Vec::is_empty", default)]
    pub assertion_method: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub service: Vec<ServiceEndpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated: Option<String>,
}

// ============================================
// JIS Identifier Parsing and Validation
// ============================================

/// Valid JIS identifier characters
fn is_valid_jis_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == ':' || c == '.' || c == '_' || c == '-'
}

/// Parse a JIS URI into a JISIdentity
///
/// Accepts both `jis:alice` (canonical) and `did:jis:alice` (W3C compat)
pub fn parse_jis(input: &str) -> Option<JISIdentity> {
    let id = if input.starts_with("jis:") {
        &input[4..]
    } else if input.starts_with("did:jis:") {
        &input[8..]
    } else if input.ends_with(".aint") {
        // alice.aint → alice
        &input[..input.len() - 5]
    } else {
        return None;
    };

    if id.is_empty() || !id.chars().all(is_valid_jis_char) {
        return None;
    }

    // Derive .aint domain from the first part of the identifier
    let aint_base = id.split(':').next().unwrap_or(id);
    let aint_domain = if aint_base.contains('.') {
        None // Already dotted (e.g. sensor-001.aint would be weird)
    } else {
        Some(format!("{}.aint", aint_base))
    };

    Some(JISIdentity {
        uri: format!("jis:{}", id),
        id: id.to_string(),
        aint_domain,
        did_compat: format!("did:jis:{}", id),
    })
}

/// Parse a DID string into its components (legacy compat)
pub fn parse_did(did: &str) -> Option<ParsedDID> {
    if !did.starts_with("did:") {
        return None;
    }

    let parts: Vec<&str> = did.splitn(3, ':').collect();
    if parts.len() < 3 {
        return None;
    }

    Some(ParsedDID {
        method: parts[1].to_string(),
        id: parts[2].to_string(),
    })
}

/// Validate a JIS identifier (accepts jis: and did:jis: formats)
pub fn is_valid_jis(input: &str) -> bool {
    parse_jis(input).is_some()
}

/// Validate a did:jis identifier (legacy compat)
pub fn is_valid_did(did: &str) -> bool {
    if !did.starts_with("did:jis:") {
        return false;
    }
    is_valid_jis(did)
}

/// Create a JIS URI from parts
///
/// ```rust
/// let uri = jis_core::create_jis(&["humotica", "root_idd"]).unwrap();
/// assert_eq!(uri, "jis:humotica:root_idd");
/// ```
pub fn create_jis(parts: &[&str]) -> Result<String, &'static str> {
    if parts.is_empty() {
        return Err("JIS identifier must have at least one part");
    }

    let id = parts.join(":");

    if !id.chars().all(is_valid_jis_char) {
        return Err("JIS identifier contains invalid characters");
    }

    Ok(format!("jis:{}", id))
}

/// Create a did:jis identifier from parts (legacy compat)
pub fn create_did(parts: &[&str]) -> Result<String, &'static str> {
    if parts.is_empty() {
        return Err("DID must have at least one identifier part");
    }

    let id = parts.join(":");

    if !id.chars().all(is_valid_jis_char) {
        return Err("DID contains invalid characters");
    }

    Ok(format!("did:jis:{}", id))
}

/// Convert between JIS and .aint domain
///
/// ```rust
/// assert_eq!(jis_core::jis_to_aint("jis:alice"), Some("alice.aint".to_string()));
/// assert_eq!(jis_core::aint_to_jis("alice.aint"), Some("jis:alice".to_string()));
/// ```
pub fn jis_to_aint(jis_uri: &str) -> Option<String> {
    parse_jis(jis_uri).and_then(|id| id.aint_domain)
}

pub fn aint_to_jis(domain: &str) -> Option<String> {
    parse_jis(domain).map(|id| id.uri)
}

// ============================================
// DID Document Builder
// ============================================

/// Builder for DID Documents
pub struct DIDDocumentBuilder {
    doc: DIDDocument,
}

impl DIDDocumentBuilder {
    /// Create a new DID Document builder
    pub fn new(did: &str) -> Result<Self, &'static str> {
        if !is_valid_did(did) {
            return Err("Invalid DID");
        }

        let now = get_timestamp();

        Ok(Self {
            doc: DIDDocument {
                context: vec![
                    "https://www.w3.org/ns/did/v1".to_string(),
                    "https://w3id.org/security/suites/ed25519-2020/v1".to_string(),
                    "https://humotica.com/ns/jis/v1".to_string(),
                ],
                id: did.to_string(),
                controller: None,
                verification_method: Vec::new(),
                authentication: Vec::new(),
                assertion_method: Vec::new(),
                service: Vec::new(),
                created: Some(now.clone()),
                updated: Some(now),
            },
        })
    }

    /// Set the controller
    pub fn set_controller(mut self, controller: &str) -> Self {
        self.doc.controller = Some(controller.to_string());
        self
    }

    /// Add an Ed25519 verification method
    pub fn add_verification_method_ed25519(mut self, key_id: &str, public_key_hex: &str) -> Self {
        let full_id = format!("{}#{}", self.doc.id, key_id);

        self.doc.verification_method.push(VerificationMethod {
            id: full_id,
            method_type: VerificationMethodType::Ed25519VerificationKey2020.to_string(),
            controller: self.doc.id.clone(),
            public_key_multibase: None,
            public_key_hex: Some(public_key_hex.to_string()),
        });
        self
    }

    /// Add a verification method with multibase encoding
    pub fn add_verification_method_multibase(mut self, key_id: &str, method_type: VerificationMethodType, public_key_multibase: &str) -> Self {
        let full_id = format!("{}#{}", self.doc.id, key_id);

        self.doc.verification_method.push(VerificationMethod {
            id: full_id,
            method_type: method_type.to_string(),
            controller: self.doc.id.clone(),
            public_key_multibase: Some(public_key_multibase.to_string()),
            public_key_hex: None,
        });
        self
    }

    /// Add authentication method reference
    pub fn add_authentication(mut self, key_id: &str) -> Self {
        let full_id = format!("{}#{}", self.doc.id, key_id);
        self.doc.authentication.push(full_id);
        self
    }

    /// Add assertion method reference
    pub fn add_assertion_method(mut self, key_id: &str) -> Self {
        let full_id = format!("{}#{}", self.doc.id, key_id);
        self.doc.assertion_method.push(full_id);
        self
    }

    /// Add a service endpoint
    pub fn add_service(mut self, service_id: &str, service_type: &str, endpoint: &str) -> Self {
        let full_id = format!("{}#{}", self.doc.id, service_id);

        self.doc.service.push(ServiceEndpoint {
            id: full_id,
            service_type: service_type.to_string(),
            endpoint: endpoint.to_string(),
        });
        self
    }

    /// Add bilateral consent service
    pub fn add_consent_service(self, endpoint: &str) -> Self {
        self.add_service("bilateral-consent", "BilateralConsentService", endpoint)
    }

    /// Add TIBET provenance service
    pub fn add_tibet_service(self, endpoint: &str) -> Self {
        self.add_service("tibet-provenance", "TIBETProvenanceService", endpoint)
    }

    /// Add AINS (.aint domain) resolution service
    pub fn add_ains_service(self, endpoint: &str) -> Self {
        self.add_service("ains-resolution", "AINSResolutionService", endpoint)
    }

    /// Add I-Poll messaging service
    pub fn add_ipoll_service(self, endpoint: &str) -> Self {
        self.add_service("ipoll-messaging", "IPollMessagingService", endpoint)
    }

    /// Add MUX intent routing service
    pub fn add_mux_service(self, endpoint: &str) -> Self {
        self.add_service("tibet-mux", "TIBETMuxService", endpoint)
    }

    /// Build the DID document
    pub fn build(mut self) -> DIDDocument {
        self.doc.updated = Some(get_timestamp());
        self.doc
    }

    /// Build and return as JSON string
    #[cfg(feature = "std")]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.doc).unwrap_or_default()
    }
}

// ============================================
// DID Engine (with cryptography)
// ============================================

/// DID Engine with Ed25519 key management
pub struct DIDEngine {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl DIDEngine {
    /// Create a new DID Engine with a fresh keypair
    pub fn new() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Create from existing secret key bytes (32 bytes)
    pub fn from_secret_key(secret: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(secret);
        let verifying_key = signing_key.verifying_key();

        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Get the public key as hex string
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.verifying_key.as_bytes())
    }

    /// Get the public key as multibase (z-base58btc prefix)
    pub fn public_key_multibase(&self) -> String {
        // Simple hex with 'f' prefix (multibase hex)
        format!("f{}", self.public_key_hex())
    }

    /// Create a JIS identity (canonical format)
    pub fn create_jis(&self, id: &str) -> String {
        format!("jis:{}", id)
    }

    /// Create a new did:jis identifier (W3C compat)
    pub fn create_did(&self, id: &str) -> String {
        format!("did:jis:{}", id)
    }

    /// Create a JIS identity from the public key hash
    pub fn create_jis_from_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.verifying_key.as_bytes());
        let hash = hasher.finalize();
        let short_hash = &hex::encode(hash)[..16];
        format!("jis:{}", short_hash)
    }

    /// Create a DID from the public key hash (legacy compat)
    pub fn create_did_from_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.verifying_key.as_bytes());
        let hash = hasher.finalize();
        let short_hash = &hex::encode(hash)[..16];
        format!("did:jis:{}", short_hash)
    }

    /// Get the .aint domain for an identity
    pub fn aint_domain(&self, id: &str) -> String {
        format!("{}.aint", id)
    }

    /// Create a signed JIS claim
    #[cfg(feature = "std")]
    pub fn create_claim(&self, id: &str, clearance: JISClearance, role: &str, dept: &str) -> JISClaim {
        let claimed_at = get_timestamp();
        let claim_data = format!("{}:{}:{}:{}:{}", id, self.public_key_hex(), clearance, role, claimed_at);
        let signature = self.sign_string(&claim_data);

        JISClaim {
            identity: format!("jis:{}", id),
            ed25519_pub: self.public_key_hex(),
            clearance,
            role: role.to_string(),
            dept: dept.to_string(),
            claimed_at,
            signature,
        }
    }

    /// Sign data and return hex-encoded signature
    pub fn sign(&self, data: &[u8]) -> String {
        let signature = self.signing_key.sign(data);
        hex::encode(signature.to_bytes())
    }

    /// Sign a string message
    pub fn sign_string(&self, message: &str) -> String {
        self.sign(message.as_bytes())
    }

    /// Verify a signature
    pub fn verify(&self, data: &[u8], signature_hex: &str) -> bool {
        let sig_bytes = match hex::decode(signature_hex) {
            Ok(b) => b,
            Err(_) => return false,
        };

        let signature = match Signature::from_slice(&sig_bytes) {
            Ok(s) => s,
            Err(_) => return false,
        };

        self.verifying_key.verify(data, &signature).is_ok()
    }

    /// Verify with a different public key
    pub fn verify_with_key(data: &[u8], signature_hex: &str, public_key_hex: &str) -> bool {
        let pub_bytes = match hex::decode(public_key_hex) {
            Ok(b) => b,
            Err(_) => return false,
        };

        let pub_key_bytes: [u8; 32] = match pub_bytes.try_into() {
            Ok(b) => b,
            Err(_) => return false,
        };

        let verifying_key = match VerifyingKey::from_bytes(&pub_key_bytes) {
            Ok(k) => k,
            Err(_) => return false,
        };

        let sig_bytes = match hex::decode(signature_hex) {
            Ok(b) => b,
            Err(_) => return false,
        };

        let signature = match Signature::from_slice(&sig_bytes) {
            Ok(s) => s,
            Err(_) => return false,
        };

        verifying_key.verify(data, &signature).is_ok()
    }

    /// Create a signed DID document
    #[cfg(feature = "std")]
    pub fn create_signed_document(&self, did: &str) -> Result<String, &'static str> {
        let doc = DIDDocumentBuilder::new(did)?
            .add_verification_method_ed25519("key-1", &self.public_key_hex())
            .add_authentication("key-1")
            .add_assertion_method("key-1")
            .build();

        let doc_json = serde_json::to_string(&doc).map_err(|_| "Serialization failed")?;
        let signature = self.sign_string(&doc_json);

        // Return document with proof
        let signed = serde_json::json!({
            "document": doc,
            "proof": {
                "type": "Ed25519Signature2020",
                "created": get_timestamp(),
                "verificationMethod": format!("{}#key-1", did),
                "proofPurpose": "assertionMethod",
                "proofValue": signature
            }
        });

        serde_json::to_string_pretty(&signed).map_err(|_| "Serialization failed")
    }
}

impl Default for DIDEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================
// Timestamp helper
// ============================================

fn get_timestamp() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

// ============================================
// Python Bindings
// ============================================

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PyModuleMethods;
    use pyo3::exceptions::PyValueError;
    use pyo3::Bound;

    #[pyclass(name = "DIDEngine")]
    pub struct PyDIDEngine {
        engine: DIDEngine,
    }

    #[pymethods]
    impl PyDIDEngine {
        #[new]
        fn new() -> Self {
            Self {
                engine: DIDEngine::new(),
            }
        }

        #[staticmethod]
        fn from_secret_key(secret_hex: &str) -> PyResult<Self> {
            let secret_bytes = hex::decode(secret_hex)
                .map_err(|_| PyValueError::new_err("Invalid hex string"))?;

            let secret: [u8; 32] = secret_bytes.try_into()
                .map_err(|_| PyValueError::new_err("Secret key must be 32 bytes"))?;

            Ok(Self {
                engine: DIDEngine::from_secret_key(&secret),
            })
        }

        #[getter]
        fn public_key(&self) -> String {
            self.engine.public_key_hex()
        }

        #[getter]
        fn public_key_multibase(&self) -> String {
            self.engine.public_key_multibase()
        }

        fn create_jis(&self, id: &str) -> String {
            self.engine.create_jis(id)
        }

        fn create_did(&self, id: &str) -> String {
            self.engine.create_did(id)
        }

        fn create_jis_from_key(&self) -> String {
            self.engine.create_jis_from_key()
        }

        fn create_did_from_key(&self) -> String {
            self.engine.create_did_from_key()
        }

        fn aint_domain(&self, id: &str) -> String {
            self.engine.aint_domain(id)
        }

        fn sign(&self, message: &str) -> String {
            self.engine.sign_string(message)
        }

        fn verify(&self, message: &str, signature: &str) -> bool {
            self.engine.verify(message.as_bytes(), signature)
        }

        #[staticmethod]
        fn verify_with_key(message: &str, signature: &str, public_key: &str) -> bool {
            DIDEngine::verify_with_key(message.as_bytes(), signature, public_key)
        }

        fn create_document(&self, did: &str) -> PyResult<String> {
            self.engine.create_signed_document(did)
                .map_err(|e| PyValueError::new_err(e))
        }
    }

    #[pyclass(name = "DIDDocumentBuilder")]
    pub struct PyDIDDocumentBuilder {
        did: String,
        controller: Option<String>,
        verification_methods: Vec<(String, String)>, // (key_id, public_key_hex)
        authentication: Vec<String>,
        assertion_methods: Vec<String>,
        services: Vec<(String, String, String)>, // (id, type, endpoint)
    }

    #[pymethods]
    impl PyDIDDocumentBuilder {
        #[new]
        fn new(did: &str) -> PyResult<Self> {
            if !is_valid_did(did) {
                return Err(PyValueError::new_err("Invalid DID"));
            }

            Ok(Self {
                did: did.to_string(),
                controller: None,
                verification_methods: Vec::new(),
                authentication: Vec::new(),
                assertion_methods: Vec::new(),
                services: Vec::new(),
            })
        }

        fn set_controller(&mut self, controller: &str) -> PyResult<()> {
            self.controller = Some(controller.to_string());
            Ok(())
        }

        fn add_verification_method(&mut self, key_id: &str, public_key_hex: &str) -> PyResult<()> {
            self.verification_methods.push((key_id.to_string(), public_key_hex.to_string()));
            Ok(())
        }

        fn add_authentication(&mut self, key_id: &str) -> PyResult<()> {
            self.authentication.push(key_id.to_string());
            Ok(())
        }

        fn add_assertion_method(&mut self, key_id: &str) -> PyResult<()> {
            self.assertion_methods.push(key_id.to_string());
            Ok(())
        }

        fn add_service(&mut self, service_id: &str, service_type: &str, endpoint: &str) -> PyResult<()> {
            self.services.push((service_id.to_string(), service_type.to_string(), endpoint.to_string()));
            Ok(())
        }

        fn add_consent_service(&mut self, endpoint: &str) -> PyResult<()> {
            self.add_service("bilateral-consent", "BilateralConsentService", endpoint)
        }

        fn add_tibet_service(&mut self, endpoint: &str) -> PyResult<()> {
            self.add_service("tibet-provenance", "TIBETProvenanceService", endpoint)
        }

        fn build(&self) -> PyResult<String> {
            let mut builder = DIDDocumentBuilder::new(&self.did)
                .map_err(|e| PyValueError::new_err(e))?;

            if let Some(ref controller) = self.controller {
                builder = builder.set_controller(controller);
            }

            for (key_id, public_key) in &self.verification_methods {
                builder = builder.add_verification_method_ed25519(key_id, public_key);
            }

            for key_id in &self.authentication {
                builder = builder.add_authentication(key_id);
            }

            for key_id in &self.assertion_methods {
                builder = builder.add_assertion_method(key_id);
            }

            for (service_id, service_type, endpoint) in &self.services {
                builder = builder.add_service(service_id, service_type, endpoint);
            }

            Ok(builder.to_json())
        }
    }

    /// Parse a JIS URI (jis:, did:jis:, or .aint)
    #[pyfunction]
    fn parse_jis_py(input: &str) -> PyResult<Option<(String, String, Option<String>, String)>> {
        Ok(parse_jis(input).map(|id| (id.uri, id.id, id.aint_domain, id.did_compat)))
    }

    /// Parse a DID string (legacy compat)
    #[pyfunction]
    fn parse_did_py(did: &str) -> PyResult<Option<(String, String)>> {
        Ok(parse_did(did).map(|p| (p.method, p.id)))
    }

    /// Validate a JIS identifier (jis: or did:jis:)
    #[pyfunction]
    fn is_valid_jis_py(input: &str) -> bool {
        is_valid_jis(input)
    }

    /// Validate a did:jis identifier (legacy compat)
    #[pyfunction]
    fn is_valid_did_py(did: &str) -> bool {
        is_valid_did(did)
    }

    /// Create a JIS URI from parts
    #[pyfunction]
    fn create_jis_py(parts: Vec<String>) -> PyResult<String> {
        let parts_ref: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
        create_jis(&parts_ref).map_err(|e| PyValueError::new_err(e))
    }

    /// Create a did:jis identifier (legacy compat)
    #[pyfunction]
    fn create_did_py(parts: Vec<String>) -> PyResult<String> {
        let parts_ref: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
        create_did(&parts_ref).map_err(|e| PyValueError::new_err(e))
    }

    /// Convert JIS URI to .aint domain
    #[pyfunction]
    fn jis_to_aint_py(jis_uri: &str) -> Option<String> {
        jis_to_aint(jis_uri)
    }

    /// Convert .aint domain to JIS URI
    #[pyfunction]
    fn aint_to_jis_py(domain: &str) -> Option<String> {
        aint_to_jis(domain)
    }

    #[pymodule]
    fn jis_core(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
        m.add_class::<PyDIDEngine>()?;
        m.add_class::<PyDIDDocumentBuilder>()?;
        // JIS functions (canonical)
        m.add_function(wrap_pyfunction!(parse_jis_py, m)?)?;
        m.add_function(wrap_pyfunction!(is_valid_jis_py, m)?)?;
        m.add_function(wrap_pyfunction!(create_jis_py, m)?)?;
        m.add_function(wrap_pyfunction!(jis_to_aint_py, m)?)?;
        m.add_function(wrap_pyfunction!(aint_to_jis_py, m)?)?;
        // DID compat
        m.add_function(wrap_pyfunction!(parse_did_py, m)?)?;
        m.add_function(wrap_pyfunction!(is_valid_did_py, m)?)?;
        m.add_function(wrap_pyfunction!(create_did_py, m)?)?;
        m.add("__version__", "0.3.0")?;
        Ok(())
    }
}

// ============================================
// WASM Bindings
// ============================================

#[cfg(feature = "wasm")]
mod wasm {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmDIDEngine {
        engine: DIDEngine,
    }

    #[wasm_bindgen]
    impl WasmDIDEngine {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                engine: DIDEngine::new(),
            }
        }

        #[wasm_bindgen(js_name = fromSecretKey)]
        pub fn from_secret_key(secret_hex: &str) -> Result<WasmDIDEngine, JsValue> {
            let secret_bytes = hex::decode(secret_hex)
                .map_err(|_| JsValue::from_str("Invalid hex string"))?;

            let secret: [u8; 32] = secret_bytes.try_into()
                .map_err(|_| JsValue::from_str("Secret key must be 32 bytes"))?;

            Ok(Self {
                engine: DIDEngine::from_secret_key(&secret),
            })
        }

        #[wasm_bindgen(getter, js_name = publicKey)]
        pub fn public_key(&self) -> String {
            self.engine.public_key_hex()
        }

        #[wasm_bindgen(getter, js_name = publicKeyMultibase)]
        pub fn public_key_multibase(&self) -> String {
            self.engine.public_key_multibase()
        }

        #[wasm_bindgen(js_name = createDid)]
        pub fn create_did(&self, id: &str) -> String {
            self.engine.create_did(id)
        }

        #[wasm_bindgen(js_name = createDidFromKey)]
        pub fn create_did_from_key(&self) -> String {
            self.engine.create_did_from_key()
        }

        pub fn sign(&self, message: &str) -> String {
            self.engine.sign_string(message)
        }

        pub fn verify(&self, message: &str, signature: &str) -> bool {
            self.engine.verify(message.as_bytes(), signature)
        }

        #[wasm_bindgen(js_name = verifyWithKey)]
        pub fn verify_with_key(message: &str, signature: &str, public_key: &str) -> bool {
            DIDEngine::verify_with_key(message.as_bytes(), signature, public_key)
        }

        #[wasm_bindgen(js_name = createDocument)]
        pub fn create_document(&self, did: &str) -> Result<String, JsValue> {
            self.engine.create_signed_document(did)
                .map_err(|e| JsValue::from_str(e))
        }
    }

    /// Parse a DID string
    #[wasm_bindgen(js_name = parseDid)]
    pub fn parse_did_wasm(did: &str) -> JsValue {
        match parse_did(did) {
            Some(parsed) => {
                ::serde_wasm_bindgen::to_value(&parsed).unwrap_or(JsValue::NULL)
            }
            None => JsValue::NULL,
        }
    }

    /// Validate a did:jis identifier
    #[wasm_bindgen(js_name = isValidDid)]
    pub fn is_valid_did_wasm(did: &str) -> bool {
        is_valid_did(did)
    }

    /// Create a did:jis identifier
    #[wasm_bindgen(js_name = createDid)]
    pub fn create_did_wasm(parts: Vec<String>) -> Result<String, JsValue> {
        let parts_ref: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
        create_did(&parts_ref).map_err(|e| JsValue::from_str(e))
    }
}

// ============================================
// C Bindings (FFI)
// ============================================

#[cfg(feature = "cbind")]
mod cbind {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;
    use std::ptr;

    /// Opaque engine handle for C
    pub struct CEngine {
        inner: DIDEngine,
    }

    /// Create a new DID engine
    #[no_mangle]
    pub extern "C" fn did_engine_new() -> *mut CEngine {
        let engine = Box::new(CEngine {
            inner: DIDEngine::new(),
        });
        Box::into_raw(engine)
    }

    /// Create engine from secret key (hex string)
    #[no_mangle]
    pub extern "C" fn did_engine_from_secret(secret_hex: *const c_char) -> *mut CEngine {
        if secret_hex.is_null() {
            return ptr::null_mut();
        }

        let secret_str = unsafe {
            match CStr::from_ptr(secret_hex).to_str() {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            }
        };

        let secret_bytes = match hex::decode(secret_str) {
            Ok(b) => b,
            Err(_) => return ptr::null_mut(),
        };

        if secret_bytes.len() != 32 {
            return ptr::null_mut();
        }

        let secret: [u8; 32] = match secret_bytes.try_into() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        let engine = Box::new(CEngine {
            inner: DIDEngine::from_secret_key(&secret),
        });
        Box::into_raw(engine)
    }

    /// Free the engine
    #[no_mangle]
    pub extern "C" fn did_engine_free(engine: *mut CEngine) {
        if !engine.is_null() {
            unsafe {
                drop(Box::from_raw(engine));
            }
        }
    }

    /// Get public key as hex string
    #[no_mangle]
    pub extern "C" fn did_get_public_key(engine: *const CEngine) -> *mut c_char {
        if engine.is_null() {
            return ptr::null_mut();
        }

        let engine = unsafe { &*engine };
        let pubkey = engine.inner.public_key_hex();

        match CString::new(pubkey) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    }

    /// Get public key in multibase format
    #[no_mangle]
    pub extern "C" fn did_get_public_key_multibase(engine: *const CEngine) -> *mut c_char {
        if engine.is_null() {
            return ptr::null_mut();
        }

        let engine = unsafe { &*engine };
        let pubkey = engine.inner.public_key_multibase();

        match CString::new(pubkey) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    }

    /// Create a did:jis identifier
    #[no_mangle]
    pub extern "C" fn did_create(engine: *const CEngine, id: *const c_char) -> *mut c_char {
        if engine.is_null() || id.is_null() {
            return ptr::null_mut();
        }

        let engine = unsafe { &*engine };
        let id_str = unsafe {
            match CStr::from_ptr(id).to_str() {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            }
        };

        let did = engine.inner.create_did(id_str);

        match CString::new(did) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    }

    /// Create a DID from public key hash
    #[no_mangle]
    pub extern "C" fn did_create_from_key(engine: *const CEngine) -> *mut c_char {
        if engine.is_null() {
            return ptr::null_mut();
        }

        let engine = unsafe { &*engine };
        let did = engine.inner.create_did_from_key();

        match CString::new(did) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    }

    /// Parse a DID string
    #[no_mangle]
    pub extern "C" fn did_parse(
        did: *const c_char,
        method_out: *mut c_char,
        id_out: *mut c_char,
    ) -> bool {
        if did.is_null() || method_out.is_null() || id_out.is_null() {
            return false;
        }

        let did_str = unsafe {
            match CStr::from_ptr(did).to_str() {
                Ok(s) => s,
                Err(_) => return false,
            }
        };

        match parse_did(did_str) {
            Some(parsed) => {
                unsafe {
                    // Copy method (max 31 chars + null)
                    let method_bytes = parsed.method.as_bytes();
                    let method_len = method_bytes.len().min(31);
                    std::ptr::copy_nonoverlapping(method_bytes.as_ptr(), method_out as *mut u8, method_len);
                    *method_out.add(method_len) = 0;

                    // Copy id (max 255 chars + null)
                    let id_bytes = parsed.id.as_bytes();
                    let id_len = id_bytes.len().min(255);
                    std::ptr::copy_nonoverlapping(id_bytes.as_ptr(), id_out as *mut u8, id_len);
                    *id_out.add(id_len) = 0;
                }
                true
            }
            None => false,
        }
    }

    /// Validate a did:jis identifier
    #[no_mangle]
    pub extern "C" fn did_is_valid(did: *const c_char) -> bool {
        if did.is_null() {
            return false;
        }

        let did_str = unsafe {
            match CStr::from_ptr(did).to_str() {
                Ok(s) => s,
                Err(_) => return false,
            }
        };

        is_valid_did(did_str)
    }

    /// Create a signed DID document
    #[no_mangle]
    pub extern "C" fn did_create_document(engine: *const CEngine, did: *const c_char) -> *mut c_char {
        if engine.is_null() || did.is_null() {
            return ptr::null_mut();
        }

        let engine = unsafe { &*engine };
        let did_str = unsafe {
            match CStr::from_ptr(did).to_str() {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            }
        };

        match engine.inner.create_signed_document(did_str) {
            Ok(doc) => match CString::new(doc) {
                Ok(s) => s.into_raw(),
                Err(_) => ptr::null_mut(),
            },
            Err(_) => ptr::null_mut(),
        }
    }

    /// Sign a message
    #[no_mangle]
    pub extern "C" fn did_sign(engine: *const CEngine, message: *const c_char) -> *mut c_char {
        if engine.is_null() || message.is_null() {
            return ptr::null_mut();
        }

        let engine = unsafe { &*engine };
        let msg_str = unsafe {
            match CStr::from_ptr(message).to_str() {
                Ok(s) => s,
                Err(_) => return ptr::null_mut(),
            }
        };

        let signature = engine.inner.sign_string(msg_str);

        match CString::new(signature) {
            Ok(s) => s.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    }

    /// Verify a signature
    #[no_mangle]
    pub extern "C" fn did_verify(
        engine: *const CEngine,
        message: *const c_char,
        signature: *const c_char,
    ) -> bool {
        if engine.is_null() || message.is_null() || signature.is_null() {
            return false;
        }

        let engine = unsafe { &*engine };
        let msg_str = unsafe {
            match CStr::from_ptr(message).to_str() {
                Ok(s) => s,
                Err(_) => return false,
            }
        };
        let sig_str = unsafe {
            match CStr::from_ptr(signature).to_str() {
                Ok(s) => s,
                Err(_) => return false,
            }
        };

        engine.inner.verify(msg_str.as_bytes(), sig_str)
    }

    /// Verify with external public key
    #[no_mangle]
    pub extern "C" fn did_verify_with_key(
        message: *const c_char,
        signature: *const c_char,
        public_key_hex: *const c_char,
    ) -> bool {
        if message.is_null() || signature.is_null() || public_key_hex.is_null() {
            return false;
        }

        let msg_str = unsafe {
            match CStr::from_ptr(message).to_str() {
                Ok(s) => s,
                Err(_) => return false,
            }
        };
        let sig_str = unsafe {
            match CStr::from_ptr(signature).to_str() {
                Ok(s) => s,
                Err(_) => return false,
            }
        };
        let pubkey_str = unsafe {
            match CStr::from_ptr(public_key_hex).to_str() {
                Ok(s) => s,
                Err(_) => return false,
            }
        };

        DIDEngine::verify_with_key(msg_str.as_bytes(), sig_str, pubkey_str)
    }

    /// Free a string allocated by DID functions
    #[no_mangle]
    pub extern "C" fn did_free_string(s: *mut c_char) {
        if !s.is_null() {
            unsafe {
                drop(CString::from_raw(s));
            }
        }
    }

    /// Get DID:JIS version
    #[no_mangle]
    pub extern "C" fn did_version() -> *const c_char {
        static VERSION: &[u8] = b"0.1.0\0";
        VERSION.as_ptr() as *const c_char
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_did() {
        let parsed = parse_did("did:jis:alice").unwrap();
        assert_eq!(parsed.method, "jis");
        assert_eq!(parsed.id, "alice");

        let parsed2 = parse_did("did:jis:org:company:employee").unwrap();
        assert_eq!(parsed2.id, "org:company:employee");

        assert!(parse_did("invalid").is_none());
    }

    #[test]
    fn test_is_valid_did() {
        assert!(is_valid_did("did:jis:alice"));
        assert!(is_valid_did("did:jis:org:company:employee42"));
        assert!(!is_valid_did("did:web:example.com"));
        assert!(!is_valid_did("invalid"));
    }

    #[test]
    fn test_create_did() {
        assert_eq!(create_did(&["alice"]).unwrap(), "did:jis:alice");
        assert_eq!(create_did(&["org", "company", "42"]).unwrap(), "did:jis:org:company:42");
    }

    #[test]
    fn test_did_engine() {
        let engine = DIDEngine::new();

        // Check public key is valid hex
        let pk = engine.public_key_hex();
        assert_eq!(pk.len(), 64); // 32 bytes = 64 hex chars

        // Test signing and verification
        let message = "Hello, DID!";
        let signature = engine.sign_string(message);
        assert!(engine.verify(message.as_bytes(), &signature));

        // Wrong message should fail
        assert!(!engine.verify(b"Wrong message", &signature));
    }

    #[test]
    fn test_did_document_builder() {
        let engine = DIDEngine::new();
        let did = "did:jis:alice";

        let doc = DIDDocumentBuilder::new(did).unwrap()
            .add_verification_method_ed25519("key-1", &engine.public_key_hex())
            .add_authentication("key-1")
            .add_consent_service("https://api.example.com/consent")
            .build();

        assert_eq!(doc.id, did);
        assert_eq!(doc.verification_method.len(), 1);
        assert_eq!(doc.authentication.len(), 1);
        assert_eq!(doc.service.len(), 1);
    }
}
