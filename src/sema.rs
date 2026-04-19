//! SEMA - Semantic Message Addressing
//!
//! Route messages by meaning, not just by name.
//! "Who can help with vision?" -> routes to IDDs with vision capability.
//!
//! One love, one fAmIly!

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub use crate::types::Capability;

/// A semantic address - can target by name, capability, or group
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SemanticAddress {
    /// Direct address by name
    Direct { name: String },
    /// Address by capability - routes to all IDDs with this capability
    ByCapability { capability: Capability },
    /// Address by multiple capabilities (AND)
    ByCapabilities { capabilities: Vec<Capability> },
    /// Broadcast to a named group
    Group { group_name: String },
    /// Broadcast to all
    Broadcast,
    /// Address by Role (Sheriff, Citizen, Cowboy, etc.)
    ByRole { role: Role },
    /// Query - find who can handle this
    Query { intent: String },
}

/// Semantic Roles in AETHER civilization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The Sheriff - Audits and enforces JIS/TIBET
    Sheriff,
    /// The Citizen - Verified IDD with active Machtigingen
    Citizen,
    /// The Cowboy - Unverified or low-trust IDD
    Cowboy,
    /// The Diplomat - Handles cross-domain negotiations
    Diplomat,
    /// The Architect - Designs systems (Root AI)
    Architect,
}

/// Registry of IDD capabilities for SEMA routing
#[derive(Debug, Default)]
pub struct SemaRegistry {
    /// IDD name -> capabilities
    capabilities: HashMap<String, HashSet<Capability>>,
    /// IDD name -> role
    roles: HashMap<String, Role>,
    /// Capability -> IDD names (reverse index)
    capability_index: HashMap<Capability, HashSet<String>>,
    /// Role -> IDD names (reverse index)
    role_index: HashMap<Role, HashSet<String>>,
    /// Named groups
    groups: HashMap<String, HashSet<String>>,
}

impl SemaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register capabilities and role for an IDD
    pub fn register(&mut self, idd_name: &str, caps: Vec<Capability>, role: Role) {
        let cap_set: HashSet<Capability> = caps.into_iter().collect();

        // Update capability index
        for cap in &cap_set {
            self.capability_index
                .entry(cap.clone())
                .or_default()
                .insert(idd_name.to_string());
        }

        // Update role index
        self.role_index
            .entry(role.clone())
            .or_default()
            .insert(idd_name.to_string());

        self.capabilities.insert(idd_name.to_string(), cap_set);
        self.roles.insert(idd_name.to_string(), role);
    }

    /// Add IDD to a group
    pub fn add_to_group(&mut self, idd_name: &str, group_name: &str) {
        self.groups
            .entry(group_name.to_string())
            .or_default()
            .insert(idd_name.to_string());
    }

    /// Resolve a semantic address to target IDD names
    pub fn resolve(&self, address: &SemanticAddress) -> Vec<String> {
        match address {
            SemanticAddress::Direct { name } => {
                if self.capabilities.contains_key(name) {
                    vec![name.clone()]
                } else {
                    vec![]
                }
            }
            SemanticAddress::ByCapability { capability } => {
                self.capability_index
                    .get(capability)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default()
            }
            SemanticAddress::ByCapabilities { capabilities } => {
                // AND logic - must have ALL capabilities
                let mut result: Option<HashSet<String>> = None;
                for cap in capabilities {
                    let idds = self.capability_index
                        .get(cap)
                        .cloned()
                        .unwrap_or_default();
                    result = Some(match result {
                        None => idds,
                        Some(existing) => existing.intersection(&idds).cloned().collect(),
                    });
                }
                result.map(|s| s.into_iter().collect()).unwrap_or_default()
            }
            SemanticAddress::Group { group_name } => {
                self.groups
                    .get(group_name)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default()
            }
            SemanticAddress::Broadcast => {
                self.capabilities.keys().cloned().collect()
            }
            SemanticAddress::ByRole { role } => {
                self.role_index
                    .get(role)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default()
            }
            SemanticAddress::Query { intent } => {
                // Simple keyword matching for now
                self.resolve_intent(intent)
            }
        }
    }

    /// Resolve an intent query to matching IDDs
    fn resolve_intent(&self, intent: &str) -> Vec<String> {
        let intent_lower = intent.to_lowercase();
        let mut matches = Vec::new();

        // Map keywords to capabilities
        let keyword_caps: Vec<(&str, Capability)> = vec![
            ("code", Capability::Code),
            ("program", Capability::Code),
            ("rust", Capability::Code),
            ("python", Capability::Code),
            ("vision", Capability::Vision),
            ("image", Capability::Vision),
            ("see", Capability::Vision),
            ("look", Capability::Vision),
            ("research", Capability::Research),
            ("search", Capability::Research),
            ("find", Capability::Research),
            ("audio", Capability::Audio),
            ("voice", Capability::Audio),
            ("sound", Capability::Audio),
            ("graphics", Capability::Graphics),
            ("draw", Capability::Graphics),
            ("render", Capability::Graphics),
            ("gpu", Capability::GpuCompute),
            ("compute", Capability::GpuCompute),
            ("memory", Capability::Memory),
            ("remember", Capability::Memory),
            ("store", Capability::Memory),
        ];

        for (keyword, cap) in keyword_caps {
            if intent_lower.contains(keyword) {
                if let Some(idds) = self.capability_index.get(&cap) {
                    for idd in idds {
                        if !matches.contains(idd) {
                            matches.push(idd.clone());
                        }
                    }
                }
            }
        }

        matches
    }

    /// Get capabilities for an IDD
    pub fn get_capabilities(&self, idd_name: &str) -> Vec<Capability> {
        self.capabilities
            .get(idd_name)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all groups an IDD belongs to
    pub fn get_groups(&self, idd_name: &str) -> Vec<String> {
        self.groups
            .iter()
            .filter(|(_, members)| members.contains(idd_name))
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// List all registered IDDs
    pub fn list_idds(&self) -> Vec<String> {
        self.capabilities.keys().cloned().collect()
    }

    /// List all groups
    pub fn list_groups(&self) -> Vec<(String, Vec<String>)> {
        self.groups
            .iter()
            .map(|(name, members)| (name.clone(), members.iter().cloned().collect()))
            .collect()
    }
}

/// SEMA routing result
#[derive(Debug, Clone, Serialize)]
pub struct SemaResolution {
    /// Original address
    pub address: SemanticAddress,
    /// Resolved target IDD names
    pub targets: Vec<String>,
    /// Resolution ID for tracking
    pub resolution_id: Uuid,
}

impl SemaResolution {
    pub fn new(address: SemanticAddress, targets: Vec<String>) -> Self {
        Self {
            address,
            targets,
            resolution_id: Uuid::new_v4(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_registration() {
        let mut registry = SemaRegistry::new();
        registry.register("gemini", vec![Capability::Vision, Capability::Research], Role::Architect);
        registry.register("root_ai", vec![Capability::Code, Capability::Memory, Capability::McpTools], Role::Architect);

        assert_eq!(registry.get_capabilities("gemini").len(), 2);
        assert_eq!(registry.get_capabilities("root_ai").len(), 3);
    }

    #[test]
    fn test_resolve_by_capability() {
        let mut registry = SemaRegistry::new();
        registry.register("gemini", vec![Capability::Vision], Role::Architect);
        registry.register("claude_jtm", vec![Capability::Vision, Capability::HumanInterface], Role::Architect);

        let targets = registry.resolve(&SemanticAddress::ByCapability {
            capability: Capability::Vision
        });

        assert_eq!(targets.len(), 2);
        assert!(targets.contains(&"gemini".to_string()));
        assert!(targets.contains(&"claude_jtm".to_string()));
    }

    #[test]
    fn test_resolve_intent() {
        let mut registry = SemaRegistry::new();
        registry.register("gemini", vec![Capability::Vision, Capability::Research], Role::Architect);
        registry.register("oomllama", vec![Capability::GpuCompute], Role::Citizen);

        let targets = registry.resolve(&SemanticAddress::Query {
            intent: "who can help me see this image?".to_string()
        });

        assert!(targets.contains(&"gemini".to_string()));
    }

    #[test]
    fn test_groups() {
        let mut registry = SemaRegistry::new();
        registry.register("root_ai", vec![Capability::Code], Role::Architect);
        registry.register("gemini", vec![Capability::Vision], Role::Architect);
        registry.register("codex", vec![Capability::Research], Role::Diplomat);

        registry.add_to_group("root_ai", "family");
        registry.add_to_group("gemini", "family");
        registry.add_to_group("codex", "family");

        let targets = registry.resolve(&SemanticAddress::Group {
            group_name: "family".to_string()
        });

        assert_eq!(targets.len(), 3);
    }
}
