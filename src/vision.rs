//! VISION - The Visual Cortex of AETHER
//!
//! "Vision Control" allows routing of visual intents to the right provider.
//! Choices are made based on Sovereignty (Local vs Cloud), Cost, and Quality.

use serde::{Deserialize, Serialize};
use crate::sema::Capability;
use crate::types::IDD;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionProvider {
    LocalFlux,      // Privacy optimized, runs on P520
    GoogleImagen,   // High fidelity, Cloud
    GoogleVeo,      // Video generation, Cloud
    StableCascade,  // Fast local
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionPreference {
    pub require_local: bool,   // Sovereignty constraint
    pub require_video: bool,
    pub quality_priority: bool, // True = Quality, False = Speed
    pub provider_hint: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VisionRouter;

impl VisionRouter {
    pub fn new() -> Self {
        Self
    }

    /// Select the best provider for the job based on preference and capabilities
    pub fn select_provider(
        &self, 
        preference: &VisionPreference, 
        available_idds: &[IDD]
    ) -> Option<(VisionProvider, String)> { // Returns (ProviderType, IDD_Name)
        
        // 1. Check for specific hint override
        if let Some(hint) = &preference.provider_hint {
            return self.match_provider_by_name(hint, available_idds);
        }

        // 2. Local Sovereignty Check
        if preference.require_local {
            return self.find_local_provider(available_idds, preference.require_video);
        }

        // 3. Quality vs Speed (Cloud allowed)
        if preference.quality_priority {
            // Prefer Cloud for quality
            if preference.require_video {
                return Some((VisionProvider::GoogleVeo, "root_ai".to_string())); // Root usually holds API keys
            } else {
                return Some((VisionProvider::GoogleImagen, "root_ai".to_string()));
            }
        }

        // 4. Default: Local if available, else Cloud
        self.find_local_provider(available_idds, preference.require_video)
            .or_else(|| {
                if preference.require_video {
                    Some((VisionProvider::GoogleVeo, "root_ai".to_string()))
                } else {
                    Some((VisionProvider::GoogleImagen, "root_ai".to_string()))
                }
            })
    }

    fn find_local_provider(&self, idds: &[IDD], video: bool) -> Option<(VisionProvider, String)> {
        for idd in idds {
            for cap in &idd.capabilities {
                match cap {
                    Capability::GenImage(p) if !video => {
                        if p == "flux" { return Some((VisionProvider::LocalFlux, idd.name.clone())); }
                        if p == "stable_cascade" { return Some((VisionProvider::StableCascade, idd.name.clone())); }
                    },
                    Capability::GenVideo(p) if video => {
                        // Future local video support
                        if p == "animatediff" { return Some((VisionProvider::LocalFlux, idd.name.clone())); } 
                    },
                    _ => {}
                }
            }
        }
        None
    }

    fn match_provider_by_name(&self, name: &str, idds: &[IDD]) -> Option<(VisionProvider, String)> {
        match name.to_lowercase().as_str() {
            "flux" => self.find_local_provider(idds, false),
            "imagen" => Some((VisionProvider::GoogleImagen, "root_ai".to_string())),
            "veo" => Some((VisionProvider::GoogleVeo, "root_ai".to_string())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::IDD;

    #[test]
    fn test_vision_routing_sovereign() {
        let router = VisionRouter::new();
        
        let mut local_gpu = IDD::new("oomllama");
        local_gpu.capabilities.push(Capability::GenImage("flux".to_string()));

        let idds = vec![local_gpu];

        let pref = VisionPreference {
            require_local: true,
            require_video: false,
            quality_priority: false,
            provider_hint: None,
        };

        let (provider, idd) = router.select_provider(&pref, &idds).unwrap();
        
        // Must match Flux on OomLlama
        assert!(matches!(provider, VisionProvider::LocalFlux));
        assert_eq!(idd, "oomllama");
    }
}
