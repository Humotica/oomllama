//! THE REFINERY - Project PURITY
//!
//! The Data Airlock for AETHER.
//! Filters, sanitizes, and certifies data before it enters the ecosystem.
//!
//! "From Raw Ore to Pure Knowledge."

use regex::Regex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::tibet::TibetFactory;

/// Purity level of the data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PurityLevel {
    /// Pure, verified knowledge (Trust > 0.9)
    Crystal,
    /// Cleaned data, no trackers (Trust > 0.7)
    Refined,
    /// Raw data, potentially useful but unverified (Trust > 0.4)
    Raw,
    /// Contaminated with ads/trackers/malware (Trust = 0.0)
    Toxic,
}

/// Result of the refinement process
#[derive(Debug, Clone, Serialize)]
pub struct RefineResult {
    /// Unique ID for this ingestion
    pub id: Uuid,
    /// The resulting clean text/data
    pub content: String,
    /// The determined purity level
    pub purity: PurityLevel,
    /// What was removed (stats)
    pub contaminants_removed: usize,
    /// List of detected trackers/threats
    pub detected_threats: Vec<String>,
    /// TIBET token proving this refinement happened
    pub tibet_token: Option<String>,
}

/// The Refinery Engine
pub struct Refinery {
    tracker_patterns: Vec<Regex>,
    ad_patterns: Vec<Regex>,
    script_patterns: Vec<Regex>,
    tibet: TibetFactory,
}

impl Refinery {
    pub fn new() -> Self {
        Self {
            tracker_patterns: Self::compile_tracker_patterns(),
            ad_patterns: Self::compile_ad_patterns(),
            script_patterns: Self::compile_script_patterns(),
            tibet: TibetFactory::new("refinery"),
        }
    }

    /// Ingest and purify raw text/html
    pub fn purify(&self, raw_content: &str, source: &str) -> RefineResult {
        let mut content = raw_content.to_string();
        let mut threats = Vec::new();
        let mut contaminants = 0;

        // 1. Script Stripping (Sanitization)
        for pattern in &self.script_patterns {
            if pattern.is_match(&content) {
                threats.push("malicious_script".to_string());
                let new_content = pattern.replace_all(&content, "[REDACTED_SCRIPT]");
                if new_content != content {
                    contaminants += 1;
                    content = new_content.to_string();
                }
            }
        }

        // 2. Tracker Removal
        for pattern in &self.tracker_patterns {
            if pattern.is_match(&content) {
                threats.push("tracker".to_string());
                let new_content = pattern.replace_all(&content, "");
                if new_content != content {
                    contaminants += 1;
                    content = new_content.to_string();
                }
            }
        }

        // 3. Ad Removal
        for pattern in &self.ad_patterns {
            if pattern.is_match(&content) {
                let new_content = pattern.replace_all(&content, "");
                if new_content != content {
                    contaminants += 1;
                    content = new_content.to_string();
                }
            }
        }

        // 4. Determine Purity
        let purity = if threats.contains(&"malicious_script".to_string()) {
            PurityLevel::Toxic
        } else if !threats.is_empty() {
            PurityLevel::Refined // Cleaned, but was dirty
        } else if contaminants > 0 {
            PurityLevel::Refined
        } else {
            PurityLevel::Crystal // Was clean to begin with
        };

        // 5. Mint TIBET Token (The Royal Seal)
        let token = self.tibet.action(
            "refine_data",
            "kmbit_vault",
            serde_json::json!({
                "source": source,
                "purity": purity,
                "contaminants": contaminants,
            }),
        );

        RefineResult {
            id: Uuid::new_v4(),
            content,
            purity,
            contaminants_removed: contaminants,
            detected_threats: threats,
            tibet_token: Some(token.id),
        }
    }

    fn compile_tracker_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"(?i)google-analytics\.com").unwrap(),
            Regex::new(r"(?i)facebook\.com/tr").unwrap(),
            Regex::new(r"(?i)doubleclick\.net").unwrap(),
            Regex::new(r"(?i)hotjar\.com").unwrap(),
        ]
    }

    fn compile_ad_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"(?i)<div[^>]*class=[^>]*ad-[^>]*>").unwrap(),
            Regex::new(r"(?i)sponsored by").unwrap(),
            Regex::new(r"(?i)advertisement").unwrap(),
        ]
    }

    fn compile_script_patterns() -> Vec<Regex> {
        vec![
            Regex::new(r"(?i)<script[^>]*>.*?</script>").unwrap(),
            Regex::new(r"(?i)javascript:").unwrap(),
            Regex::new(r"(?i)onload=").unwrap(),
        ]
    }
}

impl Default for Refinery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinery_cleaning() {
        let refinery = Refinery::new();
        let raw = "Hello world! <script>alert('hack')</script> This is sponsored by BigCorp. Also google-analytics.com is watching.";
        
        let result = refinery.purify(raw, "test_source");
        
        println!("Original: {}", raw);
        println!("Refined:  {}", result.content);
        println!("Purity:   {:?}", result.purity);

        assert!(result.content.contains("Hello world!"));
        assert!(!result.content.contains("<script>"));
        assert!(!result.content.contains("google-analytics.com"));
        assert_eq!(result.purity, PurityLevel::Toxic); // Script found = Toxic
        assert!(result.tibet_token.is_some());
    }
}
