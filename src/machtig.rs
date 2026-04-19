//! Authority & Delegation (Machtigingen)
//!
//! Regelt dat de 'Heart' (Jasper) machten delegeert aan 'Souls' (IDD's).
//! De 'Diplomatieke Laag' van het systeem.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::tibet::{TibetToken, TokenType, TibetFactory};

/// Machtiging structuur die autoriteit delegeert van Heart naar Soul.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Machtiging {
    pub id: Uuid,
    pub grantor: String, // Meestal "Jasper" of "Heart"
    pub grantee: String, // De IDD die de macht krijgt
    pub action: String,  // bijv. "Payment", "MedicalAccess", "Admin"
    pub constraints: Vec<Constraint>,
    pub signature: String, // Cryptografische zegel van de grantor (Hardware Anchor)
}

/// Beperkingen die op een machtiging kunnen rusten.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Maximale waarde (bijv. in Euro of Trust units)
    MaxAmount(f64),
    /// Toegestane valuta
    Currency(String),
    /// Welke modules deze machtiging mogen gebruiken
    AllowedModules(Vec<String>),
    /// Wanneer de machtiging verloopt
    ExpiresAt(chrono::DateTime<chrono::Utc>),
    /// Domein specificatie (bijv. "Legal", "Medical", "Financial")
    Domain(String),
}

impl Machtiging {
    /// Maakt een nieuwe machtiging aan van Jasper voor een specifieke IDD.
    pub fn new(grantee: &str, action: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            grantor: "Jasper".to_string(),
            grantee: grantee.to_string(),
            action: action.to_string(),
            constraints: Vec::new(),
            signature: "ASP-ANCHOR-SIG-MOCK".to_string(),
        }
    }

    /// Voegt een constraint toe aan de machtiging.
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Verifieert of een voorgenomen actie binnen de machtiging valt.
    pub fn verify(&self, action: &str, amount: Option<f64>, module: &str) -> bool {
        if self.action != action {
            return false;
        }

        for constraint in &self.constraints {
            match constraint {
                Constraint::MaxAmount(max) => {
                    if let Some(amt) = amount {
                        if amt > *max { return false; }
                    }
                }
                Constraint::AllowedModules(mods) => {
                    if !mods.contains(&module.to_string()) { return false; }
                }
                Constraint::ExpiresAt(expiry) => {
                    if chrono::Utc::now() > *expiry { return false; }
                }
                _ => {}
            }
        }
        true
    }

    /// Genereert een TIBET audit token voor deze machtiging.
    pub fn audit_token(&self, factory: &TibetFactory) -> TibetToken {
        factory.action(
            "Delegation",
            &self.grantee,
            serde_json::to_value(self).unwrap_or(serde_json::Value::Null),
        )
    }
}
