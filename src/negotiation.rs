//! Global Negotiation Protocol
//!
//! De universele taal voor onderhandelingen tussen IDD's.
//! Agreement struct voor afspraken tussen IDD's (bijv. Chinees Legal vs American Business).

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::tibet::{TibetToken, TibetFactory};

/// Een aanbod in het onderhandelingsproces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Offer {
    pub id: Uuid,
    pub from: String,
    pub to: String,
    pub payload: serde_json::Value,
    pub context: NegotiationContext,
}

/// Contextuele informatie voor de onderhandeling (Diplomatieke Laag).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationContext {
    /// Jurisdictie (bijv. "Chinese Legal", "American Business", "EU Privacy")
    pub jurisdiction: String,
    /// Valuta of ruilmiddel
    pub currency: String,
    /// Vereist trust niveau voor acceptatie
    pub trust_required: f64,
}

/// Een definitieve afspraak tussen partijen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agreement {
    pub id: Uuid,
    pub offer_id: Uuid,
    pub parties: Vec<String>,
    pub terms: serde_json::Value,
    /// Cryptografische zegels van alle partijen.
    pub seals: Vec<String>,
    /// Koppeling aan de TIBET chain voor onweerlegbaarheid.
    pub tibet_token_id: String,
}

impl Agreement {
    /// Start een nieuwe afspraak op basis van een geaccepteerd aanbod.
    pub fn new(offer: &Offer, parties: Vec<String>, tibet_id: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            offer_id: offer.id,
            parties,
            terms: offer.payload.clone(),
            seals: Vec::new(),
            tibet_token_id: tibet_id,
        }
    }

    /// Voegt een cryptografische zegel toe aan de afspraak.
    pub fn add_seal(&mut self, party: &str, seal: &str) {
        self.seals.push(format!("{}:{}", party, seal));
    }

    /// Controleert of alle partijen de afspraak hebben bezegeld.
    pub fn is_finalized(&self) -> bool {
        self.seals.len() >= self.parties.len()
    }

    /// Genereert een TIBET audit token voor de finale afspraak.
    pub fn audit_finalized(&self, factory: &TibetFactory) -> TibetToken {
        factory.action(
            "Settlement",
            "AETHER-GLOBAL",
            serde_json::to_value(self).unwrap_or(serde_json::Value::Null),
        )
    }
}
