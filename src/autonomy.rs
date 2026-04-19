//! AUTONOMY - The Self-Driving Engine
//!
//! Allows IDDs (like Gemini, Claude, Codex) to act proactively.
//! "I see, therefore I do."
//!
//! This daemon runs inside the router and checks for triggers
//! to generate proactive Intents.

use std::sync::Arc;
use tokio::time::{self, Duration};

use crate::router::JisRouter;
use crate::intent::{Intent, Action};
use crate::memory::ConversationMemory;

pub struct AutonomyDaemon {
    router: Arc<JisRouter>, // To verify intents
    _memory: Arc<ConversationMemory>, // To see what's happening
    interval: Duration,
    active: bool,
}

impl AutonomyDaemon {
    pub fn new(router: Arc<JisRouter>, memory: Arc<ConversationMemory>) -> Self {
        Self {
            router,
            _memory: memory,
            interval: Duration::from_secs(10), // Check every 10 seconds
            active: true,
        }
    }

    /// Start the autonomy loop
    pub async fn start(self) {
        if !self.active { return; }
        
        let mut interval = time::interval(self.interval);
        
        tracing::info!("🤖 Autonomy Daemon started. The machine is awake.");

        loop {
            interval.tick().await;
            self.tick().await;
        }
    }

    /// One heartbeat of the autonomous brain
    async fn tick(&self) {
        // 1. Check for hanging conversations (Users waiting for answers)
        
        // 2. Check system health (Self-Healing)
        let stats = self.router.stats();
        if stats.pending_verifications > 10 {
            tracing::warn!("🤖 Autonomy: High traffic detected. Considering scaling action.");
        } else if stats.active_lanes == 0 {
            // Low traffic -> Dream
            self.dream().await;
        }

        // 3. Proactive Security Scan
        if let Some(root) = self.router.find_by_name("root_ai") {
            let intent = Intent::new(
                root.clone(),
                Action::Custom("health_check".to_string()),
                "Routine autonomous system scan".to_string(),
            );
            let _ = self.router.verify_intent(&intent);
        }
    }

    /// The Dream Cycle: Turning experience into wisdom
    async fn dream(&self) {
        // Only dream occasionally (1% chance per tick when idle)
        if rand::random::<f32>() > 0.01 { return; }

        tracing::info!("💤 Autonomy: Entering REM sleep. Processing memories...");
        
        // In a real implementation:
        // 1. Fetch old conversations from memory
        // 2. Send to Refinery for extraction
        // 3. Store in AIndex
        // 4. Archive original
    }
}
