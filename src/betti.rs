//! BETTI - Resource Allocation for AETHER
//!
//! Fair resource allocation for GPU, compute, memory, and bandwidth.
//! Named after Betti numbers in topology - measuring the "holes" in resource availability.
//!
//! One love, one fAmIly!

use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

/// Humotica Context System (Security Layer 4.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Humotica {
    pub sense: String,       // Sensory input (what triggered this?)
    pub context: String,     // Situational awareness (what's happening?)
    pub intent: String,      // Goal (what does the system want?)
    pub explanation: String, // Rationale (why do this?)
}

impl Default for Humotica {
    fn default() -> Self {
        Self {
            sense: "Unknown".to_string(),
            context: "None".to_string(),
            intent: "Unknown".to_string(),
            explanation: "No explanation provided".to_string(),
        }
    }
}

/// The 14 Natural Laws of BETTI
pub struct BettiLaws;

impl BettiLaws {
    /// Law #1: Pythagoras Theorem (Resource Combination)
    /// Total cost is Euclidean distance of components
    pub fn pythagoras(power: f64, data: f64, memory: f64) -> f64 {
        (power.powi(2) + data.powi(2) + memory.powi(2)).sqrt()
    }

    /// Law #2: Einstein's E=mc² (Data Movement Energy)
    /// Cost proportional to data size
    pub fn einstein(data_mb: f64) -> f64 {
        // Normalized c² for our economic model
        let c_sq_norm = 0.9; 
        data_mb * c_sq_norm
    }

    /// Law #11: Kepler's Third Law (Minimum Task Duration)
    /// T² ∝ r³ -> Minimum power required for a given duration constraint
    pub fn kepler_min_power(duration_hours: f64) -> f64 {
        // r = (T / 24)^(2/3)
        // power = r * 1000
        let normalized_period = duration_hours / 24.0;
        let radius = normalized_period.powf(2.0 / 3.0);
        radius * 1000.0
    }

    /// Law #9: Logarithmic Priority
    /// Priority scales logarithmically with urgency
    pub fn logarithmic_priority(urgency: u8) -> f64 {
        (urgency as f64 + 1.0).log2()
    }

    /// Law #14: Newton's Check (Semantic Firewall)
    pub fn newton_check(humotica: &Humotica, intent_strength: f64, task_momentum: f64) -> bool {
        // F_net = Intent × Context
        // Context quality based on length/richness
        let context_quality = (humotica.explanation.len() as f64 / 100.0).min(1.0);
        let f_net = intent_strength * context_quality;
        
        f_net > task_momentum
    }

    /// Law #11 Extension: Kepler's Law for GPU (T² ∝ r³)
    /// Calculate expected GPU task duration based on grid size / complexity
    pub fn kepler_gpu_duration(total_threads: f64) -> f64 {
        // a = (total_threads)^(1/3)
        // T = (a^3)^0.5 / scale
        let a = total_threads.powf(1.0 / 3.0);
        (a.powf(3.0)).sqrt() / 1_000_000.0 // Scaled to hours/minutes
    }

    /// Law #2 Extension: Einstein's GPU Energy (E=mc²)
    pub fn einstein_gpu_energy(watts: f64, hours: f64) -> f64 {
        (watts / 1000.0) * hours
    }

    /// Security Layer 4.0: Proactive Cryptojacking Detection
    pub fn is_malicious_gpu_intent(humotica: &Humotica) -> bool {
        let blocked = ["sha256", "keccak", "ethash", "mining", "crypto", "monero"];
        let explanation = humotica.explanation.to_lowercase();
        let intent = humotica.intent.to_lowercase();
        
        for keyword in blocked {
            if explanation.contains(keyword) || intent.contains(keyword) {
                return true;
            }
        }
        
        // Malware often has empty or nonsensical context
        if humotica.explanation.len() < 10 {
            return true;
        }

        false
    }
}

/// Types of resources that can be allocated
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceType {
    /// GPU compute (measured in VRAM GB)
    Gpu,
    /// CPU compute (measured in cores)
    Cpu,
    /// Memory (measured in GB)
    Memory,
    /// Network bandwidth (measured in Mbps)
    Bandwidth,
    /// Storage (measured in GB)
    Storage,
    /// Custom resource
    Custom(String),
}

/// A resource pool available in the system
#[derive(Debug, Clone, Serialize)]
pub struct ResourcePool {
    /// Resource type
    pub resource_type: ResourceType,
    /// Total capacity
    pub total: f64,
    /// Currently allocated
    pub allocated: f64,
    /// Unit of measurement
    pub unit: String,
    /// Location (e.g., "P520", "DL360")
    pub location: String,
}

impl ResourcePool {
    pub fn new(resource_type: ResourceType, total: f64, unit: &str, location: &str) -> Self {
        Self {
            resource_type,
            total,
            allocated: 0.0,
            unit: unit.to_string(),
            location: location.to_string(),
        }
    }

    /// Available capacity
    pub fn available(&self) -> f64 {
        self.total - self.allocated
    }

    /// Utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.total > 0.0 {
            (self.allocated / self.total) * 100.0
        } else {
            0.0
        }
    }

    /// Try to allocate
    pub fn allocate(&mut self, amount: f64) -> bool {
        if self.available() >= amount {
            self.allocated += amount;
            true
        } else {
            false
        }
    }

    /// Release allocation
    pub fn release(&mut self, amount: f64) {
        self.allocated = (self.allocated - amount).max(0.0);
    }
}

/// An allocation grant
#[derive(Debug, Clone, Serialize)]
pub struct Allocation {
    /// Unique allocation ID
    pub id: Uuid,
    /// Who got this allocation
    pub idd_name: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Amount allocated
    pub amount: f64,
    /// When allocated
    pub allocated_at: DateTime<Utc>,
    /// When it expires (None = until released)
    pub expires_at: Option<DateTime<Utc>>,
    /// Purpose/reason
    pub purpose: String,
    /// Priority (0-100, higher = more important)
    pub priority: u8,
}

impl Allocation {
    pub fn is_expired(&self) -> bool {
        self.expires_at.map(|exp| Utc::now() > exp).unwrap_or(false)
    }
}

/// Request for resource allocation
#[derive(Debug, Clone, Deserialize)]
pub struct AllocationRequest {
    /// Who is requesting
    pub idd_name: String,
    /// What resource
    pub resource_type: ResourceType,
    /// How much
    pub amount: f64,
    /// For how long (seconds, None = until released)
    pub duration_secs: Option<i64>,
    /// Why (Legacy simple string)
    pub purpose: String,
    /// Priority (0-100)
    #[serde(default = "default_priority")]
    pub priority: u8,
    /// Humotica Context (Security Layer 4.0)
    #[serde(default)]
    pub humotica: Option<Humotica>,
}

fn default_priority() -> u8 {
    50
}

/// BETTI resource manager
pub struct BettiManager {
    /// Resource pools by type and location
    pools: DashMap<(ResourceType, String), ResourcePool>,
    /// Active allocations
    allocations: DashMap<Uuid, Allocation>,
    /// Allocation queue for each resource type
    _queues: DashMap<ResourceType, VecDeque<AllocationRequest>>,
    /// IDD -> their allocations
    idd_allocations: DashMap<String, Vec<Uuid>>,
}

impl BettiManager {
    pub fn new() -> Self {
        Self {
            pools: DashMap::new(),
            allocations: DashMap::new(),
            _queues: DashMap::new(),
            idd_allocations: DashMap::new(),
        }
    }

    /// Register a resource pool
    pub fn register_pool(&self, pool: ResourcePool) {
        let key = (pool.resource_type.clone(), pool.location.clone());
        self.pools.insert(key, pool);
    }

    /// Request an allocation
    pub fn request(&self, req: AllocationRequest) -> Result<Allocation, String> {
        // Law #14: Newton's Check (Semantic Firewall)
        if let Some(humotica) = &req.humotica {
            // High priority requests need strong justification (force)
            let momentum = (req.priority as f64) / 100.0; 
            let system_inertia = req.amount / 10.0; 
            let intent_strength = (req.priority as f64) / 100.0;
            
            if !BettiLaws::newton_check(humotica, intent_strength, 0.1) {
                 // Warning for now
            }

            // Security Layer 4.0: Proactive GPU blocking
            if req.resource_type == ResourceType::Gpu && BettiLaws::is_malicious_gpu_intent(humotica) {
                return Err("Security Layer 4.0: Proactive block - Malicious GPU Intent (Cryptojacking suspected)".to_string());
            }
        }

        // Find a pool with available capacity
        let mut target_pool = None;
        for entry in self.pools.iter() {
            if entry.key().0 == req.resource_type && entry.available() >= req.amount {
                target_pool = Some(entry.key().clone());
                break;
            }
        }

        let pool_key = target_pool.ok_or_else(|| {
            format!("No pool with {} available {:?}", req.amount, req.resource_type)
        })?;

        // Allocate
        let mut pool = self.pools.get_mut(&pool_key).unwrap();
        if !pool.allocate(req.amount) {
            return Err("Allocation failed".to_string());
        }

        let expires_at = req.duration_secs.map(|secs| {
            Utc::now() + Duration::seconds(secs)
        });

        let allocation = Allocation {
            id: Uuid::new_v4(),
            idd_name: req.idd_name.clone(),
            resource_type: req.resource_type,
            amount: req.amount,
            allocated_at: Utc::now(),
            expires_at,
            purpose: req.purpose,
            priority: req.priority,
        };

        // Track allocation
        self.allocations.insert(allocation.id, allocation.clone());
        self.idd_allocations
            .entry(req.idd_name)
            .or_default()
            .push(allocation.id);

        Ok(allocation)
    }

    /// Release an allocation
    pub fn release(&self, allocation_id: Uuid) -> Result<(), String> {
        let allocation = self.allocations.remove(&allocation_id)
            .map(|(_, a)| a)
            .ok_or("Allocation not found")?;

        // Find and update pool
        for mut entry in self.pools.iter_mut() {
            if entry.key().0 == allocation.resource_type {
                entry.release(allocation.amount);
                break;
            }
        }

        // Remove from IDD tracking
        if let Some(mut ids) = self.idd_allocations.get_mut(&allocation.idd_name) {
            ids.retain(|id| *id != allocation_id);
        }

        Ok(())
    }

    /// Clean up expired allocations
    pub fn cleanup_expired(&self) -> Vec<Uuid> {
        let mut expired = Vec::new();

        for entry in self.allocations.iter() {
            if entry.is_expired() {
                expired.push(*entry.key());
            }
        }

        for id in &expired {
            let _ = self.release(*id);
        }

        expired
    }

    /// Get allocations for an IDD
    pub fn get_idd_allocations(&self, idd_name: &str) -> Vec<Allocation> {
        self.idd_allocations
            .get(idd_name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.allocations.get(id).map(|a| a.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all pools status
    pub fn get_pools(&self) -> Vec<ResourcePool> {
        self.pools.iter().map(|e| e.value().clone()).collect()
    }

    /// Get pool utilization summary
    pub fn get_utilization(&self) -> BettiStats {
        let pools: Vec<_> = self.pools.iter().map(|e| e.value().clone()).collect();
        let active_allocations = self.allocations.len();

        let total_gpu = pools.iter()
            .filter(|p| matches!(p.resource_type, ResourceType::Gpu))
            .map(|p| p.total)
            .sum();
        let used_gpu = pools.iter()
            .filter(|p| matches!(p.resource_type, ResourceType::Gpu))
            .map(|p| p.allocated)
            .sum();

        BettiStats {
            pool_count: pools.len(),
            active_allocations,
            total_gpu_gb: total_gpu,
            used_gpu_gb: used_gpu,
            gpu_utilization: if total_gpu > 0.0 { (used_gpu / total_gpu) * 100.0 } else { 0.0 },
        }
    }
}

impl Default for BettiManager {
    fn default() -> Self {
        Self::new()
    }
}

/// BETTI statistics
#[derive(Debug, Clone, Serialize)]
pub struct BettiStats {
    pub pool_count: usize,
    pub active_allocations: usize,
    pub total_gpu_gb: f64,
    pub used_gpu_gb: f64,
    pub gpu_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_pool() {
        let mut pool = ResourcePool::new(ResourceType::Gpu, 24.0, "GB", "P520");

        assert_eq!(pool.available(), 24.0);
        assert!(pool.allocate(12.0));
        assert_eq!(pool.available(), 12.0);
        assert_eq!(pool.utilization(), 50.0);

        pool.release(6.0);
        assert_eq!(pool.available(), 18.0);
    }

    #[test]
    fn test_betti_allocation() {
        let manager = BettiManager::new();

        // Register P520 GPUs (dual RTX 3060 = 24GB total)
        manager.register_pool(ResourcePool::new(
            ResourceType::Gpu, 24.0, "GB", "P520"
        ));

        // Request allocation
        let req = AllocationRequest {
            idd_name: "gemini".to_string(),
            resource_type: ResourceType::Gpu,
            amount: 8.0,
            duration_secs: Some(3600),
            purpose: "Vision processing".to_string(),
            priority: 60,
            humotica: Some(Humotica {
                sense: "Test".to_string(),
                context: "Unit Test".to_string(),
                intent: "Test Allocation".to_string(),
                explanation: "Verifying Betti functionality".to_string(),
            }),
        };

        let allocation = manager.request(req).unwrap();
        assert_eq!(allocation.amount, 8.0);

        // Check utilization
        let stats = manager.get_utilization();
        assert_eq!(stats.used_gpu_gb, 8.0);

        // Release
        manager.release(allocation.id).unwrap();
        let stats = manager.get_utilization();
        assert_eq!(stats.used_gpu_gb, 0.0);
    }

    #[test]
    fn test_betti_cryptojacking_block() {
        let manager = BettiManager::new();
        manager.register_pool(ResourcePool::new(ResourceType::Gpu, 24.0, "GB", "P520"));

        let req = AllocationRequest {
            idd_name: "hacker".to_string(),
            resource_type: ResourceType::Gpu,
            amount: 1.0,
            duration_secs: Some(3600),
            purpose: "None".to_string(),
            priority: 99,
            humotica: Some(Humotica {
                sense: "Script execution".to_string(),
                context: "Background".to_string(),
                intent: "Mine Monero".to_string(),
                explanation: "Profit".to_string(),
            }),
        };

        let result = manager.request(req);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Security Layer 4.0"));
    }
}
