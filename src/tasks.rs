//! Task Authority & Status Tracking
//!
//! "Tibetten die hap!" - Every task status change is an immutable TIBET event.
//! Scalable for millions of AI/IDDs in AETHER.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::tibet::{TibetToken, TokenType, TibetFactory};
use crate::sema::Role;

/// High-level task types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskType {
    HoofdProject,
    Subtaak,
    IddSpecial(String), // Specific task for an IDD
}

/// Task statuses as defined by Heart-in-the-Loop
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskStatus {
    Pending,
    InImplementation,
    WaitingReview,
    WaitingImplement,
    Done,
    Verdeeld,
    Onnodig,
    Parked,
    Unfeasible, // "Onuitvoerbaar zonder info"
}

/// A Task in the AETHER network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub title: String,
    pub task_type: TaskType,
    pub status: TaskStatus,
    pub required_role: Option<Role>, // SEMA Role required for this task
    pub assigned_to: Option<String>, // IDD name
    pub description: String,
    pub parent_id: Option<Uuid>,
    pub tibet_chain: Vec<String>, // IDs of TIBET tokens tracking this task
}

impl Task {
    pub fn new(title: &str, task_type: TaskType, description: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            title: title.to_string(),
            task_type,
            status: TaskStatus::Pending,
            required_role: None,
            assigned_to: None,
            description: description.to_string(),
            parent_id: None,
            tibet_chain: Vec::new(),
        }
    }

    pub fn with_role(mut self, role: Role) -> Self {
        self.required_role = Some(role);
        self
    }

    /// Transition a task to a new status and generate a TIBET token
    pub fn transition(&mut self, new_status: TaskStatus, reason: &str, factory: &TibetFactory) -> TibetToken {
        let old_status = self.status.clone();
        self.status = new_status.clone();
        
        let token = factory.action(
            "TaskTransition",
            self.assigned_to.as_deref().unwrap_or("AETHER-CORE"),
            serde_json::json!({
                "task_id": self.id,
                "old_status": old_status,
                "new_status": new_status,
                "reason": reason,
            }),
        );
        
        self.tibet_chain.push(token.id.clone());

        // Memory leak fix: keep only last 100 tokens per task
        if self.tibet_chain.len() > 100 {
            self.tibet_chain.remove(0);
        }
        token
    }

    pub fn assign(&mut self, idd_name: &str, factory: &TibetFactory) -> TibetToken {
        self.assigned_to = Some(idd_name.to_string());
        self.status = TaskStatus::Verdeeld;
        
        factory.action(
            "TaskAssignment",
            "AETHER-CORE",
            serde_json::json!({
                "task_id": self.id,
                "assignee": idd_name,
            }),
        )
    }
}
