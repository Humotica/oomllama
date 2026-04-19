//! Error types for JIS Router

use thiserror::Error;
use uuid::Uuid;

use crate::intent::VerificationFailure;

/// JIS Router errors
#[derive(Error, Debug)]
pub enum JisError {
    #[error("Intent verification failed: {0:?}")]
    VerificationFailed(Vec<VerificationFailure>),

    #[error("IDD not found: {0}")]
    IddNotFound(Uuid),

    #[error("Lane not found: {0}")]
    LaneNotFound(Uuid),

    #[error("Insufficient trust: required {required}, got {actual}")]
    InsufficientTrust { required: f64, actual: f64 },

    #[error("Outside timeslot window")]
    OutsideTimeslot,

    #[error("Intent expired")]
    IntentExpired,

    #[error("Missing capability: {0}")]
    MissingCapability(String),

    #[error("Lane already exists between {from} and {to}")]
    LaneAlreadyExists { from: Uuid, to: Uuid },

    #[error("Rate limited")]
    RateLimited,

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("TIBET token error: {0}")]
    TibetError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for JIS operations
pub type JisResult<T> = Result<T, JisError>;
