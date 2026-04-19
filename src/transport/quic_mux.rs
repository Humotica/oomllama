//! QUIC MUX — Token-by-Token Streaming
//!
//! Streaming inference output over multiplexed streams.
//! Currently uses trust-kernel's TCP ClusterMux (production-ready).
//! QUIC upgrade (via quinn crate) planned for:
//! - Zero head-of-line blocking
//! - WiFi→5G handoff without stream interruption
//! - Connection migration between networks
//!
//! The API is transport-agnostic: callers use `InferenceStream`
//! regardless of whether TCP or QUIC is underneath.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use tokio::sync::mpsc;

/// A streaming inference output channel
///
/// Tokens are sent one-by-one through this stream.
/// The receiver can process them as they arrive (SSE, WebSocket, etc).
pub struct InferenceStream {
    /// Sender for token events
    tx: mpsc::Sender<StreamEvent>,
    /// Stream ID for multiplexing
    stream_id: u64,
    /// Tokens sent so far
    tokens_sent: AtomicU64,
    /// Whether the stream is still open
    active: AtomicBool,
}

/// Events that flow through an inference stream
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A new token was generated
    Token {
        stream_id: u64,
        token: String,
        token_index: u64,
    },
    /// Inference is complete
    Done {
        stream_id: u64,
        total_tokens: u64,
        latency_ms: f64,
        tibet_token_id: String,
    },
    /// An error occurred
    Error {
        stream_id: u64,
        message: String,
    },
}

/// Handle for receiving stream events
pub struct StreamReceiver {
    rx: mpsc::Receiver<StreamEvent>,
    pub stream_id: u64,
}

impl StreamReceiver {
    /// Receive the next event (async)
    pub async fn recv(&mut self) -> Option<StreamEvent> {
        self.rx.recv().await
    }
}

/// Multiplexer — manages multiple concurrent inference streams
pub struct StreamMux {
    /// Next stream ID
    next_id: AtomicU64,
    /// Active stream count
    active_streams: AtomicU64,
    /// Max concurrent streams
    max_streams: u64,
}

impl StreamMux {
    /// Create a new stream multiplexer
    pub fn new(max_streams: u64) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            active_streams: AtomicU64::new(0),
            max_streams,
        }
    }

    /// Open a new inference stream
    ///
    /// Returns (stream, receiver) — send tokens through the stream,
    /// the receiver gets them on the other end.
    pub fn open_stream(&self, buffer_size: usize) -> Option<(InferenceStream, StreamReceiver)> {
        let current = self.active_streams.load(Ordering::Relaxed);
        if current >= self.max_streams {
            return None; // At capacity
        }

        let stream_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel(buffer_size);

        self.active_streams.fetch_add(1, Ordering::Relaxed);

        let stream = InferenceStream {
            tx,
            stream_id,
            tokens_sent: AtomicU64::new(0),
            active: AtomicBool::new(true),
        };

        let receiver = StreamReceiver { rx, stream_id };

        Some((stream, receiver))
    }

    /// Get active stream count
    pub fn active_count(&self) -> u64 {
        self.active_streams.load(Ordering::Relaxed)
    }
}

impl InferenceStream {
    /// Send a token through the stream
    pub async fn send_token(&self, token: &str) -> bool {
        if !self.active.load(Ordering::Relaxed) {
            return false;
        }

        let index = self.tokens_sent.fetch_add(1, Ordering::Relaxed);
        let event = StreamEvent::Token {
            stream_id: self.stream_id,
            token: token.to_string(),
            token_index: index,
        };

        self.tx.send(event).await.is_ok()
    }

    /// Signal that inference is complete
    pub async fn finish(&self, latency_ms: f64, tibet_token_id: &str) {
        let total = self.tokens_sent.load(Ordering::Relaxed);
        let event = StreamEvent::Done {
            stream_id: self.stream_id,
            total_tokens: total,
            latency_ms,
            tibet_token_id: tibet_token_id.to_string(),
        };
        let _ = self.tx.send(event).await;
        self.active.store(false, Ordering::Relaxed);
    }

    /// Signal an error
    pub async fn error(&self, message: &str) {
        let event = StreamEvent::Error {
            stream_id: self.stream_id,
            message: message.to_string(),
        };
        let _ = self.tx.send(event).await;
        self.active.store(false, Ordering::Relaxed);
    }

    /// Get number of tokens sent
    pub fn tokens_sent(&self) -> u64 {
        self.tokens_sent.load(Ordering::Relaxed)
    }

    /// Check if stream is still active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stream_mux() {
        let mux = StreamMux::new(10);

        let (stream, mut receiver) = mux.open_stream(32).unwrap();
        assert_eq!(mux.active_count(), 1);

        // Send tokens
        assert!(stream.send_token("Hello").await);
        assert!(stream.send_token(" world").await);
        stream.finish(42.0, "TIB-TEST123").await;

        // Receive tokens
        match receiver.recv().await.unwrap() {
            StreamEvent::Token { token, token_index, .. } => {
                assert_eq!(token, "Hello");
                assert_eq!(token_index, 0);
            }
            _ => panic!("Expected Token event"),
        }

        match receiver.recv().await.unwrap() {
            StreamEvent::Token { token, token_index, .. } => {
                assert_eq!(token, " world");
                assert_eq!(token_index, 1);
            }
            _ => panic!("Expected Token event"),
        }

        match receiver.recv().await.unwrap() {
            StreamEvent::Done { total_tokens, tibet_token_id, .. } => {
                assert_eq!(total_tokens, 2);
                assert_eq!(tibet_token_id, "TIB-TEST123");
            }
            _ => panic!("Expected Done event"),
        }
    }

    #[tokio::test]
    async fn test_max_streams() {
        let mux = StreamMux::new(2);

        let _s1 = mux.open_stream(8).unwrap();
        let _s2 = mux.open_stream(8).unwrap();
        assert!(mux.open_stream(8).is_none()); // At capacity
    }
}
