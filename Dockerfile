# OomLlama v1.0 — Sovereign AI Inference Engine
# Multi-stage build for minimal final image
#
# Build:  docker build -t humotica/oomllama:1.0.0-alpha .
# Run:    docker run -v /models:/models humotica/oomllama:1.0.0-alpha serve --model /models/my.oom

# Stage 1: Build Rust binary
FROM rust:latest AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy oomllama source
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY python/ ./python/

# Copy dependent crates (trust-kernel, tibet-store-mmu, jis-core)
# In Docker context these need to be available as relative paths
COPY trust-kernel/ ./trust-kernel/
COPY tibet-store-mmu/ ./tibet-store-mmu/
COPY jis-core/ ./jis-core/

# Build release binary (oomllama-server + gguf2oom)
RUN cargo build --release --bin oomllama-server --bin gguf2oom 2>/dev/null || \
    cargo build --release --bin oomllama --bin gguf2oom

# Stage 2: Minimal runtime image
FROM debian:trixie-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3t64 \
    libzstd1 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries from builder
COPY --from=builder /build/target/release/oomllama-server /usr/local/bin/oomllama 2>/dev/null || true
COPY --from=builder /build/target/release/oomllama /usr/local/bin/ 2>/dev/null || true
COPY --from=builder /build/target/release/gguf2oom /usr/local/bin/

# Create model directory
RUN mkdir -p /models

# Expose default port
EXPOSE 3000

# Default entrypoint
ENTRYPOINT ["oomllama"]
CMD ["--help"]

# Labels
LABEL org.opencontainers.image.title="OomLlama"
LABEL org.opencontainers.image.description="Sovereign AI Inference Engine — .oom quantization, Trust Kernel encryption, JIS identity, TIBET provenance"
LABEL org.opencontainers.image.vendor="HumoticaOS"
LABEL org.opencontainers.image.source="https://github.com/humotica/oomllama"
LABEL org.opencontainers.image.version="1.0.0-alpha"
