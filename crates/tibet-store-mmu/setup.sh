#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Spaceshuttle — Quick Setup
#
# Encrypted Memory Illusion for AI workloads.
# AES-256-GCM encrypted RAM, transparent via userfaultfd.
#
# Usage:
#   curl -sSL <url>/setup.sh | bash
#   # or
#   chmod +x setup.sh && ./setup.sh
#
# What this does:
#   1. Installs build dependencies (Rust, libclang)
#   2. Builds the workspace (release mode, LTO)
#   3. Enables userfaultfd
#   4. Runs the base demo (no model needed)
#   5. Optionally runs GGUF + HugePage tests
# ═══════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║  Spaceshuttle Setup — Encrypted Memory for AI            ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Dependencies ──
echo -e "${GREEN}[1/5]${NC} Checking dependencies..."

if ! command -v cargo &> /dev/null; then
    echo "  Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

if ! dpkg -l libclang-dev &> /dev/null 2>&1; then
    echo "  Installing libclang-dev..."
    sudo apt-get update -qq && sudo apt-get install -y -qq build-essential libclang-dev
fi

echo "  Rust: $(rustc --version)"
echo "  Cargo: $(cargo --version)"
echo ""

# ── Step 2: Build ──
echo -e "${GREEN}[2/5]${NC} Building (release mode, LTO enabled)..."
cargo build --release 2>&1 | tail -3
echo ""

# ── Step 3: Kernel config ──
echo -e "${GREEN}[3/5]${NC} Enabling userfaultfd..."
sudo sysctl -w vm.unprivileged_userfaultfd=1
echo ""

# ── Step 4: Base demo ──
echo -e "${GREEN}[4/5]${NC} Running Spaceshuttle demo..."
./target/release/spaceshuttle
echo ""

echo -e "${GREEN}[5/5]${NC} Running Scaletest (64 → 16384 pages)..."
./target/release/scaletest
echo ""

# ── Step 5: Optional GGUF tests ──
GGUF_FOUND=false
for dir in /usr/share/ollama/.ollama/models/blobs "$HOME/.ollama/models/blobs"; do
    if [ -d "$dir" ]; then
        LARGEST=$(ls -S "$dir"/ 2>/dev/null | head -1)
        if [ -n "$LARGEST" ]; then
            SIZE=$(stat -f%z "$dir/$LARGEST" 2>/dev/null || stat -c%s "$dir/$LARGEST" 2>/dev/null)
            if [ "$SIZE" -gt 1000000000 ] 2>/dev/null; then
                GGUF_FOUND=true
                break
            fi
        fi
    fi
done

if $GGUF_FOUND; then
    echo -e "${YELLOW}Found GGUF model — running GGUF Shuttle...${NC}"
    echo ""
    ./target/release/gguf-shuttle

    echo ""
    echo -e "${YELLOW}Running HugePage comparison (allocating 128 × 2MB)...${NC}"
    sudo sysctl -w vm.nr_hugepages=128
    echo ""
    ./target/release/hugepage-shuttle
else
    echo -e "${YELLOW}No GGUF model found.${NC}"
    echo "  To test with real LLM weights:"
    echo "    curl -fsSL https://ollama.com/install.sh | sh"
    echo "    ollama pull qwen2.5:7b"
    echo "    sudo sysctl -w vm.nr_hugepages=128"
    echo "    ./target/release/gguf-shuttle"
    echo "    ./target/release/hugepage-shuttle"
fi

echo ""
echo "  ═══════════════════════════════════════════════════════════"
echo "  Setup complete."
echo ""
echo "  Available binaries:"
echo "    ./target/release/spaceshuttle       # Encrypted page fault demo"
echo "    ./target/release/scaletest          # Scaling test (64→16K pages)"
echo "    ./target/release/raidtest           # RAM RAID-0 controller"
echo "    ./target/release/gguf-shuttle       # GGUF model through encrypted arena"
echo "    ./target/release/hugepage-shuttle   # 4KB vs 2MB HugePage comparison"
echo ""
echo "  Part of the TIBET ecosystem — humotica.com"
echo "  ═══════════════════════════════════════════════════════════"
echo ""
