#!/bin/bash
# OomLlama Docker Build Script
# Prepares build context with dependent crates and builds the image
#
# Usage: ./docker-build.sh [tag]
# Example: ./docker-build.sh humotica/oomllama:1.0.0-alpha

set -e

TAG="${1:-humotica/oomllama:1.0.0-alpha}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR=$(mktemp -d)

echo "=== OomLlama Docker Build ==="
echo "Tag: $TAG"
echo "Build dir: $BUILD_DIR"

# Copy oomllama source
cp -r "$SCRIPT_DIR/Cargo.toml" "$SCRIPT_DIR/Cargo.lock" "$BUILD_DIR/"
cp -r "$SCRIPT_DIR/src" "$BUILD_DIR/src"
cp -r "$SCRIPT_DIR/python" "$BUILD_DIR/python"
cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/"

# Copy dependent crates (-L follows symlinks)
echo "Copying trust-kernel..."
cp -rL "$SCRIPT_DIR/../trust-kernel" "$BUILD_DIR/trust-kernel"
echo "Copying tibet-store-mmu..."
cp -rL "$SCRIPT_DIR/../tibet-store-mmu" "$BUILD_DIR/tibet-store-mmu"
echo "Copying jis-core..."
cp -rL "$SCRIPT_DIR/../jis-core" "$BUILD_DIR/jis-core"

# Fix paths in Cargo.toml for Docker context
sed -i 's|path = "\.\./trust-kernel"|path = "trust-kernel"|g' "$BUILD_DIR/Cargo.toml"
sed -i 's|path = "\.\./tibet-store-mmu"|path = "tibet-store-mmu"|g' "$BUILD_DIR/Cargo.toml"
sed -i 's|path = "\.\./jis-core"|path = "jis-core"|g' "$BUILD_DIR/Cargo.toml"

# Also fix tibet-store-mmu's dep on trust-kernel
if [ -f "$BUILD_DIR/tibet-store-mmu/Cargo.toml" ]; then
    sed -i 's|path = "\.\./trust-kernel"|path = "../trust-kernel"|g' "$BUILD_DIR/tibet-store-mmu/Cargo.toml"
fi

echo "Building Docker image..."
docker build -t "$TAG" "$BUILD_DIR"

echo "Cleaning up..."
rm -rf "$BUILD_DIR"

echo ""
echo "=== Done! ==="
echo "Image: $TAG"
echo "Run: docker run -v /models:/models $TAG serve --model /models/my.oom"
