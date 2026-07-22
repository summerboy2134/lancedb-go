#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

# Build script for cross-platform native binaries
# Usage: ./scripts/build-native.sh [platform] [architecture]
# Example: ./scripts/build-native.sh darwin arm64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_ROOT/rust"
LIB_DIR="$PROJECT_ROOT/lib"
INCLUDE_DIR="$PROJECT_ROOT/include"

# Lance 8 protobuf generation requires protoc.  Respect an explicitly pinned
# compiler (and optional PROTOC_INCLUDE); otherwise use the first one on PATH.
if [[ -n "${PROTOC:-}" ]]; then
    if [[ ! -x "$PROTOC" ]]; then
        echo "PROTOC does not point to an executable: $PROTOC" >&2
        exit 1
    fi
elif command -v protoc >/dev/null 2>&1; then
    export PROTOC="$(command -v protoc)"
else
    echo "protoc is required to build Lance 8 (set PROTOC or install protobuf)" >&2
    exit 1
fi

# Default to current platform if not specified
PLATFORM="${1:-$(uname -s | tr '[:upper:]' '[:lower:]')}"
ARCH="${2:-$(uname -m)}"

# Normalize architecture names
case "$ARCH" in
    "x86_64"|"amd64") ARCH="amd64" ;;
    "arm64"|"aarch64") ARCH="arm64" ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

# Normalize platform names
case "$PLATFORM" in
    "darwin"|"macos") PLATFORM="darwin" ;;
    "linux") PLATFORM="linux" ;;
    "windows"|"win32"|"win64") PLATFORM="windows" ;;
    "windows-gnu") PLATFORM="windows-gnu" ;;
    "windows-msvc") PLATFORM="windows-msvc" ;;
    *) echo "Unsupported platform: $PLATFORM" >&2; exit 1 ;;
esac

TARGET_DIR="$LIB_DIR/${PLATFORM}_${ARCH}"
# Keep Cargo intermediates configurable so CI and local builds can place the
# very large Lance graph outside the source tree.
BUILD_TARGET_DIR="${CARGO_TARGET_DIR:-$RUST_DIR/target}"

echo "🏗️ Building lancedb-go native library"
echo "   Platform: $PLATFORM"
echo "   Architecture: $ARCH"
echo "   Target directory: $TARGET_DIR"
echo ""

# Ensure target directory exists
mkdir -p "$TARGET_DIR"

# Set up Rust target
RUST_TARGET=""
case "$PLATFORM-$ARCH" in
    "darwin-amd64") RUST_TARGET="x86_64-apple-darwin" ;;
    "darwin-arm64") RUST_TARGET="aarch64-apple-darwin" ;;
    "linux-amd64") RUST_TARGET="x86_64-unknown-linux-gnu" ;;
    "linux-arm64") RUST_TARGET="aarch64-unknown-linux-gnu" ;;
    "windows-amd64") RUST_TARGET="x86_64-pc-windows-gnu" ;;
    "windows-arm64") RUST_TARGET="aarch64-pc-windows-gnullvm" ;;
    "windows-gnu-amd64") RUST_TARGET="x86_64-pc-windows-gnu" ;;
    "windows-msvc-amd64") RUST_TARGET="x86_64-pc-windows-msvc" ;;
    *) echo "Unsupported target: $PLATFORM-$ARCH" >&2; exit 1 ;;
esac

HOST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
if [[ -z "$HOST_TARGET" ]]; then
    echo "Unable to determine the active Rust host target" >&2
    exit 1
fi

USE_ZIGBUILD=false
if [[ "$RUST_TARGET" != "$HOST_TARGET" ]]; then
    USE_ZIGBUILD=true
    # Cross builds use cargo-zigbuild.  Native builds intentionally do not
    # install or invoke Zig, which keeps the common path reproducible/offline.
    if ! command -v cargo-zigbuild >/dev/null 2>&1; then
        if command -v python3 >/dev/null 2>&1; then PY=python3
        elif command -v python >/dev/null 2>&1; then PY=python
        else echo "Python 3 is required for cross builds" >&2; exit 1
        fi
        echo "📦 Installing cargo-zigbuild via pip..."
        "$PY" -m pip install --break-system-packages cargo-zigbuild
        export PATH="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"
    fi
    echo "🦀 Installing Rust target: $RUST_TARGET"
    rustup target add "$RUST_TARGET"
fi

echo "🔨 Building Rust library..."
cd "$RUST_DIR"

# Set macOS deployment target to match SDK version
if [[ "$PLATFORM" == "darwin" ]]; then
    SDK_VERSION=$(xcrun --show-sdk-version 2>/dev/null || echo "15.0")
    export MACOSX_DEPLOYMENT_TARGET="$SDK_VERSION"
    echo "   MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
fi

# Build the library.  Cloud features remain the compatibility default and may
# be overridden (or disabled with an empty value) for constrained local builds.
NATIVE_FEATURES="${LANCEDB_GO_NATIVE_FEATURES-aws,gcs,azure}"
FEATURE_ARGS=()
if [[ -n "$NATIVE_FEATURES" ]]; then
    FEATURE_ARGS=(--features "$NATIVE_FEATURES")
fi
if [[ "$USE_ZIGBUILD" == true ]]; then
    CARGO_TARGET_DIR="$BUILD_TARGET_DIR" cargo zigbuild --locked --release --target "$RUST_TARGET" "${FEATURE_ARGS[@]}"
else
    CARGO_TARGET_DIR="$BUILD_TARGET_DIR" cargo build --locked --release --target "$RUST_TARGET" "${FEATURE_ARGS[@]}"
fi

# Copy library to distribution directory
echo "📦 Copying library files..."
case "$PLATFORM" in
    "darwin"|"linux")
        cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.a" "$TARGET_DIR/"
        if [ -f "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.dylib" ]; then
            cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.dylib" "$TARGET_DIR/"
        fi
        if [ -f "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.so" ]; then
            cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.so" "$TARGET_DIR/"
        fi
        ;;
    "windows")
        # GNU target (default for CGO compatibility) produces liblancedb_go.a
        if [ -f "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.a" ]; then
            cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.a" "$TARGET_DIR/"
        else
            echo "❌ No static library found for GNU target" >&2; exit 1
        fi
        if [ -f "$BUILD_TARGET_DIR/$RUST_TARGET/release/lancedb_go.dll" ]; then
            cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/lancedb_go.dll" "$TARGET_DIR/"
        fi
        ;;
    "windows-msvc")
        # MSVC target produces lancedb_go.lib
        if [ -f "$BUILD_TARGET_DIR/$RUST_TARGET/release/lancedb_go.lib" ]; then
            cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/lancedb_go.lib" "$TARGET_DIR/"
        else
            echo "❌ No static library found for MSVC target" >&2; exit 1
        fi
        if [ -f "$BUILD_TARGET_DIR/$RUST_TARGET/release/lancedb_go.dll" ]; then
            cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/lancedb_go.dll" "$TARGET_DIR/"
        fi
        ;;
    "windows-gnu")
        if [ -f "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.a" ]; then
            cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/liblancedb_go.a" "$TARGET_DIR/"
        else
            echo "❌ No static library found for GNU target" >&2; exit 1
        fi
        if [ -f "$BUILD_TARGET_DIR/$RUST_TARGET/release/lancedb_go.dll" ]; then
            cp "$BUILD_TARGET_DIR/$RUST_TARGET/release/lancedb_go.dll" "$TARGET_DIR/"
        fi
        ;;
esac

# build.rs uses the checked-in cbindgen configuration for every build.  Reuse
# that exact output instead of requiring a second global cbindgen installation.
echo "📝 Installing generated C header..."
GENERATED_HEADER=""
while IFS= read -r candidate; do
    if [[ -z "$GENERATED_HEADER" || "$candidate" -nt "$GENERATED_HEADER" ]]; then
        GENERATED_HEADER="$candidate"
    fi
done < <(find "$BUILD_TARGET_DIR/$RUST_TARGET/release/build" -path '*/out/lancedb.h' -type f -print)
if [[ -z "$GENERATED_HEADER" ]]; then
    echo "Generated lancedb.h was not found under $BUILD_TARGET_DIR" >&2
    exit 1
fi
mkdir -p "$INCLUDE_DIR"
cp "$GENERATED_HEADER" "$INCLUDE_DIR/lancedb.h"
if [[ -d "$PROJECT_ROOT/examples/include" ]]; then
    cp "$GENERATED_HEADER" "$PROJECT_ROOT/examples/include/lancedb.h"
fi

echo "✅ Build completed successfully!"
echo "   Library: $TARGET_DIR"
echo "   Header: $INCLUDE_DIR/lancedb.h"
