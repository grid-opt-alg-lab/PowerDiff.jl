#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building docs with SITE_BUILD=true..."
SITE_BUILD=true julia --project="$REPO_DIR/docs/" "$REPO_DIR/docs/make.jl"

TARGET="${SITE_REPO:-$HOME/web/samueltalkington.com}/_documenter/PowerModelsDiff.jl"
rm -rf "$TARGET"
mkdir -p "$(dirname "$TARGET")"
cp -r "$REPO_DIR/docs/build/" "$TARGET"

echo "Docs deployed to $TARGET"
