#!/usr/bin/env bash
# One-time setup on a new machine: point Clearpath at this repo's robot.yaml.
set -e
REPO_YAML="$(cd "$(dirname "$0")/.." && pwd)/config/robot.yaml"
mkdir -p "$HOME/clearpath"
ln -sf "$REPO_YAML" "$HOME/clearpath/robot.yaml"
echo "Linked $HOME/clearpath/robot.yaml -> $REPO_YAML"
