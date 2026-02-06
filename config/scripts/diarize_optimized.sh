#!/usr/bin/env bash
# optimized diarization wrapper script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

exec uv run python "$SCRIPT_DIR/diarize_batch.py" "$@"
