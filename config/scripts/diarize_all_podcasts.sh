#!/usr/bin/env bash
#
# automatically process all podcast subdirectories in transcripts-dir.
# discovers subdirectories and runs diarization for each one.

set -euo pipefail

# default values.
TRANSCRIPTS_ROOT=""
AUDIO_BASE_DIR=""
OUTPUT_DIR=""
HF_TOKEN="${HF_TOKEN:-}"
WORKERS=4
COMPILE_MODE="reduce-overhead"
DRY_RUN=false
PROCESS_ALL=false

# get script directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# usage information.
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Automatically process podcast directories (single podcast or multiple podcasts).

Required Options:
  --transcripts-root DIR    Directory containing JSON transcripts
                            Can be a single podcast dir or root with subdirs
  --audio-base-dir DIR      Directory containing audio files (matching structure)
  --output-dir DIR          Output directory for RTTM files

Optional:
  --hf-token TOKEN          HuggingFace API token (or set HF_TOKEN env var)
  --workers N               Number of parallel workers (default: 4)
  --compile-mode MODE       torch.compile mode (default: reduce-overhead)
                            Options: default, reduce-overhead, max-autotune
                            Note: max-autotune requires H100/A100 GPUs
  --dry-run                 Show what would be processed without running
  -h, --help                Show this help message

Modes:
  Single Podcast Mode:
    When --transcripts-root contains JSON files directly
    Example structure:
      transcripts/the_show/    downloads/the_show/
        episode.json             episode.mp3

  Multi-Podcast Mode:
    When --transcripts-root contains subdirectories with JSON files
    Example structure:
      transcripts/             downloads/
        podcast_a/               podcast_a/
          episode.json             episode.mp3
        podcast_b/               podcast_b/
          episode.json             episode.mp3

Examples:
  # Process a single podcast
  $(basename "$0") \\
    --transcripts-root /path/to/transcripts/the_megyn_kelly_show \\
    --audio-base-dir /path/to/downloads/the_megyn_kelly_show \\
    --output-dir /path/to/diarizations/the_megyn_kelly_show \\
    --hf-token YOUR_TOKEN \\
    --workers 5

  # Process all podcasts from root
  $(basename "$0") \\
    --transcripts-root /path/to/transcripts \\
    --audio-base-dir /path/to/downloads \\
    --output-dir /path/to/diarizations \\
    --hf-token YOUR_TOKEN

  # Dry run to preview
  $(basename "$0") \\
    --transcripts-root /path/to/transcripts \\
    --audio-base-dir /path/to/downloads \\
    --output-dir /path/to/diarizations \\
    --dry-run
EOF
    exit 1
}

# parse arguments.
while [[ $# -gt 0 ]]; do
    case $1 in
        --transcripts-root)
            TRANSCRIPTS_ROOT="$2"
            shift 2
            ;;
        --audio-base-dir)
            AUDIO_BASE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --compile-mode)
            COMPILE_MODE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --process-all)
            PROCESS_ALL=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

# validate required arguments.
if [[ -z "$TRANSCRIPTS_ROOT" ]] || [[ -z "$AUDIO_BASE_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments"
    usage
fi

if [[ ! -d "$TRANSCRIPTS_ROOT" ]]; then
    echo "Error: Transcripts root directory not found: $TRANSCRIPTS_ROOT"
    exit 1
fi

if [[ ! -d "$AUDIO_BASE_DIR" ]]; then
    echo "Error: Audio base directory not found: $AUDIO_BASE_DIR"
    exit 1
fi

# check HF token for non-dry-run.
if [[ "$DRY_RUN" == "false" ]] && [[ -z "$HF_TOKEN" ]]; then
    echo "Error: HuggingFace token required (set HF_TOKEN env var or use --hf-token)"
    exit 1
fi

# discover podcast subdirectories or check if this is a single podcast directory.
echo "🔍 Discovering podcasts in: $TRANSCRIPTS_ROOT"
echo ""

# debug: show what's actually in the directory.
echo "📋 Debug: Checking directory contents..."
if [[ ! -d "$TRANSCRIPTS_ROOT" ]]; then
    echo "❌ ERROR: Directory does not exist: $TRANSCRIPTS_ROOT"
    echo ""
    echo "Hint: Use an absolute path or relative path from current directory:"
    echo "  Current directory: $(pwd)"
    echo "  Example absolute: /full/path/to/transcripts/the_show"
    echo "  Example relative: outputs/transcripts/the_show"
    exit 1
fi

json_count_direct=$(find "$TRANSCRIPTS_ROOT" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
subdir_count=$(find "$TRANSCRIPTS_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')

echo "  • JSON files in directory: $json_count_direct"
echo "  • Subdirectories found: $subdir_count"
echo ""

# check if the provided directory itself contains JSON files (single podcast mode).
if [[ "$json_count_direct" -gt 0 ]]; then
    echo "📁 Single podcast mode: Directory contains JSON files directly"
    echo "  • $(basename "$TRANSCRIPTS_ROOT") ($json_count_direct transcripts)"
    echo ""
    SINGLE_PODCAST_MODE=true
else
    # multi-podcast mode: look for subdirectories with JSON files.
    echo "📁 Multi-podcast mode: Looking for subdirectories with JSON files"
    SINGLE_PODCAST_MODE=false

    podcast_dirs=()
    for dir in "$TRANSCRIPTS_ROOT"/*; do
        if [[ -d "$dir" ]]; then
            # check if directory contains JSON files.
            subdir_json_count=$(find "$dir" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$subdir_json_count" -gt 0 ]]; then
                podcast_name=$(basename "$dir")
                podcast_dirs+=("$podcast_name")
            fi
        fi
    done

    if [[ ${#podcast_dirs[@]} -eq 0 ]]; then
        echo "❌ No podcast subdirectories found with JSON files"
        echo ""
        echo "Debug information:"
        echo "  • Looking in: $TRANSCRIPTS_ROOT"
        echo "  • Found $subdir_count subdirectories"
        echo "  • None contain JSON files"
        echo ""
        echo "Possible issues:"
        echo "  1. Wrong path - check if the directory path is correct"
        echo "  2. No JSON files - directory might be empty or contain other file types"
        echo ""
        echo "Directory listing:"
        ls -la "$TRANSCRIPTS_ROOT" | head -10
        exit 1
    fi

    echo "📁 Found ${#podcast_dirs[@]} podcast subdirectories:"
    for podcast in "${podcast_dirs[@]}"; do
        json_count=$(find "$TRANSCRIPTS_ROOT/$podcast" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "  • $podcast ($json_count transcripts)"
    done
    echo ""
fi

# dry run mode.
if [[ "$DRY_RUN" == "true" ]]; then
    echo "🔍 DRY RUN MODE - Command that would be executed:"
    echo ""
    echo "$SCRIPT_DIR/diarize_optimized.sh \\"
    echo "  --transcripts-dir \"$TRANSCRIPTS_ROOT\" \\"
    echo "  --audio-base-dir \"$AUDIO_BASE_DIR\" \\"
    echo "  --output-dir \"$OUTPUT_DIR\" \\"
    echo "  --hf-token \"<token>\" \\"
    echo "  --workers $WORKERS \\"
    echo "  --compile-mode $COMPILE_MODE"
    echo ""
    if [[ "$SINGLE_PODCAST_MODE" == "true" ]]; then
        echo "This will process the single podcast directory."
    else
        echo "This will process all ${#podcast_dirs[@]} podcast subdirectories from the root."
    fi
    exit 0
fi

# process from the provided directory.
if [[ "$SINGLE_PODCAST_MODE" == "true" ]]; then
    echo "🚀 Processing single podcast: $(basename "$TRANSCRIPTS_ROOT")"
    echo ""
else
    echo "🚀 Processing all podcasts from root directory..."
    echo "   (This maintains correct relative paths for audio file resolution)"
    echo ""
fi

start_time=$(date +%s)

"$SCRIPT_DIR/diarize_optimized.sh" \
    --transcripts-dir "$TRANSCRIPTS_ROOT" \
    --audio-base-dir "$AUDIO_BASE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --hf-token "$HF_TOKEN" \
    --workers "$WORKERS" \
    --compile-mode "$COMPILE_MODE"

exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 FINAL SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [[ "$SINGLE_PODCAST_MODE" == "true" ]]; then
    echo "Podcast: $(basename "$TRANSCRIPTS_ROOT")"
else
    echo "Total podcasts discovered: ${#podcast_dirs[@]}"
fi
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo ""

if [[ $exit_code -eq 0 ]]; then
    echo "✅ Processing complete!"
else
    echo "⚠️  Processing completed with some errors (see above)"
fi

exit $exit_code
