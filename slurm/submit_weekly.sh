#!/usr/bin/env bash
# submit_weekly.sh — weekly podcast sync coordinator.
#
# Run from the project root on an RC login node. Downloads new episodes for
# all channels in config/channels.yaml, then submits SLURM GPU jobs for
# transcription and diarization (each diarization job depends on its
# transcription job finishing successfully).
#
# Usage:
#   bash slurm/submit_weekly.sh [--dry-run]
#
# Cron entry (every Monday at 2:00am Mountain) — cron only runs sbatch, not the script directly:
#   0 2 * * 1 cd /projects/alle5715/pypodcast && sbatch slurm/submit_weekly.sh
#
# Prerequisites on RC:
#   - HF_TOKEN env var set (add to ~/.bashrc on RC):
#       export HF_TOKEN="your_huggingface_token"
#   - Project cloned to /projects/alle5715/pypodcast
#   - uv available (module load uv/0.8.15)


#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00

#SBATCH --job-name=weekly_sync
#SBATCH --output=logs/weekly_sync_%j.log
#SBATCH --chdir=/projects/alle5715/pypodcast
#SBATCH --mail-user=alle5715@colorado.edu
#SBATCH --mail-type=FAIL

set -euo pipefail  # exit on error (-e), undefined vars (-u), pipe failures (-o pipefail)

# BASH_SOURCE[0] resolves to /var/spool/slurmd/logs when run via sbatch on CURC,
# so derive PROJECT_ROOT from the hardcoded path instead.
PROJECT_ROOT=/projects/alle5715/pypodcast
LOGS_DIR="$PROJECT_ROOT/logs"
CONFIG_FILE="$PROJECT_ROOT/config/channels.yaml"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$LOGS_DIR"

echo "============================================================"
echo "Weekly podcast sync — $(date)"
echo "============================================================"

# load modules needed for download and YAML parsing.
module load uv/0.8.15 2>/dev/null || true
module load ffmpeg/4.4 2>/dev/null || true

# lookback 8 days so a late run never misses an episode.
START_DATE=$(date -d '8 days ago' '+%Y-%m-%d')
echo "Downloading episodes published since: $START_DATE"
echo ""

# read config using python (available via uv).
cd "$PROJECT_ROOT"
AUDIO_ROOT=$(uv run -q python3 -c "
import yaml
with open('config/channels.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg['base_paths']['audio_root'])
")
OUTPUTS_ROOT=$(uv run -q python3 -c "
import yaml
with open('config/channels.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg['base_paths']['outputs_root'])
")

# emit one line per channel: slug|feed_url|language
CHANNELS=$(uv run -q python3 -c "
import yaml
with open('config/channels.yaml') as f:
    cfg = yaml.safe_load(f)
for ch in cfg['channels']:
    print(f\"{ch['slug']}|{ch['feed_url']}|{ch.get('language','en')}\")
")

while IFS='|' read -r SLUG FEED_URL LANGUAGE; do  # split each "slug|feed_url|language" line into three variables
    echo "──────────────────────────────────────"
    echo "Channel: $SLUG ($FEED_URL)"

    AUDIO_DIR="$AUDIO_ROOT/$SLUG/audio"
    TRANSCRIPTS_DIR="$OUTPUTS_ROOT/transcripts/$SLUG"
    DIARIZATIONS_DIR="$OUTPUTS_ROOT/diarizations/$SLUG"

    CHANNEL_LOGS_DIR="$LOGS_DIR/$SLUG"
    mkdir -p "$AUDIO_DIR" "$TRANSCRIPTS_DIR" "$DIARIZATIONS_DIR" "$CHANNEL_LOGS_DIR"

    # # ── 1. Download new episodes ──────────────────────────────────
    # DOWNLOAD_CMD=(
    #     uv run podcast-dl download
    #     --feed-url "$FEED_URL"
    #     --output-dir "$AUDIO_DIR"
    #     --start-date "$START_DATE"
    #     --skip-existing
    #     --no-confirm
    # )

    # if [[ "$DRY_RUN" == true ]]; then
    #     echo "  [DRY RUN] ${DOWNLOAD_CMD[*]}"
    # else
    #     echo "  Downloading…"
    #     if ! "${DOWNLOAD_CMD[@]}"; then
    #         echo "  ⚠️  Download failed for $SLUG — skipping GPU jobs."
    #         continue
    #     fi
    # fi

    # # ── 2. Submit transcription job ───────────────────────────────
    # TRANSCRIBE_ARGS=(
    #     --parsable
    #     --export "CHANNEL=$SLUG,AUDIO_DIR=$AUDIO_DIR,TRANSCRIPTS_DIR=$TRANSCRIPTS_DIR,LANGUAGE=$LANGUAGE"
    #     --job-name "transcribe_${SLUG}"
    #     --output "$CHANNEL_LOGS_DIR/transcribe_%j.log"
    # )

    # if [[ "$DRY_RUN" == true ]]; then
    #     echo "  [DRY RUN] sbatch ${TRANSCRIBE_ARGS[*]} slurm/transcribe_channel.sh"
    #     TRANSCRIBE_JID=99999
    # else
    #     TRANSCRIBE_JID=$(sbatch "${TRANSCRIBE_ARGS[@]}" "$PROJECT_ROOT/slurm/transcribe_channel.sh")
    #     echo "  ✓ Transcription submitted: job $TRANSCRIBE_JID"
    # fi

    # ── 3. Submit diarization job (waits for transcription) ───────
    DIARIZE_ARGS=(
        --parsable
        --dependency "afterok:$TRANSCRIBE_JID"
        --export "CHANNEL=$SLUG,TRANSCRIPTS_DIR=$TRANSCRIPTS_DIR,AUDIO_DIR=$AUDIO_DIR,DIARIZATIONS_DIR=$DIARIZATIONS_DIR"
        --job-name "diarize_${SLUG}"
        --output "$CHANNEL_LOGS_DIR/diarize_%j.log"
    )

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] sbatch ${DIARIZE_ARGS[*]} slurm/diarize_channel.sh"
    else
        DIARIZE_JID=$(sbatch "${DIARIZE_ARGS[@]}" "$PROJECT_ROOT/slurm/diarize_channel.sh")
        echo "  ✓ Diarization submitted: job $DIARIZE_JID (waits on $TRANSCRIBE_JID)"
    fi

    echo ""
done <<< "$CHANNELS"

echo "============================================================"
echo "Done — $(date)"
echo "============================================================"
