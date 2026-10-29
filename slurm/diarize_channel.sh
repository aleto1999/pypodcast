#!/bin/bash
# diarize_channel.sh — SLURM GPU job for speaker diarization (pyannote).
# Submitted by submit_weekly.sh with --dependency afterok:<transcribe_job_id>.
# Receives variables via --export:
#   CHANNEL, TRANSCRIPTS_DIR, AUDIO_DIR, DIARIZATIONS_DIR
#
# Requires HF_TOKEN to be set in the environment (add to ~/.bashrc on RC):
#   export HF_TOKEN="your_huggingface_token"

#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00

#SBATCH --mail-user=alle5715@colorado.edu
#SBATCH --mail-type=END,FAIL
# job-name and output are set by submit_weekly.sh

set -euo pipefail

echo "================================================"
echo "Diarization job: $CHANNEL"
echo "Transcripts dir:  $TRANSCRIPTS_DIR"
echo "Audio dir:        $AUDIO_DIR"
echo "Diarizations dir: $DIARIZATIONS_DIR"
echo "Started:          $(date)"
echo "Node:             $SLURMD_NODENAME"
echo "================================================"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Add 'export HF_TOKEN=...' to ~/.bashrc on RC."
    exit 1
fi

PROJECT_ROOT=/projects/alle5715/pypodcast

module load cuda/12.1.1
module load uv/0.8.15
module load ffmpeg/4.4

export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="$SLURM_SCRATCH/cache/HF"
export TORCH_HOME=/scratch/alpine/alle5715/.cache
export MPLCONFIGDIR=/scratch/alpine/alle5715/.cache
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
uv sync --extra gpu

./scripts/diarize_all_podcasts.sh \
    --transcripts-root "$TRANSCRIPTS_DIR" \
    --audio-base-dir  "$AUDIO_DIR" \
    --output-dir      "$DIARIZATIONS_DIR" \
    --hf-token        "$HF_TOKEN" \
    --workers         20 \
    --compile-mode    max-autotune


# ── Cleanup: remove audio files that have been successfully diarized ──────
echo "================================================"
echo "Cleaning up audio files with completed diarizations…"

deleted=0
skipped=0
# Audio files sit one level below AUDIO_DIR in a subdir podcast-dl creates
# (e.g. AUDIO_DIR/candace/episode.mp3). RTTMs mirror that subdir under
# DIARIZATIONS_DIR (e.g. DIARIZATIONS_DIR/candace/episode.rttm).
while IFS= read -r -d '' audio_file; do
    base=$(basename "$audio_file")
    base_noext="${base%.*}"
    subdir=$(basename "$(dirname "$audio_file")")
    rttm_file="$DIARIZATIONS_DIR/$subdir/${base_noext}.rttm"
    if [[ -f "$rttm_file" ]]; then
        rm "$audio_file"
        echo "  deleted: $subdir/$base"
        ((deleted++)) || true
    else
        echo "  kept (no RTTM found): $subdir/$base"
        ((skipped++)) || true
    fi
done < <(find "$AUDIO_DIR" -maxdepth 2 \( -name "*.mp3" -o -name "*.m4a" \) -print0)

echo "Deleted $deleted audio file(s), kept $skipped (no matching RTTM)."

echo "================================================"
echo "Diarization complete: $(date)"
echo "================================================"

