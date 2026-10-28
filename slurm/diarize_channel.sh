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
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00

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

PROJECT_ROOT=/scratch/alpine/alle5715/pypodcast

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

echo "================================================"
echo "Diarization complete: $(date)"
echo "================================================"
