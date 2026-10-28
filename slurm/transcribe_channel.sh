#!/bin/bash
# transcribe_channel.sh — SLURM GPU job for WhisperX transcription.
# Submitted by submit_weekly.sh; receives variables via --export:
#   CHANNEL, AUDIO_DIR, TRANSCRIPTS_DIR, LANGUAGE

#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=50G
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00

#SBATCH --mail-user=alle5715@colorado.edu
#SBATCH --mail-type=FAIL
# job-name and output are set by submit_weekly.sh via sbatch --job-name / --output

set -euo pipefail

echo "================================================"
echo "Transcription job: $CHANNEL"
echo "Audio dir:        $AUDIO_DIR"
echo "Transcripts dir:  $TRANSCRIPTS_DIR"
echo "Language:         $LANGUAGE"
echo "Started:          $(date)"
echo "Node:             $SLURMD_NODENAME"
echo "================================================"

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

uv run scripts/transcribe_batch.py \
    --audio-dir "$AUDIO_DIR" \
    --output-dir "$TRANSCRIPTS_DIR" \
    --language "$LANGUAGE"

echo "================================================"
echo "Transcription complete: $(date)"
echo "================================================"
