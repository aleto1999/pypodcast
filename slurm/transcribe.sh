#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1

#SBATCH --mail-user=alle5715@colorado.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=transcribe
#SBATCH --output=logs/transcribe.log


module load cuda/12.1.1
module load uv/0.8.15
module load ffmpeg/4.4

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export TORCH_HOME=/scratch/alpine/alle5715/.cache
export MPLCONFIGDIR=/scratch/alpine/alle5715/.cache
export PYTHONPATH=/projects/alle5715/pypodcast

uv sync --extra gpu


uv run scripts/transcribe_batch.py \
  --audio-dir /pl/active/blast-data/corpora/podcasts_selection/candace/audio \
  --output-dir outputs/transcripts/candace \
  --language en