

module load cuda/12.1.1
module load uv/0.8.15
module load ffmpeg/4.4

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export TORCH_HOME=/scratch/alpine/alle5715/.cache
export MPLCONFIGDIR=/scratch/alpine/alle5715/.cache

uv sync
# uv pip install flash-attn --no-build-isolation


# uv run python -m podcast_downloader download "The Daily" --max-episodes 10