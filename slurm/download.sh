

module load cuda/12.1.1
module load uv/0.8.15


uv sync
uv pip install flash-attn --no-build-isolation


uv run python -m podcast_downloader download "The Daily" --max-episodes 1