# running stuff on rc

## a) using `sinteractive` on our partition

```bash
# start a job with 20 cpu cores, 4g ram-per-cpu core, and one "full" h100 gpu: 
sinteractive --partition=blanca-blast-lecs --account=blanca-blast-lecs --qos=blanca-blast-lecs --nodes=1 --ntasks=20 --mem-per-cpu=4000m --gres=gpu:h100_7g.80gb
```

maria said that we should use $5$ `ntasks` per $\texttt{20 gb}$ of gpu usage. therefore, the complete one ($\texttt{80 gb}$) should have $20$ workers. however, rohan might mention u in the slack channel if u do this lol.

## alternative (in case ($\texttt{80 gb}$ is being used by someone else)  - using `sinteractive` on the `clearlab` partition

both partitions work for the transcription step. the `clearlab` one is not useful for the diarization pipeline since it needs at least $\texttt{80 gb}$ of memory, which none of those gpus have. what i did, for speeding things up, was using all of these gpus - plus the ones in our partition.

```bash
sinteractive --partition=blanca-clearlab1 --qos=blanca-clearlab1 --account=blanca-clearlab1 --nodelist=bgpu-g4-u30 --gpu=1
```

## b) add the gpu to the environment

```bash
export CUDA_VISIBLE_DEVICES="0"
```

## c) load the modules needed

```bash
module load cuda/12.1.1.lua
```

```bash
module load ffmpeg/4.4
```

```bash
module load uv
```

## d) add the dependencies needed for correct gpu detection/usage

```bash
uv sync --extra gpu
```

## e) transcription

```bash
wget -O /scratch/alpine/juva3822/.torchcache/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth
```

the path after `--audio-dir` is where the audio files i downloaded are stored. u can use those for testing, or use the ones u wanna process (ofc, lol).

```bash
uv run scripts/transcribe_batch.py \
	--audio-dir /pl/active/blast-data/juan/outputs_podcasts/downloads/candace/ \
	--output-dir outputs/transcripts/candace/ \
	--language en \
	--device cuda
```

this is what i'm seeing after running the previous command:

> [!info] note
> if the gpu utilization shows $0.0\%$ or very low usage, that means that $\texttt{pynvml}$ is likely not returning utilization for mig instances.
>
> mig partitions often report $0\%$ via `nvmldevicegetutilizationrates` because $\texttt{nvml}$ doesn't support per-mig-instance utilization tracking. this is a known nvidia limitation 😒.

## g) diarization

this is a sample command i used for my pipeline. i leave my hf token here, so u can use it if u want to. u should adapt the number of `workers` depending on how many cpus u asked for when running `sinteractive … --ntasks=5`.

the `--compile-mode` is just an optimization for running things on (in ?) a gpu.

```bash
./scripts/diarize_all_podcasts.sh \
	--transcripts-root /outputs/transcripts/candace \
        --audio-base-dir  /pl/active/blast-data/juan/outputs_podcasts/downloads/candace \
        --output-dir /outputs/diarizations/candace \
        --hf-token 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 \
        --workers 5 \
        --compile-mode max-autotune
```
