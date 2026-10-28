# Weekly Podcast Sync on RC

Automated pipeline for downloading and processing new podcast episodes on CU Boulder Research Computing (Alpine). Runs weekly via cron on a login node: downloads new audio, then submits two chained SLURM GPU jobs for transcription and diarization. Audio files are deleted after diarization to save scratch space.

## File Overview

```
config/channels.yaml          — list of podcast channels and base paths
slurm/submit_weekly.sh      — coordinator: runs download, submits SLURM jobs
slurm/transcribe_channel.sh   — SLURM job: WhisperX transcription
slurm/diarize_channel.sh      — SLURM job: pyannote diarization + audio cleanup
```

## Storage Layout

| Data | Location |
|---|---|
| Raw audio (temporary) | `/scratch/alpine/alle5715/pypodcast/downloads/<slug>/audio` |
| Transcripts (permanent) | `/pl/active/blast-data/corpora/podcasts_selection/<slug>/transcripts` |
| Diarizations (permanent) | `/pl/active/blast-data/corpora/podcasts_selection/<slug>/diarizations` |

Audio files are deleted from scratch after their `.rttm` diarization output is confirmed to exist.

## One-Time Setup on RC

**1. Set your HuggingFace token** (required by pyannote for diarization):

```bash
echo 'export HF_TOKEN="your_huggingface_token"' >> ~/.bashrc
source ~/.bashrc
```

**2. Make sure the project is cloned to scratch:**

```bash
ls /scratch/alpine/alle5715/pypodcast
```

**3. Create the log directory:**

```bash
mkdir -p /scratch/alpine/alle5715/pypodcast/logs
```

**4. Do a dry run** to verify the config and paths without downloading or submitting anything.

> **Note:** scripts cannot be run directly on RC login nodes. Start an interactive job first:

```bash
sinteractive --partition=blanca-curc-gpu --account=blanca-curc-gpu --qos=blanca-curc-gpu --nodes=1 --ntasks=4 --mem=8G --time=0:30:00
```

Then from inside the interactive job:

```bash
cd /scratch/alpine/alle5715/pypodcast
bash slurm/submit_weekly.sh --dry-run
```

## Setting Up the Weekly Cron

SSH into an RC login node and open your crontab:

```bash
crontab -e
```

Add this line (runs every Monday at 2:00am Mountain time). The cron entry only calls `sbatch` — which is allowed on a login node — and the actual work runs inside a compute node:

```
0 2 * * 1 cd /scratch/alpine/alle5715/pypodcast && sbatch slurm/submit_weekly.sh
```

Verify it was saved:

```bash
crontab -l
```

## Pipeline Flow

Each weekly run follows this sequence:

```
submit_weekly.sh (login node)
│
├── podcast-dl download          # fetch episodes published in last 8 days
│   └── skips already-downloaded files
│
├── sbatch transcribe_channel.sh # SLURM GPU job (blanca-curc-gpu)
│   └── WhisperX → outputs/transcripts/<slug>/*.json
│
└── sbatch diarize_channel.sh    # SLURM GPU job, starts only after transcription succeeds
    ├── pyannote → outputs/diarizations/<slug>/*.rttm
    └── deletes audio files from scratch that have a matching .rttm
```

The diarization job is submitted with `--dependency afterok:<transcribe_job_id>`, so it only starts if transcription completes without error. If transcription fails, the diarization job is automatically cancelled by SLURM.

## SLURM Job Resources

Both GPU jobs use the same allocation:

| Setting | Value |
|---|---|
| Account / partition | `blanca-curc-gpu` |
| Nodes | 1 |
| Tasks | 32 |
| Memory | 50G |
| GPUs | 2 |
| Max walltime | 7 days |

Transcription sends a FAIL email. Diarization sends END and FAIL emails.

## Adding a New Channel

Edit `config/channels.yaml` and add an entry under `channels`:

```yaml
  - slug: megyn_kelly
    display_name: "The Megyn Kelly Show"
    feed_url: https://feeds.example.com/megyn
    language: en
```

The `slug` is used as the directory name under both the scratch downloads path and the PetaLibrary outputs path. No other files need to change.

## Running Manually

To trigger a sync outside the cron schedule, submit it as a job from the login node:

```bash
cd /scratch/alpine/alle5715/pypodcast
sbatch slurm/submit_weekly.sh
```

To reprocess a specific channel from scratch (bypassing skip logic), submit the SLURM scripts directly with `--force` added to the relevant command inside the job script, or pass the audio and output dirs manually:

```bash
# transcription only, force reprocess
sbatch --export="CHANNEL=candace,AUDIO_DIR=/scratch/alpine/alle5715/pypodcast/downloads/candace/audio,TRANSCRIPTS_DIR=/pl/active/blast-data/corpora/podcasts_selection/candace/transcripts,LANGUAGE=en" \
    slurm/transcribe_channel.sh
```

## Monitoring Jobs

```bash
# check your queued and running jobs
squeue -u alle5715

# watch a job's log in real time
tail -f logs/transcribe_candace_<jobid>.log

# check cron output
tail -f logs/weekly_cron.log
```
