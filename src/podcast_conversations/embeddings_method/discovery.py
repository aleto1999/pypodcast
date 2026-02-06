"""show discovery for embeddings analysis."""

from pathlib import Path

from rich.console import Console


def discover_shows(base_dir: Path | str | None = None) -> dict[str, list[Path]]:
    """
    scan for shows with transcript files.

    Args:
        base_dir: directory to scan for shows. defaults to transcripts_with_speaker_labels_postprocessed_with_classification_labels
                 if it exists, otherwise falls back to transcripts_with_diarization_labels_postprocessed.

    Returns:
        dict mapping show names to lists of transcript file paths.
    """
    if base_dir is None:
        # try classified transcripts first (with labels).
        classified_dir = Path("outputs/transcripts_with_speaker_labels_postprocessed_with_classification_labels")
        postprocessed_dir = Path("outputs/transcripts_with_diarization_labels_postprocessed")

        if classified_dir.exists():
            base_dir = classified_dir
        elif postprocessed_dir.exists():
            base_dir = postprocessed_dir
        else:
            return {}
    else:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            return {}

    shows: dict[str, list[Path]] = {}

    for show_dir in base_dir.iterdir():
        if show_dir.is_dir():
            json_files = list(show_dir.glob("*.json"))
            if json_files:
                shows[show_dir.name] = sorted(json_files)

    return shows


def display_available_shows(shows: dict[str, list[Path]]) -> None:
    """display shows in formatted list."""
    console = Console()
    console.print(f"\n📂 available shows for analysis ({len(shows)} found):\n")

    for idx, (show_name, files) in enumerate(sorted(shows.items()), 1):
        console.print(f"[cyan][{idx}][/cyan] {show_name}")
        console.print(f"    └─ {len(files)} json files available\n")
