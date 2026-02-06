"""Keyword analysis browser with parallel processing."""

import json
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from terminal_data_visualizer.config import (
    COLOR_DIM,
    COLOR_INFO,
    COLOR_SUCCESS,
    COLOR_WARNING,
    OUTPUTS_PATH,
    TRANSCRIPTS_DIR,
)

console = Console(record=True)

KEYWORD_ANALYSIS_PATH = Path("outputs/analysis/keyword_analysis")
TRANSCRIPTS_PATH = OUTPUTS_PATH / TRANSCRIPTS_DIR
LABELED_TRANSCRIPTS_PATH = OUTPUTS_PATH / TRANSCRIPTS_DIR

# labels considered "negative" for sentiment calculation.
NEGATIVE_LABELS = {
    # hate_speech_detection model.
    "HATE",
    # hate_against_minorities model.
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    # hostile_content model.
    "OFFENSIVE",
    "NEITHER",  # exclude from negative count.
}

# labels considered "positive" or neutral.
POSITIVE_LABELS = {
    "NOT-HATE",
    "non-toxic",
    "NOT_OFFENSIVE",
}

# color scheme for speakers.
SPEAKER_COLORS = {
    "SPEAKER_00": "bright_cyan",
    "SPEAKER_01": "bright_green",
    "SPEAKER_02": "bright_magenta",
    "SPEAKER_03": "bright_yellow",
    "SPEAKER_04": "bright_blue",
    "SPEAKER_05": "bright_red",
}


def get_speaker_color(speaker: str) -> str:
    """Get consistent color for a speaker."""
    return SPEAKER_COLORS.get(speaker, "white")


@dataclass
class KeywordMatch:
    """A single keyword match from analysis."""

    keyword: str
    matched_text: str
    context_before: str
    context_after: str
    speaker: str
    start_time: float
    end_time: float
    category: str
    confidence_tier: str
    segment_id: int = 0


@dataclass
class CategoryStats:
    """Statistics for a category across all files."""

    category: str
    match_count: int
    file_count: int
    files: list[Path]  # files containing this category


@dataclass
class TranscriptSegment:
    """A segment from a transcript."""

    segment_id: int
    text: str
    speaker: str
    start_time: float
    end_time: float


@dataclass
class AnalysisFile:
    """Parsed analysis file with matches."""

    path: Path
    transcript_file: str
    matches: list[KeywordMatch]


@dataclass
class DocumentSentiment:
    """Document-level sentiment aggregation from utterance labels."""

    total_segments: int
    negative_count: int
    positive_count: int
    neutral_count: int
    hostile_count: int  # from hostile_content model specifically.
    toxic_count: int  # from hate_against_minorities model.

    @property
    def negative_pct(self) -> float:
        """Percentage of segments with negative labels."""
        if self.total_segments == 0:
            return 0.0
        return (self.negative_count / self.total_segments) * 100

    @property
    def hostile_pct(self) -> float:
        """Percentage of segments labeled as OFFENSIVE."""
        if self.total_segments == 0:
            return 0.0
        return (self.hostile_count / self.total_segments) * 100

    def format_summary(self) -> str:
        """Format a short summary string for display."""
        if self.total_segments == 0:
            return f"[{COLOR_DIM}]N/A[/{COLOR_DIM}]"

        # use hostile content as primary indicator (most reliable).
        if self.hostile_count > 0:
            pct = self.hostile_pct
            if pct >= 50:
                color = "red"
            elif pct >= 25:
                color = "yellow"
            else:
                color = "green"
            return f"[{color}]{pct:.0f}%[/{color}] ({self.hostile_count})"

        return f"[{COLOR_SUCCESS}]0%[/{COLOR_SUCCESS}] (0)"


def parse_analysis_file(file_path: Path) -> AnalysisFile | None:
    """Parse a keyword analysis JSON file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        matches = []
        for match in data.get("matches", []):
            matches.append(
                KeywordMatch(
                    keyword=match.get("keyword", ""),
                    matched_text=match.get("matched_text", ""),
                    context_before=match.get("context_before", ""),
                    context_after=match.get("context_after", ""),
                    speaker=match.get("speaker", ""),
                    start_time=match.get("start_time", 0.0),
                    end_time=match.get("end_time", 0.0),
                    category=match.get("category", ""),
                    confidence_tier=match.get("confidence_tier", ""),
                    segment_id=match.get("segment_id", 0),
                )
            )

        return AnalysisFile(
            path=file_path,
            transcript_file=data.get("transcript_file", ""),
            matches=matches,
        )
    except (json.JSONDecodeError, OSError):
        return None


def load_transcript_segments(transcript_path: Path) -> list[TranscriptSegment]:
    """Load segments from a transcript file."""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)

        segments = []
        for idx, seg in enumerate(data.get("segments", [])):
            segments.append(
                TranscriptSegment(
                    segment_id=idx,
                    text=seg.get("text", ""),
                    speaker=seg.get("speaker", "UNKNOWN"),
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                )
            )
        return segments
    except (json.JSONDecodeError, OSError):
        return []


def load_document_sentiment(analysis_file: Path) -> DocumentSentiment | None:
    """Load and calculate document-level sentiment from labeled transcript.

    Args:
        analysis_file: Path to the keyword analysis file (used to find transcript).

    Returns:
        DocumentSentiment with aggregated label counts, or None if not found.
    """
    # find corresponding labeled transcript.
    podcast_name = analysis_file.parent.name
    analysis_stem = analysis_file.stem
    if analysis_stem.endswith("_keywords"):
        transcript_stem = analysis_stem[: -len("_keywords")]
    else:
        transcript_stem = analysis_stem

    labeled_path = LABELED_TRANSCRIPTS_PATH / podcast_name / f"{transcript_stem}.json"

    if not labeled_path.exists():
        return None

    try:
        with open(labeled_path, encoding="utf-8") as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            return None

        total = len(segments)
        negative_count = 0
        positive_count = 0
        neutral_count = 0
        hostile_count = 0
        toxic_count = 0

        for seg in segments:
            classifications = seg.get("classifications", [])
            seg_is_negative = False
            seg_is_positive = False

            for clf in classifications:
                model = clf.get("model_name", "")
                label = clf.get("label", "")

                # check hostile_content model (most reliable for offensive content).
                if model == "hostile_content":
                    if label == "OFFENSIVE":
                        hostile_count += 1
                        seg_is_negative = True

                # check hate_against_minorities model.
                elif model == "hate_against_minorities":
                    if label in {
                        "toxic",
                        "severe_toxic",
                        "obscene",
                        "threat",
                        "insult",
                        "identity_hate",
                    }:
                        toxic_count += 1
                        seg_is_negative = True

                # check hate_speech_detection model.
                elif model == "hate_speech_detection":
                    if label == "HATE":
                        seg_is_negative = True
                    elif label == "NOT-HATE":
                        seg_is_positive = True

            if seg_is_negative:
                negative_count += 1
            elif seg_is_positive:
                positive_count += 1
            else:
                neutral_count += 1

        return DocumentSentiment(
            total_segments=total,
            negative_count=negative_count,
            positive_count=positive_count,
            neutral_count=neutral_count,
            hostile_count=hostile_count,
            toxic_count=toxic_count,
        )

    except (json.JSONDecodeError, OSError):
        return None


# cache for document sentiment to avoid repeated loading.
_sentiment_cache: dict[str, DocumentSentiment | None] = {}


def get_document_sentiment_cached(analysis_file: Path) -> DocumentSentiment | None:
    """Get document sentiment with caching."""
    cache_key = str(analysis_file)
    if cache_key not in _sentiment_cache:
        _sentiment_cache[cache_key] = load_document_sentiment(analysis_file)
    return _sentiment_cache[cache_key]


def find_transcript_path(analysis_file: Path, original_path: str) -> Path | None:
    """Find the local transcript path from an analysis file."""
    # extract podcast name and filename from the analysis file path.
    podcast_name = analysis_file.parent.name
    # the analysis file is named like: episode_name_keywords.json
    # the transcript is named like: episode_name.json
    analysis_stem = analysis_file.stem
    if analysis_stem.endswith("_keywords"):
        transcript_stem = analysis_stem[: -len("_keywords")]
    else:
        transcript_stem = analysis_stem

    # try local path first.
    local_path = TRANSCRIPTS_PATH / podcast_name / f"{transcript_stem}.json"
    if local_path.exists():
        return local_path

    # try extracting from original path.
    if original_path:
        original = Path(original_path)
        local_path = TRANSCRIPTS_PATH / podcast_name / original.name
        if local_path.exists():
            return local_path

    return None


def get_all_keywords_parallel(folder_path: Path, max_workers: int = 8) -> set[str]:
    """Extract all unique keywords from analysis files in parallel."""
    if not folder_path.exists():
        return set()

    json_files = list(folder_path.glob("*.json"))
    keywords: set[str] = set()

    def extract_keywords(file_path: Path) -> set[str]:
        result = parse_analysis_file(file_path)
        if result:
            return {m.keyword for m in result.matches}
        return set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_keywords, f): f for f in json_files}
        for future in as_completed(futures):
            try:
                keywords.update(future.result())
            except Exception:
                pass

    return keywords


def get_all_categories_parallel(folder_path: Path, max_workers: int = 8) -> set[str]:
    """Extract all unique categories from analysis files in parallel."""
    if not folder_path.exists():
        return set()

    json_files = list(folder_path.glob("*.json"))
    categories: set[str] = set()

    def extract_categories(file_path: Path) -> set[str]:
        result = parse_analysis_file(file_path)
        if result:
            return {m.category for m in result.matches if m.category}
        return set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_categories, f): f for f in json_files}
        for future in as_completed(futures):
            try:
                categories.update(future.result())
            except Exception:
                pass

    return categories


def get_category_stats_parallel(
    folder_path: Path, max_workers: int = 8
) -> dict[str, CategoryStats]:
    """
    Extract category statistics with file associations from analysis files.

    Returns a dict mapping category name to CategoryStats with match counts
    and list of files containing that category.
    """
    if not folder_path.exists():
        return {}

    json_files = list(folder_path.glob("*.json"))
    # category -> {match_count, files set}
    category_data: dict[str, dict] = {}

    def extract_category_info(file_path: Path) -> list[tuple[str, int]]:
        """Return list of (category, match_count) for this file."""
        result = parse_analysis_file(file_path)
        if not result:
            return []
        # count matches per category in this file.
        cat_counts: dict[str, int] = {}
        for m in result.matches:
            if m.category:
                cat_counts[m.category] = cat_counts.get(m.category, 0) + 1
        return list(cat_counts.items())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_category_info, f): f for f in json_files}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                cat_info = future.result()
                for category, count in cat_info:
                    if category not in category_data:
                        category_data[category] = {"match_count": 0, "files": set()}
                    category_data[category]["match_count"] += count
                    category_data[category]["files"].add(file_path)
            except Exception:
                pass

    # convert to CategoryStats objects.
    return {
        cat: CategoryStats(
            category=cat,
            match_count=data["match_count"],
            file_count=len(data["files"]),
            files=sorted(data["files"], key=lambda p: p.name),
        )
        for cat, data in category_data.items()
    }


def get_matches_for_category(file_path: Path, category: str) -> list[KeywordMatch]:
    """Get all matches for a specific category from a file."""
    result = parse_analysis_file(file_path)
    if not result:
        return []
    return [m for m in result.matches if m.category == category]


def get_matches_for_category_in_folder(
    folder_path: Path, category: str, max_workers: int = 8
) -> list[tuple[Path, KeywordMatch]]:
    """Get all matches for a category across all files in a folder."""
    if not folder_path.exists():
        return []

    json_files = list(folder_path.glob("*.json"))
    results: list[tuple[Path, KeywordMatch]] = []

    def search_file(file_path: Path) -> list[tuple[Path, KeywordMatch]]:
        matches = get_matches_for_category(file_path, category)
        return [(file_path, m) for m in matches]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(search_file, f): f for f in json_files}
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception:
                pass

    return results


def search_keywords_parallel(
    folder_path: Path, keywords: list[str], max_workers: int = 8
) -> list[tuple[Path, KeywordMatch]]:
    """Search for specific keywords in analysis files using parallel processing."""
    if not folder_path.exists():
        return []

    json_files = list(folder_path.glob("*.json"))
    results: list[tuple[Path, KeywordMatch]] = []
    keywords_lower = {k.lower() for k in keywords}

    def search_file(file_path: Path) -> list[tuple[Path, KeywordMatch]]:
        file_results = []
        analysis = parse_analysis_file(file_path)
        if analysis:
            for match in analysis.matches:
                if match.keyword.lower() in keywords_lower:
                    file_results.append((file_path, match))
        return file_results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(search_file, f): f for f in json_files}
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception:
                pass

    return results


def format_text_as_paragraph(text: str, width: int = 80) -> str:
    """Format text into readable paragraphs with word wrapping."""
    # clean up extra whitespace.
    text = " ".join(text.split())
    # wrap text.
    return textwrap.fill(text, width=width)


def highlight_keywords_in_text(text: str, keywords: list[str]) -> Text:
    """Highlight keywords in text with colors."""
    rich_text = Text(text)

    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for match in pattern.finditer(text):
            rich_text.stylize("bold yellow on dark_red", match.start(), match.end())

    return rich_text


def display_keyword_matches(
    matches: list[tuple[Path, KeywordMatch]],
    title: str = "Keyword Matches",
    show_sentiment: bool = True,
) -> None:
    """Display keyword matches in a formatted table.

    Args:
        matches: List of (file_path, KeywordMatch) tuples.
        title: Table title.
        show_sentiment: Whether to show document-level sentiment column.
    """
    if not matches:
        console.print(f"[{COLOR_WARNING}]No matches found.[/{COLOR_WARNING}]")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", style="yellow", justify="right", width=4)
    table.add_column("Episode", style="cyan", max_width=30)
    table.add_column("Keyword", style="bold green", width=12)
    table.add_column("Category", style="blue", width=16)
    table.add_column("Speaker", style="magenta", width=12)
    table.add_column("Time", style="dim", width=8)

    if show_sentiment:
        table.add_column("Sentiment", width=14, justify="right")

    # pre-load sentiment data for unique files to avoid repeated loading.
    sentiment_data: dict[str, DocumentSentiment | None] = {}
    if show_sentiment:
        unique_files = {str(fp) for fp, _ in matches[:100]}
        for fp_str in unique_files:
            fp = Path(fp_str)
            sentiment_data[fp_str] = get_document_sentiment_cached(fp)

    for idx, (file_path, match) in enumerate(matches[:100], 1):  # limit display.
        # format episode name more readably.
        episode_name = file_path.stem.replace("_", " ").replace("keywords", "").strip()
        if len(episode_name) > 28:
            episode_name = episode_name[:25] + "..."

        # format time.
        time_str = f"{match.start_time:.0f}s"

        # speaker with color.
        speaker_color = get_speaker_color(match.speaker)

        # get sentiment for this file.
        row_data = [
            str(idx),
            episode_name,
            f"[bold]{match.keyword}[/bold]",
            match.category,
            f"[{speaker_color}]{match.speaker}[/{speaker_color}]",
            time_str,
        ]

        if show_sentiment:
            sentiment = sentiment_data.get(str(file_path))
            if sentiment:
                row_data.append(sentiment.format_summary())
            else:
                row_data.append(f"[{COLOR_DIM}]N/A[/{COLOR_DIM}]")

        table.add_row(*row_data)

    console.print(table)

    # show legend for sentiment column.
    if show_sentiment:
        console.print(
            "\n[dim]Sentiment: % of utterances labeled OFFENSIVE (hostile_content model)[/{COLOR_DIM}]"
        )

    console.print(f"[{COLOR_DIM}]Showing {min(len(matches), 100)} of {len(matches)} matches[/{COLOR_DIM}]")


def display_utterance_detail(
    file_path: Path,
    match: KeywordMatch,
    context_n: int = 0,
    transcript_file: str = "",
) -> tuple[list[TranscriptSegment], int]:
    """Display detailed utterance with surrounding context segments.

    Returns segments and target_segment_id for export functionality.
    """
    # format episode name.
    episode_name = file_path.stem.replace("_", " ").replace("keywords", "").strip()

    # speaker color.
    speaker_color = get_speaker_color(match.speaker)

    # build info table for better layout.
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Label", style="cyan")
    info_table.add_column("Value")

    info_table.add_row("Keyword", f"[bold yellow]{match.keyword}[/bold yellow]")
    info_table.add_row("Category", f"[{COLOR_INFO}]{match.category}[/{COLOR_INFO}]")
    info_table.add_row("Speaker", f"[{speaker_color}]{match.speaker}[/{speaker_color}]")
    info_table.add_row("Time", f"{match.start_time:.1f}s - {match.end_time:.1f}s")
    info_table.add_row("Confidence", match.confidence_tier)
    info_table.add_row("Segment", str(match.segment_id))

    console.print(Panel(info_table, title=f"📄 {episode_name}", border_style="blue"))

    segments: list[TranscriptSegment] = []

    # load transcript for context if n > 0.
    if context_n > 0:
        transcript_path = find_transcript_path(file_path, transcript_file)
        if transcript_path:
            segments = load_transcript_segments(transcript_path)
            if segments:
                display_context_segments(segments, match.segment_id, context_n, match.keyword)
            else:
                console.print(f"[{COLOR_WARNING}]Could not load transcript segments.[/{COLOR_WARNING}]")
                display_inline_context(match)
        else:
            console.print(f"[{COLOR_WARNING}]Transcript file not found locally.[/{COLOR_WARNING}]")
            display_inline_context(match)
    else:
        display_inline_context(match)

    return segments, match.segment_id


def display_inline_context(match: KeywordMatch) -> None:
    """Display inline context from the match with proper formatting."""
    # format as paragraph.
    context_text = f"{match.context_before}{match.matched_text}{match.context_after}"
    formatted_text = format_text_as_paragraph(context_text, width=90)

    # highlight the keyword.
    highlighted = highlight_keywords_in_text(formatted_text, [match.keyword])

    console.print()
    console.print(Panel(highlighted, title="📝 Context", border_style="dim", padding=(1, 2)))


def display_context_segments(
    segments: list[TranscriptSegment],
    target_segment_id: int,
    n: int,
    highlight_keyword: str = "",
) -> None:
    """Display n segments before and after the target segment with readable formatting."""
    total_segments = len(segments)

    # calculate range.
    start_idx = max(0, target_segment_id - n)
    end_idx = min(total_segments, target_segment_id + n + 1)

    console.print()
    console.print(
        Panel(
            f"[{COLOR_DIM}]Showing {end_idx - start_idx} segments "
            f"(#{start_idx} to #{end_idx - 1}, target: #{target_segment_id})[/{COLOR_DIM}]",
            border_style="dim",
        )
    )
    console.print()

    for idx in range(start_idx, end_idx):
        seg = segments[idx]
        is_target = idx == target_segment_id

        # format time as MM:SS.
        minutes = int(seg.start_time // 60)
        seconds = int(seg.start_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        # get speaker color.
        speaker_color = get_speaker_color(seg.speaker)

        # format text as readable paragraph.
        text = format_text_as_paragraph(seg.text.strip(), width=85)

        # highlight keyword in target segment.
        if is_target and highlight_keyword:
            pattern = re.compile(re.escape(highlight_keyword), re.IGNORECASE)
            # replace with highlighted version.
            text = pattern.sub(
                lambda m: f"[bold yellow on dark_red]{m.group()}[/bold yellow on dark_red]",
                text,
            )

        # build segment display.
        if is_target:
            prefix = "▶ "
            border_style = "green"
            title_style = "bold green"
        else:
            prefix = "  "
            border_style = "dim"
            title_style = "dim"

        # create header with speaker and time.
        header = (
            f"{prefix}[{title_style}]Segment {idx}[/{title_style}] │ "
            f"[{speaker_color}]{seg.speaker}[/{speaker_color}] │ "
            f"[{COLOR_DIM}]{time_str}[/{COLOR_DIM}]"
        )

        # display segment.
        console.print(
            Panel(
                text,
                title=header,
                border_style=border_style,
                padding=(1, 2),
            )
        )


def get_subfolders(base_path: Path = KEYWORD_ANALYSIS_PATH) -> list[Path]:
    """Get all subfolders (podcasts) in keyword analysis directory."""
    if not base_path.exists():
        return []
    return sorted([p for p in base_path.iterdir() if p.is_dir()])


def display_keywords_table(keywords: set[str], title: str = "All Keywords") -> None:
    """Display keywords in a formatted table."""
    if not keywords:
        console.print(f"[{COLOR_WARNING}]No keywords found.[/{COLOR_WARNING}]")
        return

    sorted_keywords = sorted(keywords)
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="yellow", justify="right")
    table.add_column("Keyword", style="green")

    for idx, keyword in enumerate(sorted_keywords, 1):
        table.add_row(str(idx), keyword)

    console.print(table)
    console.print(f"\n[dim]Total keywords: {len(keywords)}[/{COLOR_DIM}]")


def display_categories_table(categories: set[str], title: str = "All Categories") -> None:
    """Display categories in a formatted table."""
    if not categories:
        console.print(f"[{COLOR_WARNING}]No categories found.[/{COLOR_WARNING}]")
        return

    sorted_categories = sorted(categories)
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="yellow", justify="right")
    table.add_column("Category", style="blue")

    for idx, category in enumerate(sorted_categories, 1):
        table.add_row(str(idx), category)

    console.print(table)
    console.print(f"\n[dim]Total categories: {len(categories)}[/{COLOR_DIM}]")


def display_categories_with_stats(
    category_stats: dict[str, CategoryStats],
    title: str = "📂 All Categories",
) -> list[str]:
    """
    Display categories with match counts and file counts.

    Returns sorted list of category names for selection.
    """
    if not category_stats:
        console.print(f"[{COLOR_WARNING}]No categories found.[/{COLOR_WARNING}]")
        return []

    # sort by match count descending, then alphabetically.
    sorted_cats = sorted(
        category_stats.values(),
        key=lambda x: (-x.match_count, x.category.lower()),
    )

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="yellow", justify="right", width=4)
    table.add_column("Category", style="blue")
    table.add_column("Matches", style="green", justify="right")
    table.add_column("Files", style="cyan", justify="right")

    for idx, stats in enumerate(sorted_cats, 1):
        table.add_row(
            str(idx),
            stats.category,
            str(stats.match_count),
            str(stats.file_count),
        )

    console.print(table)
    total_matches = sum(s.match_count for s in sorted_cats)
    console.print(f"\n[dim]Total: {len(sorted_cats)} categories, {total_matches} matches[/{COLOR_DIM}]")

    return [s.category for s in sorted_cats]


def display_category_files(
    category: str,
    files_with_counts: list[tuple[Path, int]],
    title: str | None = None,
) -> list[Path]:
    """
    Display files containing matches for a category.

    Args:
        category: The category name
        files_with_counts: List of (file_path, match_count) tuples
        title: Optional custom title

    Returns:
        Sorted list of file paths for selection.
    """
    if not files_with_counts:
        console.print(f"[{COLOR_WARNING}]No files found for category '{category}'.[/{COLOR_WARNING}]")
        return []

    display_title = title or f"📁 Files with '{category}' matches"

    # sort by match count descending, then by filename.
    sorted_files = sorted(
        files_with_counts,
        key=lambda x: (-x[1], x[0].name.lower()),
    )

    table = Table(title=display_title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="yellow", justify="right", width=4)
    table.add_column("File", style="cyan")
    table.add_column("Matches", style="green", justify="right")

    for idx, (file_path, count) in enumerate(sorted_files, 1):
        table.add_row(str(idx), file_path.name, str(count))

    console.print(table)
    total_matches = sum(c for _, c in sorted_files)
    console.print(f"\n[dim]Total: {len(sorted_files)} files, {total_matches} matches[/{COLOR_DIM}]")

    return [f for f, _ in sorted_files]


def display_category_matches(
    category: str,
    matches: list[KeywordMatch],
    file_name: str,
    title: str | None = None,
) -> list[KeywordMatch]:
    """
    Display matches for a category within a specific file.

    Args:
        category: The category name
        matches: List of KeywordMatch objects
        file_name: Name of the file for display
        title: Optional custom title

    Returns:
        The list of matches for selection.
    """
    if not matches:
        console.print(f"[{COLOR_WARNING}]No matches found for '{category}' in {file_name}.[/{COLOR_WARNING}]")
        return []

    display_title = title or f"🔍 '{category}' in {file_name}"

    table = Table(title=display_title, show_header=True, header_style="bold magenta")
    table.add_column("#", style="yellow", justify="right", width=4)
    table.add_column("Speaker", style="cyan", width=12)
    table.add_column("Time", style="dim", width=12)
    table.add_column("Matched Text", style="green", max_width=50, overflow="ellipsis")
    table.add_column("Confidence", style="magenta", width=10)

    for idx, match in enumerate(matches, 1):
        # format time as mm:ss.
        mins, secs = divmod(int(match.start_time), 60)
        time_str = f"{mins:02d}:{secs:02d}"

        # truncate matched text if too long.
        text = match.matched_text
        matched = text[:47] + "..." if len(text) > 50 else text

        table.add_row(
            str(idx),
            match.speaker,
            time_str,
            matched,
            match.confidence_tier,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(matches)} matches[/{COLOR_DIM}]")

    return matches


def display_match_detail(match: KeywordMatch, file_name: str) -> None:
    """
    Display detailed view of a single match with full context.

    Args:
        match: The KeywordMatch to display
        file_name: Name of the source file
    """
    # format timestamp.
    start_mins, start_secs = divmod(int(match.start_time), 60)
    end_mins, end_secs = divmod(int(match.end_time), 60)
    time_range = f"{start_mins:02d}:{start_secs:02d} - {end_mins:02d}:{end_secs:02d}"

    # build context text with highlighting.
    context_parts = []
    if match.context_before:
        context_parts.append(f"[{COLOR_DIM}]{match.context_before}[/{COLOR_DIM}]")
    context_parts.append(f"[bold green]{match.matched_text}[/bold green]")
    if match.context_after:
        context_parts.append(f"[{COLOR_DIM}]{match.context_after}[/{COLOR_DIM}]")
    full_context = " ".join(context_parts)

    # get speaker color.
    speaker_color = get_speaker_color(match.speaker)

    # build the panel content.
    content = Text()
    content.append("File: ", style="bold")
    content.append(f"{file_name}\n", style="cyan")
    content.append("Speaker: ", style="bold")
    content.append(f"{match.speaker}\n", style=speaker_color)
    content.append("Time: ", style="bold")
    content.append(f"{time_range}\n", style="dim")
    content.append("Category: ", style="bold")
    content.append(f"{match.category}\n", style="blue")
    content.append("Keyword: ", style="bold")
    content.append(f"{match.keyword}\n", style="green")
    content.append("Confidence: ", style="bold")
    content.append(f"{match.confidence_tier}\n\n", style="magenta")

    panel = Panel(
        content,
        title="🔎 Match Details",
        border_style="cyan",
    )
    console.print(panel)

    # display utterance with context.
    context_panel = Panel(
        full_context,
        title="💬 Utterance Context",
        border_style="green",
    )
    console.print(context_panel)
