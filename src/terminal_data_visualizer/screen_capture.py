"""Screen capture utilities for terminal data visualizer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

from terminal_data_visualizer.config import (
    SCREENSHOT_DIR,
    SCREENSHOT_DPI,
    SCREENSHOT_QUALITY,
)


def save_current_screen(
    console: Console,
    screen_name: str | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """Save the current console output to a file.

    Attempts to save in the best available format:
    1. JPG (if cairosvg and Pillow available)
    2. PNG (if cairosvg available)
    3. SVG (always available via Rich)

    Args:
        console: Rich Console with record=True.
        screen_name: Optional name for the screenshot file.
        output_dir: Optional output directory. Defaults to SCREENSHOT_DIR.

    Returns:
        Path to the saved file, or None if save failed.
    """
    if output_dir is None:
        output_dir = SCREENSHOT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # generate filename with timestamp.
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_name = screen_name or "screenshot"
    base_name = base_name.replace(" ", "_").replace("/", "_")

    # try to export as SVG first (always available).
    svg_path = output_dir / f"{base_name}_{timestamp}.svg"
    try:
        svg_content = console.export_svg(title=screen_name or "Terminal Viewer")
        svg_path.write_text(svg_content)
    except Exception:
        return None

    # try to convert to PNG using cairosvg.
    png_path = output_dir / f"{base_name}_{timestamp}.png"
    try:
        import cairosvg

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            dpi=SCREENSHOT_DPI,
        )
    except ImportError:
        # cairosvg not available, keep SVG.
        return svg_path
    except Exception:
        # conversion failed, keep SVG.
        return svg_path

    # try to convert to JPG using Pillow.
    jpg_path = output_dir / f"{base_name}_{timestamp}.jpg"
    try:
        from PIL import Image

        with Image.open(png_path) as img:
            # convert RGBA to RGB for JPG.
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.save(jpg_path, "JPEG", quality=SCREENSHOT_QUALITY)

        # clean up intermediate files.
        svg_path.unlink(missing_ok=True)
        png_path.unlink(missing_ok=True)
        return jpg_path
    except ImportError:
        # Pillow not available, keep PNG.
        svg_path.unlink(missing_ok=True)
        return png_path
    except Exception:
        # conversion failed, keep PNG.
        svg_path.unlink(missing_ok=True)
        return png_path


def prompt_and_save(
    console: Console,
    screen_name: str | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """Prompt user to save screenshot and save if confirmed.

    Args:
        console: Rich Console with record=True.
        screen_name: Optional name for the screenshot file.
        output_dir: Optional output directory. Defaults to SCREENSHOT_DIR.

    Returns:
        Path to the saved file if user confirms and save succeeds, None otherwise.
    """
    try:
        if Confirm.ask("\n[dim]Save screenshot?[/dim]", default=False):
            saved_path = save_current_screen(console, screen_name, output_dir)
            if saved_path:
                console.print(f"[green]Saved:[/green] {saved_path}")
                return saved_path
            else:
                console.print("[red]Failed to save screenshot[/red]")
    except (KeyboardInterrupt, EOFError):
        pass
    return None
