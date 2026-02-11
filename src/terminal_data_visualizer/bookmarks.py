"""Bookmark and preset management for saved views."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from terminal_data_visualizer.config import (
    COLOR_ERROR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    KEY_BACK,
)

console = Console(record=True)

# bookmarks storage file.
BOOKMARKS_FILE = Path.home() / ".terminal_data_visualizer_bookmarks.json"


@dataclass
class Bookmark:
    """A saved view or search configuration."""

    name: str
    view_type: str  # "search", "filter", "show", "episode"
    parameters: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert bookmark to dictionary."""
        return {
            "name": self.name,
            "view_type": self.view_type,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bookmark:
        """Create bookmark from dictionary."""
        return cls(
            name=data.get("name", "Untitled"),
            view_type=data.get("view_type", "search"),
            parameters=data.get("parameters", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            description=data.get("description", ""),
        )


class BookmarkManager:
    """Manages bookmark storage and retrieval."""

    def __init__(self, storage_file: Path = BOOKMARKS_FILE) -> None:
        """Initialize bookmark manager."""
        self.storage_file = storage_file
        self.bookmarks: list[Bookmark] = []
        self._load()

    def _load(self) -> None:
        """Load bookmarks from storage file."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, encoding="utf-8") as f:
                    data = json.load(f)
                self.bookmarks = [Bookmark.from_dict(b) for b in data.get("bookmarks", [])]
            except (json.JSONDecodeError, OSError):
                self.bookmarks = []
        else:
            self.bookmarks = []

    def _save(self) -> None:
        """Save bookmarks to storage file."""
        try:
            data = {"bookmarks": [b.to_dict() for b in self.bookmarks]}
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            console.print(f"[{COLOR_ERROR}]Error saving bookmarks: {e}[/{COLOR_ERROR}]")

    def add(self, bookmark: Bookmark) -> None:
        """Add a new bookmark."""
        self.bookmarks.append(bookmark)
        self._save()

    def remove(self, index: int) -> bool:
        """Remove bookmark by index."""
        if 0 <= index < len(self.bookmarks):
            del self.bookmarks[index]
            self._save()
            return True
        return False

    def get(self, index: int) -> Bookmark | None:
        """Get bookmark by index."""
        if 0 <= index < len(self.bookmarks):
            return self.bookmarks[index]
        return None

    def get_by_name(self, name: str) -> Bookmark | None:
        """Get bookmark by name."""
        for bookmark in self.bookmarks:
            if bookmark.name.lower() == name.lower():
                return bookmark
        return None

    def get_by_type(self, view_type: str) -> list[Bookmark]:
        """Get all bookmarks of a specific type."""
        return [b for b in self.bookmarks if b.view_type == view_type]

    def clear(self) -> None:
        """Clear all bookmarks."""
        self.bookmarks = []
        self._save()


# global bookmark manager instance.
bookmark_manager = BookmarkManager()


def create_search_bookmark(
    name: str,
    query: str,
    output_types: list[str] | None = None,
    shows: list[str] | None = None,
    category: str | None = None,
    has_hate_speech: bool | None = None,
) -> Bookmark:
    """Create a bookmark for a search query."""
    parameters = {"query": query}
    if output_types:
        parameters["output_types"] = output_types
    if shows:
        parameters["shows"] = shows
    if category:
        parameters["category"] = category
    if has_hate_speech is not None:
        parameters["has_hate_speech"] = has_hate_speech

    return Bookmark(
        name=name,
        view_type="search",
        parameters=parameters,
        description=f"Search: {query}",
    )


def create_show_bookmark(name: str, show_name: str) -> Bookmark:
    """Create a bookmark for a show."""
    return Bookmark(
        name=name,
        view_type="show",
        parameters={"show_name": show_name},
        description=f"Show: {show_name}",
    )


def create_episode_bookmark(name: str, show_name: str, episode_name: str) -> Bookmark:
    """Create a bookmark for an episode."""
    return Bookmark(
        name=name,
        view_type="episode",
        parameters={"show_name": show_name, "episode_name": episode_name},
        description=f"Episode: {episode_name}",
    )


def create_filter_bookmark(
    name: str,
    shows: list[str] | None = None,
    has_hate_speech: bool | None = None,
    categories: list[str] | None = None,
) -> Bookmark:
    """Create a bookmark for a filter configuration."""
    parameters: dict[str, Any] = {}
    if shows:
        parameters["shows"] = shows
    if has_hate_speech is not None:
        parameters["has_hate_speech"] = has_hate_speech
    if categories:
        parameters["categories"] = categories

    return Bookmark(
        name=name,
        view_type="filter",
        parameters=parameters,
        description="Filter preset",
    )


def display_bookmarks(bookmarks: list[Bookmark]) -> None:
    """Display a list of bookmarks."""
    if not bookmarks:
        console.print(f"[{COLOR_WARNING}]No bookmarks saved[/{COLOR_WARNING}]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Name", style="yellow", width=25)
    table.add_column("Type", width=12)
    table.add_column("Description", width=40)
    table.add_column("Created", width=12)

    for i, bookmark in enumerate(bookmarks, 1):
        # format created date.
        try:
            created_dt = datetime.fromisoformat(bookmark.created_at)
            created_str = created_dt.strftime("%Y-%m-%d")
        except ValueError:
            created_str = "Unknown"

        # icon for type.
        type_icons = {
            "search": "🔍",
            "show": "📊",
            "episode": "📄",
            "filter": "🎯",
        }
        type_display = f"{type_icons.get(bookmark.view_type, '📌')} {bookmark.view_type}"

        desc = bookmark.description[:38] if len(bookmark.description) > 38 else bookmark.description

        table.add_row(str(i), bookmark.name, type_display, desc, created_str)

    console.print(table)


def bookmarks_menu() -> None:
    """Interactive bookmarks menu."""
    while True:
        console.clear()
        console.print(
            Panel(
                f"[bold {COLOR_PRIMARY}]📌 Bookmarks[/bold {COLOR_PRIMARY}]\n"
                "[dim]Manage saved views and search presets[/dim]",
                border_style="blue",
            )
        )

        table = Table(show_header=False, box=None)
        table.add_column("Option", style=COLOR_WARNING)
        table.add_column("Description")

        table.add_row("1", "📋 View all bookmarks")
        table.add_row("2", "➕ Create new bookmark")
        table.add_row("3", "🚀 Quick access bookmark")
        table.add_row("4", "🗑️  Delete bookmark")
        table.add_row("5", "🧹 Clear all bookmarks")
        table.add_row(KEY_BACK, "⬅️  Back")

        console.print(table)
        console.print()

        choice = Prompt.ask(f"[{COLOR_WARNING}]Select option[/{COLOR_WARNING}]", default=KEY_BACK)

        match choice.lower():
            case "1":
                _view_all_bookmarks()
            case "2":
                _create_bookmark_menu()
            case "3":
                _quick_access_bookmark()
            case "4":
                _delete_bookmark()
            case "5":
                _clear_all_bookmarks()
            case _ if choice.lower() == KEY_BACK:
                break


def _view_all_bookmarks() -> None:
    """View all saved bookmarks."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]📋 All Bookmarks[/bold {COLOR_PRIMARY}]\n")

    display_bookmarks(bookmark_manager.bookmarks)

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _create_bookmark_menu() -> None:
    """Create a new bookmark interactively."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]➕ Create Bookmark[/bold {COLOR_PRIMARY}]\n")

    console.print(f"[{COLOR_PRIMARY}]Bookmark types:[/{COLOR_PRIMARY}]")
    console.print("  1. Search query")
    console.print("  2. Show view")
    console.print("  3. Filter preset")
    console.print("  4. Hate speech filter")

    type_choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select type (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if type_choice.lower() == KEY_BACK:
        return

    name = Prompt.ask(f"[{COLOR_WARNING}]Bookmark name[/{COLOR_WARNING}]")
    if not name.strip():
        console.print(f"[{COLOR_ERROR}]Name cannot be empty[/{COLOR_ERROR}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    bookmark: Bookmark | None = None

    match type_choice:
        case "1":
            query = Prompt.ask(f"[{COLOR_WARNING}]Search query[/{COLOR_WARNING}]")
            if query.strip():
                bookmark = create_search_bookmark(name, query)
        case "2":
            show = Prompt.ask(f"[{COLOR_WARNING}]Show name[/{COLOR_WARNING}]")
            if show.strip():
                bookmark = create_show_bookmark(name, show)
        case "3":
            msg = f"[{COLOR_WARNING}]Shows (comma-separated, blank for all)[/{COLOR_WARNING}]"
            shows_input = Prompt.ask(msg, default="")
            shows = [s.strip() for s in shows_input.split(",") if s.strip()] or None
            bookmark = create_filter_bookmark(name, shows=shows)
        case "4":
            bookmark = create_filter_bookmark(name, has_hate_speech=True)
            bookmark.description = "Hate speech filter"

    if bookmark:
        bookmark_manager.add(bookmark)
        console.print(f"\n[{COLOR_SUCCESS}]✓ Bookmark created: {name}[/{COLOR_SUCCESS}]")
    else:
        console.print(f"[{COLOR_ERROR}]Could not create bookmark[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _quick_access_bookmark() -> None:
    """Quick access to a bookmark."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🚀 Quick Access[/bold {COLOR_PRIMARY}]\n")

    if not bookmark_manager.bookmarks:
        console.print(f"[{COLOR_WARNING}]No bookmarks saved[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    display_bookmarks(bookmark_manager.bookmarks)

    choice = Prompt.ask(
        f"\n[{COLOR_WARNING}]Select bookmark number (or '{KEY_BACK}' to go back)[/{COLOR_WARNING}]",
        default=KEY_BACK,
    )

    if choice.lower() == KEY_BACK:
        return

    try:
        idx = int(choice) - 1
        bookmark = bookmark_manager.get(idx)

        if bookmark:
            console.print(f"\n[{COLOR_PRIMARY}]Bookmark: {bookmark.name}[/{COLOR_PRIMARY}]")
            console.print(f"Type: {bookmark.view_type}")
            console.print(f"Parameters: {json.dumps(bookmark.parameters, indent=2)}")

            # in a full implementation, this would navigate to the view.
            console.print(f"\n[{COLOR_SUCCESS}]✓ Bookmark loaded[/{COLOR_SUCCESS}]")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")

    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _delete_bookmark() -> None:
    """Delete a bookmark."""
    console.clear()
    console.print(f"[bold {COLOR_PRIMARY}]🗑️ Delete Bookmark[/bold {COLOR_PRIMARY}]\n")

    if not bookmark_manager.bookmarks:
        console.print(f"[{COLOR_WARNING}]No bookmarks to delete[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    display_bookmarks(bookmark_manager.bookmarks)

    msg = f"\n[{COLOR_WARNING}]Select bookmark to delete ('{KEY_BACK}' to back)[/{COLOR_WARNING}]"
    choice = Prompt.ask(msg, default=KEY_BACK)

    if choice.lower() == KEY_BACK:
        return

    try:
        idx = int(choice) - 1
        bookmark = bookmark_manager.get(idx)

        if bookmark:
            confirm = Prompt.ask(
                f"[{COLOR_WARNING}]Delete '{bookmark.name}'? (y/n)[/{COLOR_WARNING}]",
                default="n",
            )

            if confirm.lower() == "y":
                bookmark_manager.remove(idx)
                console.print(f"\n[{COLOR_SUCCESS}]✓ Bookmark deleted[/{COLOR_SUCCESS}]")
            else:
                console.print("\n[dim]Cancelled[/dim]")
        else:
            console.print(f"[{COLOR_ERROR}]Invalid selection[/{COLOR_ERROR}]")

    except ValueError:
        console.print(f"[{COLOR_ERROR}]Invalid input[/{COLOR_ERROR}]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def _clear_all_bookmarks() -> None:
    """Clear all bookmarks."""
    console.clear()
    console.print(f"[bold {COLOR_WARNING}]🧹 Clear All Bookmarks[/bold {COLOR_WARNING}]\n")

    if not bookmark_manager.bookmarks:
        console.print(f"[{COLOR_WARNING}]No bookmarks to clear[/{COLOR_WARNING}]")
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        return

    console.print(f"You have {len(bookmark_manager.bookmarks)} bookmarks.\n")

    msg = f"[{COLOR_WARNING}]Delete ALL bookmarks? (yes/no)[/{COLOR_WARNING}]"
    confirm = Prompt.ask(msg, default="no")

    if confirm.lower() == "yes":
        bookmark_manager.clear()
        console.print(f"\n[{COLOR_SUCCESS}]✓ All bookmarks cleared[/{COLOR_SUCCESS}]")
    else:
        console.print("\n[dim]Cancelled[/dim]")

    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
