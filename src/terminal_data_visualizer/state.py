"""Session state persistence."""

from __future__ import annotations

import json
from typing import Any

from terminal_data_visualizer.config import REMEMBER_LAST_SELECTION, STATE_FILE


class SessionState:
    """Manages persistent session state."""

    def __init__(self) -> None:
        """Initialize session state."""
        self.state_file = STATE_FILE
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load state from file."""
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        """Save state to file."""
        if not REMEMBER_LAST_SELECTION:
            return

        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except OSError:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self.state[key] = value
        self._save()

    def get_last_podcast(self) -> str | None:
        """Get last selected podcast."""
        return self.get("last_podcast")

    def set_last_podcast(self, podcast: str) -> None:
        """Set last selected podcast."""
        self.set("last_podcast", podcast)

    def get_last_episode(self) -> str | None:
        """Get last selected episode."""
        return self.get("last_episode")

    def set_last_episode(self, episode: str) -> None:
        """Set last selected episode."""
        self.set("last_episode", episode)

    def clear(self) -> None:
        """Clear all state."""
        self.state = {}
        self._save()


# global session state instance.
session_state = SessionState()
