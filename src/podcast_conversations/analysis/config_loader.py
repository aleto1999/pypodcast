"""configuration loader for analysis settings."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KeywordAnalysisSettings(BaseModel):
    """settings for keyword analysis."""

    case_sensitive: bool = False
    partial_matches: bool = True
    min_word_length: int = 3
    context_window: int = 100
    export_formats: list[str] = Field(default_factory=lambda: ["json", "csv", "txt"])
    include_speaker_info: bool = True
    include_timestamps: bool = True
    confidence_threshold: float = 0.7


class KeywordCategory(BaseModel):
    """a category of keywords with confidence tier."""

    high_confidence: list[str] = Field(default_factory=list)
    exploratory: list[str] = Field(default_factory=list)


class KeywordAnalysisConfig(BaseModel):
    """keyword analysis configuration with categorized keywords."""

    # flat list for backward compatibility.
    keywords: list[str] = Field(default_factory=list)
    # categorized keywords by target group.
    categories: dict[str, KeywordCategory] = Field(default_factory=dict)
    settings: KeywordAnalysisSettings = Field(default_factory=KeywordAnalysisSettings)

    def get_all_keywords(self) -> list[str]:
        """get flat list of all keywords from all categories."""
        all_keywords = list(self.keywords)
        for category in self.categories.values():
            all_keywords.extend(category.high_confidence)
            all_keywords.extend(category.exploratory)
        # remove duplicates while preserving order.
        seen = set()
        unique = []
        for kw in all_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique.append(kw)
        return unique

    def get_keywords_by_category(self) -> dict[str, list[str]]:
        """get keywords organized by category name."""
        result = {}
        for cat_name, cat_data in self.categories.items():
            result[cat_name] = cat_data.high_confidence + cat_data.exploratory
        return result

    def get_high_confidence_keywords(self) -> list[str]:
        """get only high-confidence keywords from all categories."""
        keywords = []
        for category in self.categories.values():
            keywords.extend(category.high_confidence)
        return list(set(keywords))

    def get_keyword_category(self, keyword: str) -> str | None:
        """look up which category a keyword belongs to."""
        keyword_lower = keyword.lower()
        for cat_name, cat_data in self.categories.items():
            if keyword_lower in [k.lower() for k in cat_data.high_confidence]:
                return cat_name
            if keyword_lower in [k.lower() for k in cat_data.exploratory]:
                return cat_name
        return None

    def get_keyword_confidence(self, keyword: str) -> str | None:
        """get confidence tier for a keyword (high_confidence or exploratory)."""
        keyword_lower = keyword.lower()
        for cat_data in self.categories.values():
            if keyword_lower in [k.lower() for k in cat_data.high_confidence]:
                return "high_confidence"
            if keyword_lower in [k.lower() for k in cat_data.exploratory]:
                return "exploratory"
        return None


class AnalysisConfig:
    """load and manage analysis configuration from YAML file."""

    def __init__(self, config_path: Path):
        """
        initialize configuration loader.

        Args:
            config_path: path to YAML configuration file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"config file not found: {config_path}")

        self._config_data: dict[str, Any] = {}
        self.keyword_analysis: KeywordAnalysisConfig | None = None

        self._load_config()

    def _load_config(self) -> None:
        """load and parse YAML configuration file."""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                self._config_data = yaml.safe_load(f)

            # parse keyword analysis section.
            if "keyword_analysis" in self._config_data:
                keyword_config = self._config_data["keyword_analysis"]

                # parse categories if present.
                categories = {}
                if "categories" in keyword_config:
                    for cat_name, cat_data in keyword_config["categories"].items():
                        categories[cat_name] = KeywordCategory(
                            high_confidence=cat_data.get("high_confidence", []),
                            exploratory=cat_data.get("exploratory", []),
                        )

                self.keyword_analysis = KeywordAnalysisConfig(
                    keywords=keyword_config.get("keywords", []),
                    categories=categories,
                    settings=KeywordAnalysisSettings(**keyword_config.get("settings", {})),
                )

                total_keywords = len(self.keyword_analysis.get_all_keywords())
                num_categories = len(categories)
                if num_categories > 0:
                    logger.info(
                        f"loaded {total_keywords} keywords in {num_categories} categories "
                        f"from {self.config_path}"
                    )
                else:
                    logger.info(
                        f"loaded {total_keywords} keywords from {self.config_path}"
                    )
            else:
                logger.warning(
                    f"no keyword_analysis section found in {self.config_path}"
                )

        except yaml.YAMLError as e:
            raise RuntimeError(f"failed to parse YAML config {self.config_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"failed to load config {self.config_path}: {e}") from e

    def get_keywords(self) -> list[str]:
        """
        get list of all keywords for analysis.

        Returns:
            list of keyword strings from all categories.
        """
        if self.keyword_analysis is None:
            return []
        return self.keyword_analysis.get_all_keywords()

    def get_keywords_by_category(self) -> dict[str, list[str]]:
        """
        get keywords organized by category.

        Returns:
            dict mapping category names to keyword lists.
        """
        if self.keyword_analysis is None:
            return {}
        return self.keyword_analysis.get_keywords_by_category()

    def get_keyword_category(self, keyword: str) -> str | None:
        """look up which category a keyword belongs to."""
        if self.keyword_analysis is None:
            return None
        return self.keyword_analysis.get_keyword_category(keyword)

    def get_keyword_confidence(self, keyword: str) -> str | None:
        """get confidence tier for a keyword."""
        if self.keyword_analysis is None:
            return None
        return self.keyword_analysis.get_keyword_confidence(keyword)

    def get_settings(self) -> KeywordAnalysisSettings:
        """
        get keyword analysis settings.

        Returns:
            keyword analysis settings object.
        """
        if self.keyword_analysis is None:
            return KeywordAnalysisSettings()
        return self.keyword_analysis.settings

    def get_raw_config(self) -> dict[str, Any]:
        """
        get raw configuration dictionary.

        Returns:
            full configuration data.
        """
        return self._config_data
