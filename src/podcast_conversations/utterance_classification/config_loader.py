"""configuration loader for classifier models."""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """configuration for a single classifier model."""

    name: str
    description: str
    model_repo: str
    labels: list[str]


class ClassifierConfig:
    """loads and manages classifier configuration from YAML file."""

    def __init__(self, config_path: Path):
        """
        initialize configuration loader.

        Args:
            config_path: path to classifiers YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.models = self._load_config()

    def _load_config(self) -> list[ModelConfig]:
        """
        load classifier models from YAML configuration.

        Returns:
            list of ModelConfig objects.

        Raises:
            FileNotFoundError: if config file doesn't exist.
            ValueError: if config format is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"config file not found: {self.config_path}")

        try:
            with open(self.config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data or "models" not in config_data:
                raise ValueError("config must contain 'models' key")

            models = []
            for model_data in config_data["models"]:
                model = ModelConfig(
                    name=model_data["name"],
                    description=model_data["description"],
                    model_repo=model_data["model_repo"],
                    labels=model_data["labels"],
                )
                models.append(model)

            logger.info(f"loaded {len(models)} model configurations")
            return models

        except yaml.YAMLError as e:
            raise ValueError(f"invalid YAML format: {e}") from e
        except KeyError as e:
            raise ValueError(f"missing required field in config: {e}") from e

    def get_models(self) -> list[ModelConfig]:
        """get list of configured models."""
        return self.models

    def get_model_by_name(self, name: str) -> ModelConfig | None:
        """get model configuration by name."""
        for model in self.models:
            if model.name == name:
                return model
        return None
