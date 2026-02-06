"""Configuration for LLM annotation questions and prompts."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnnotationConfig:
    """Configuration for LLM-based annotation."""

    # target groups for hate speech classification.
    target_groups: list[str] = field(
        default_factory=lambda: [
            "racial and ethnic minorities",
            "religious minorities",
            "women",
            "lgbtq+ population",
            "disabled (physical and/or mental) population",
            "immigrant population",
        ]
    )

    # hate speech types.
    hate_speech_types: list[str] = field(
        default_factory=lambda: [
            "Threat to culture or identity",
            "Threat to survival or physical security",
            "Vilification or villainization",
            "Explicit dehumanization",
        ]
    )

    # system prompt for the LLM.
    system_prompt: str = (
        "You are an expert content analyst specializing in detecting "
        "hate speech and harmful content in podcast transcripts. You provide accurate, "
        "objective annotations based on the content of utterances.\n\n"
        "CRITICAL: You must respond ONLY with valid JSON. No explanations, no markdown, "
        "no text before or after. "
        "Use double quotes for all strings. Use true/false (lowercase) for booleans. "
        "Use null for empty values."
    )

    # user prompt template.
    user_prompt_template: str = """Analyze the following utterance from a podcast transcript.

UTTERANCE:
"{utterance}"

Respond with ONLY this JSON format (no other text):
{{"has_hate_speech": false, "has_advertisement": false, "target_group": null, \
"hate_speech_type": null, "main_topic": "topic"}}

Fields:
- "has_hate_speech": true or false
- "has_advertisement": true or false
- "target_group": null OR one of {target_groups}
- "hate_speech_type": null OR one of {hate_speech_types}
- "main_topic": brief 1-10 word description

Rules:
- Set target_group and hate_speech_type to null if has_hate_speech is false
- Output ONLY the JSON object"""

    # model configuration.
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 256
    timeout: int = 60

    # retry configuration.
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    def get_user_prompt(self, utterance: str) -> str:
        """Generate user prompt for a specific utterance."""
        return self.user_prompt_template.format(
            utterance=utterance,
            target_groups=json.dumps(self.target_groups),
            hate_speech_types=json.dumps(self.hate_speech_types),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "target_groups": self.target_groups,
            "hate_speech_types": self.hate_speech_types,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def load_questions_config(config_path: Path | None = None) -> AnnotationConfig:
    """
    Load annotation configuration from file or use defaults.

    If config_path is provided, loads custom configuration.
    Otherwise, returns default configuration.
    """
    if config_path is None:
        return AnnotationConfig()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        if config_path.suffix == ".json":
            data = json.load(f)
        else:
            # for markdown files, just use defaults (questions are embedded).
            return AnnotationConfig()

    return AnnotationConfig(
        target_groups=data.get("target_groups", AnnotationConfig.target_groups),
        hate_speech_types=data.get("hate_speech_types", AnnotationConfig.hate_speech_types),
        system_prompt=data.get("system_prompt", AnnotationConfig.system_prompt),
        user_prompt_template=data.get(
            "user_prompt_template", AnnotationConfig.user_prompt_template
        ),
        model_name=data.get("model_name", AnnotationConfig.model_name),
        temperature=data.get("temperature", AnnotationConfig.temperature),
        max_tokens=data.get("max_tokens", AnnotationConfig.max_tokens),
    )
