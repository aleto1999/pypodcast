"""taxonomy loading and validation."""

from pathlib import Path
from typing import Any

import yaml


def load_taxonomy() -> dict[str, Any]:
    """load and validate taxonomy configuration.

    returns a dict with:
        - target_groups: list of {id, name}
        - strategies: list of {id, name, description}
        - examples: list of {target_group_id, strategy_id, items}
    """
    config_path = Path("config/taxonomy.yml")

    if not config_path.exists():
        raise FileNotFoundError(f"taxonomy config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    taxonomy = config.get("taxonomy", {})

    if not taxonomy:
        raise ValueError("taxonomy configuration is empty")

    # extract components.
    target_groups = taxonomy.get("target_groups", [])
    strategies = taxonomy.get("othering_strategies", [])
    examples = taxonomy.get("examples", [])

    if not target_groups or not strategies:
        raise ValueError("taxonomy must contain target_groups and othering_strategies")

    # count total combinations.
    total_combinations = len(target_groups) * len(strategies)
    total_examples = sum(len(ex.get("items", [])) for ex in examples)

    print(
        f"✓ loaded {len(target_groups)} target groups, "
        f"{len(strategies)} strategies, "
        f"{total_examples} examples"
    )

    return {
        "target_groups": target_groups,
        "strategies": strategies,
        "examples": examples,
    }
