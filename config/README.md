# Configuration Files

This directory contains YAML configuration files for the podcast conversations analysis pipeline.

## Overview

| File | Purpose |
|------|---------|
| `classifiers.yaml` | Transformer models for utterance classification |
| `keyword_analysis_config.yaml` | Keywords and settings for keyword matching analysis |
| `llm_annotation_config.yaml` | LLM annotation questions and prompts |
| `taxonomy.yml` | Target groups and othering strategies for embeddings analysis |

---

## classifiers.yaml

Defines transformer-based classification models for the utterance classification pipeline.

### Structure

```yaml
models:
  - name: hate_speech_detection
    description: "A model for binary hate speech detection."
    model_repo: "cardiffnlp/twitter-roberta-base-hate-latest"
    labels: ["NOT-HATE", "HATE"]

  - name: fine_grained_hate_speech_detection
    description: "A model for fine-grained hate speech detection."
    model_repo: "GroNLP/hateBERT"
    labels: ["acceptable", "inappropriate", "offensive", "violent"]
```

### Fields

- **name**: Unique identifier for the classifier
- **description**: Human-readable description
- **model_repo**: HuggingFace model repository path
- **labels**: Expected output labels from the model

### Usage

```bash
uv run python scripts/classify_utterances.py \
  --config config/classifiers.yaml \
  --transcripts-dir outputs/transcripts_with_speakers
```

---

## keyword_analysis_config.yaml

Defines keywords and settings for regex-based keyword matching analysis.

### Structure

```yaml
keyword_analysis:
  keywords:
    - "immigrant"
    - "immigration"
    - "minority"
    - "ethnic"
    - "refugee"
    # ... 730+ keywords

  settings:
    case_sensitive: false          # Case-insensitive matching
    partial_matches: true          # "immigrant" matches "immigrants"
    context_window: 100            # Characters before/after match
    include_speaker_info: true     # Include speaker labels
    include_timestamps: true       # Include start/end times
```

### Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `case_sensitive` | bool | `false` | Match case-sensitively |
| `partial_matches` | bool | `true` | Match keyword substrings |
| `context_window` | int | `100` | Characters of context to extract |
| `include_speaker_info` | bool | `true` | Add speaker labels to results |
| `include_timestamps` | bool | `true` | Add timing info to results |

### Usage

Keyword analysis is integrated into the analysis pipeline:

```bash
uv run python scripts/run_analysis_pipeline.py \
  --transcripts-dir outputs/transcripts_postprocessed \
  --keywords-config config/keyword_analysis_config.yaml
```

---

## llm_annotation_config.yaml

Defines questions and prompts for LLM-based content annotation.

### Questions

The LLM annotator asks the following questions for each utterance:

1. **Hate Speech Detection**: Is there any hate speech in the utterance? (yes/no)
2. **Advertisement Detection**: Does this utterance contain any advertisements? (yes/no)
3. **Target Group Identification**: If hate speech detected, which group is targeted:
   - Racial and ethnic minorities
   - Religious minorities
   - Women
   - LGBTQ+ population
   - Disabled (physical and/or mental) population
   - Immigrant population
4. **Hate Speech Type Classification**:
   - Threat to culture or identity
   - Threat to survival or physical security
   - Vilification or villainization
   - Explicit dehumanization
5. **Topic Extraction**: What is the main topic of this utterance? (1 word to 1 sentence)

### Usage

```bash
uv run python scripts/annotate_with_llm.py \
  --config config/llm_annotation_config.yaml \
  --transcripts-dir outputs/transcripts_with_speakers
```

---

## taxonomy.yml

Defines the dehumanization taxonomy for semantic embeddings analysis.

### Structure

```yaml
target_groups:
  - id: immigrants
    name: Immigrants
  - id: political_opponents
    name: Political Opponents
  # ... more groups

strategies:
  - id: threat_to_culture
    name: Threat to Culture
    description: "Framing group as threats to cultural identity"
  # ... more strategies

examples:
  - target_group_id: immigrants
    strategy_id: threat_to_culture
    items:
      - "they don't share our values"
      - "they refuse to assimilate"
      # ... more examples
```

### Components

**Target Groups (6):**
- Immigrants
- Political opponents
- Religious minorities
- Racial/ethnic minorities
- LGBTQ+ population
- Women

**Othering Strategies (4):**
- Threat to culture or identity
- Threat to survival or physical security
- Vilification or villainization
- Explicit dehumanization

### Usage

```bash
uv run python scripts/analyze_embeddings.py
# Interactive mode will load taxonomy.yml automatically
```

---

## Adding New Configuration

### Custom Classifiers

Add a new model to `classifiers.yaml`:

```yaml
models:
  # ... existing models
  - name: custom_classifier
    description: "My custom classifier"
    model_repo: "organization/model-name"
    labels: ["LABEL_0", "LABEL_1"]
```

### Custom Keywords

Edit `keyword_analysis_config.yaml`:

```yaml
keyword_analysis:
  keywords:
    # Add your keywords
    - "custom_keyword"
    - "another_keyword"
```

### Custom Taxonomy

Edit `taxonomy.yml` to add new target groups or strategies:

```yaml
target_groups:
  - id: custom_group
    name: Custom Target Group

strategies:
  - id: custom_strategy
    name: Custom Strategy
    description: "Description of the strategy"

examples:
  - target_group_id: custom_group
    strategy_id: custom_strategy
    items:
      - "example phrase 1"
      - "example phrase 2"
```

---

## Related Documentation

- [Analysis Package](../src/podcast_conversations/analysis/README.md) - Keyword analysis implementation
- [Embeddings Method](../src/podcast_conversations/embeddings_method/README.md) - Taxonomy usage
- [Utterance Classification](../src/podcast_conversations/utterance_classification/README.md) - Classifier configuration
- [LLM Annotation](../src/podcast_conversations/llm_annotation/README.md) - Annotation pipeline
