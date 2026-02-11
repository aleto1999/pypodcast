"""Stage 4: LLM-based role classification.

Uses LLM to classify extracted names as HOST, GUEST, or NEITHER
based on contextual information from transcripts and metadata.

Supports:
- OpenAI API (cloud)
- Local transformers models with CUDA support
- Heuristic fallback (no ML)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from podcast_conversations.speaker_labeling.types import (
    RoleClassification,
    SpeakerRole,
    ExtractedName,
)

logger = logging.getLogger(__name__)


CLASSIFICATION_PROMPT = """You are analyzing a podcast episode to identify speaker roles.

## Podcast Information
- **Podcast Name:** {podcast_name}
- **Episode Title:** {episode_title}

## Extracted Names
The following names were extracted from the episode:
{names_list}

## Transcript Introduction (first ~500 words)
{transcript_intro}

## Task
For each extracted name, classify their role as:
- **HOST**: The regular presenter/host of this podcast series
- **GUEST**: An invited guest for this specific episode
- **NEITHER**: A person mentioned but not actually speaking in the episode

Respond in JSON format only:
```json
{{
  "classifications": [
    {{
      "name": "Person Name",
      "role": "HOST|GUEST|NEITHER",
      "confidence": 0.0-1.0,
      "reasoning": "Brief explanation"
    }}
  ]
}}
```

Consider these signals:
- Hosts typically say "welcome to [show name]" or "I'm [name], this is [show]"
- Guests are introduced with phrases like "joining us today", "our guest is"
- The podcast name often contains the host's name
- Hosts speak first and last in most episodes
- NEITHER includes people discussed but not present (historical figures, celebrities mentioned in stories)

Return ONLY the JSON, no other text."""


def detect_cuda_available() -> bool:
    """Check if CUDA is available for local inference."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_memory_gb() -> float:
    """Get available GPU memory in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except ImportError:
        pass
    return 0.0


class LocalLLMRoleClassifier:
    """Uses local transformers model with CUDA for role classification."""

    # recommended models by GPU memory.
    MODEL_RECOMMENDATIONS = {
        "small": "microsoft/phi-2",  # ~5GB VRAM.
        "medium": "mistralai/Mistral-7B-Instruct-v0.2",  # ~14GB VRAM.
        "large": "meta-llama/Meta-Llama-3-8B-Instruct",  # ~16GB VRAM.
    }

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize local LLM classifier.

        Args:
            model_name: HuggingFace model name. If None, auto-selects based on GPU memory.
            device: Device to use ("auto", "cuda", "cuda:0", "cpu").
            torch_dtype: Data type ("auto", "float16", "bfloat16", "float32").
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            load_in_8bit: Use 8-bit quantization (reduces memory ~50%).
            load_in_4bit: Use 4-bit quantization (reduces memory ~75%).
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self._model = None
        self._tokenizer = None
        self._pipeline = None

    def _auto_select_model(self) -> str:
        """Auto-select model based on available GPU memory."""
        gpu_memory = get_gpu_memory_gb()
        logger.info(f"Detected GPU memory: {gpu_memory:.1f} GB")

        if gpu_memory >= 20:
            return self.MODEL_RECOMMENDATIONS["large"]
        elif gpu_memory >= 10:
            return self.MODEL_RECOMMENDATIONS["medium"]
        elif gpu_memory >= 4:
            return self.MODEL_RECOMMENDATIONS["small"]
        else:
            # fall back to smallest model with quantization.
            self.load_in_4bit = True
            return self.MODEL_RECOMMENDATIONS["small"]

    @property
    def pipeline(self):
        """Lazy-load the transformers pipeline."""
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "transformers and torch required for local LLM. "
                    "Install with: pip install transformers torch accelerate"
                )

            model_name = self.model_name or self._auto_select_model()
            logger.info(f"Loading local model: {model_name}")

            # configure dtype.
            if self.torch_dtype == "auto":
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32

            # configure quantization.
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # configure device.
            device_map = "auto" if self.device == "auto" else None
            device = None if self.device == "auto" else self.device

            # load model.
            model_kwargs = {
                "torch_dtype": dtype,
                "device_map": device_map,
                "trust_remote_code": True,
            }
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )

            # create pipeline.
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
                torch_dtype=dtype,
            )

            logger.info(f"Model loaded on device: {self._model.device}")

        return self._pipeline

    def classify_roles(
        self,
        podcast_name: str,
        episode_title: str,
        extracted_names: list[str],
        transcript_intro: str,
    ) -> list[RoleClassification]:
        """
        Classify extracted names into roles using local LLM.

        Args:
            podcast_name: Name of the podcast series.
            episode_title: Title of this episode.
            extracted_names: List of person names extracted from transcript.
            transcript_intro: First ~500 words of transcript.

        Returns:
            List of role classifications for each name.
        """
        if not extracted_names:
            return []

        # format names list.
        names_list = "\n".join(f"- {name}" for name in extracted_names)

        # build prompt.
        prompt = CLASSIFICATION_PROMPT.format(
            podcast_name=podcast_name,
            episode_title=episode_title,
            names_list=names_list,
            transcript_intro=transcript_intro[:2000],  # shorter for local models.
        )

        try:
            # generate response.
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                return_full_text=False,
            )
            response = outputs[0]["generated_text"]
            return self._parse_response(response, extracted_names)
        except Exception as e:
            logger.warning(f"Local LLM inference failed: {e}")
            return self._fallback_classification(podcast_name, extracted_names)

    def classify_batch(
        self,
        episodes: list[dict[str, Any]],
        batch_size: int = 4,
    ) -> list[list[RoleClassification]]:
        """
        Classify roles for multiple episodes with batching.

        Args:
            episodes: List of dicts with podcast_name, episode_title, names, intro.
            batch_size: Batch size for inference.

        Returns:
            List of classification lists, one per episode.
        """
        results = []

        # build all prompts.
        prompts = []
        for ep in episodes:
            names_list = "\n".join(f"- {name}" for name in ep.get("names", []))
            prompt = CLASSIFICATION_PROMPT.format(
                podcast_name=ep.get("podcast_name", ""),
                episode_title=ep.get("episode_title", ""),
                names_list=names_list,
                transcript_intro=ep.get("intro", "")[:2000],
            )
            prompts.append(prompt)

        # batch inference.
        try:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_episodes = episodes[i:i + batch_size]

                outputs = self.pipeline(
                    batch_prompts,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    return_full_text=False,
                    batch_size=len(batch_prompts),
                )

                for j, output in enumerate(outputs):
                    response = output[0]["generated_text"]
                    ep = batch_episodes[j]
                    classifications = self._parse_response(
                        response,
                        ep.get("names", []),
                    )
                    results.append(classifications)

        except Exception as e:
            logger.warning(f"Batch inference failed: {e}, falling back to sequential")
            for ep in episodes:
                classifications = self.classify_roles(
                    podcast_name=ep.get("podcast_name", ""),
                    episode_title=ep.get("episode_title", ""),
                    extracted_names=ep.get("names", []),
                    transcript_intro=ep.get("intro", ""),
                )
                results.append(classifications)

        return results

    def _parse_response(
        self,
        response: str,
        expected_names: list[str],
    ) -> list[RoleClassification]:
        """Parse LLM response into structured classifications."""
        classifications = []

        try:
            # extract JSON from response.
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                for item in data.get("classifications", []):
                    role_str = item.get("role", "NEITHER").upper()
                    role = SpeakerRole.NEITHER
                    if role_str == "HOST":
                        role = SpeakerRole.HOST
                    elif role_str == "GUEST":
                        role = SpeakerRole.GUEST

                    classifications.append(RoleClassification(
                        name=item.get("name", ""),
                        role=role,
                        confidence=float(item.get("confidence", 0.5)),
                        reasoning=item.get("reasoning", ""),
                    ))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse local LLM response: {e}")
            return self._fallback_classification("", expected_names)

        return classifications

    def _fallback_classification(
        self,
        podcast_name: str,
        names: list[str],
    ) -> list[RoleClassification]:
        """Fallback heuristic classification when LLM fails."""
        classifications = []
        host_assigned = False
        podcast_lower = podcast_name.lower().replace("_", " ")

        for name in names:
            name_lower = name.lower()

            if not host_assigned and name_lower in podcast_lower:
                classifications.append(RoleClassification(
                    name=name,
                    role=SpeakerRole.HOST,
                    confidence=0.6,
                    reasoning="Name appears in podcast title (fallback)",
                ))
                host_assigned = True
            else:
                classifications.append(RoleClassification(
                    name=name,
                    role=SpeakerRole.GUEST,
                    confidence=0.4,
                    reasoning="Default guest classification (fallback)",
                ))

        return classifications

    def unload_model(self):
        """Unload model to free GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            del self._pipeline
            self._model = None
            self._tokenizer = None
            self._pipeline = None

            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Model unloaded, GPU memory freed")


class LLMRoleClassifier:
    """Uses LLM to classify speaker roles."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        """
        Initialize LLM classifier.

        Args:
            model: LLM model to use.
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var).
            temperature: Sampling temperature.
            max_retries: Maximum retry attempts.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required for LLM classification. Install with: pip install openai")
        return self._client

    def classify_roles(
        self,
        podcast_name: str,
        episode_title: str,
        extracted_names: list[str],
        transcript_intro: str,
    ) -> list[RoleClassification]:
        """
        Classify extracted names into roles using LLM.

        Args:
            podcast_name: Name of the podcast series.
            episode_title: Title of this episode.
            extracted_names: List of person names extracted from transcript.
            transcript_intro: First ~500 words of transcript.

        Returns:
            List of role classifications for each name.
        """
        if not extracted_names:
            return []

        if not self.api_key:
            logger.warning("No API key provided, falling back to heuristic classification")
            return self._fallback_classification(podcast_name, extracted_names)

        # format names list.
        names_list = "\n".join(f"- {name}" for name in extracted_names)

        # build prompt.
        prompt = CLASSIFICATION_PROMPT.format(
            podcast_name=podcast_name,
            episode_title=episode_title,
            names_list=names_list,
            transcript_intro=transcript_intro[:3000],  # limit context size.
        )

        # call LLM with retries.
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=1000,
                )
                content = response.choices[0].message.content
                return self._parse_response(content, extracted_names)
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return self._fallback_classification(podcast_name, extracted_names)

        return self._fallback_classification(podcast_name, extracted_names)

    def classify_batch(
        self,
        episodes: list[dict[str, Any]],
    ) -> list[list[RoleClassification]]:
        """
        Classify roles for multiple episodes.

        Args:
            episodes: List of dicts with podcast_name, episode_title, names, intro.

        Returns:
            List of classification lists, one per episode.
        """
        results = []
        for ep in episodes:
            classifications = self.classify_roles(
                podcast_name=ep.get("podcast_name", ""),
                episode_title=ep.get("episode_title", ""),
                extracted_names=ep.get("names", []),
                transcript_intro=ep.get("intro", ""),
            )
            results.append(classifications)
        return results

    def _parse_response(
        self,
        response: str,
        expected_names: list[str],
    ) -> list[RoleClassification]:
        """Parse LLM response into structured classifications."""
        classifications = []

        try:
            # extract JSON from response.
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                for item in data.get("classifications", []):
                    role_str = item.get("role", "NEITHER").upper()
                    role = SpeakerRole.NEITHER
                    if role_str == "HOST":
                        role = SpeakerRole.HOST
                    elif role_str == "GUEST":
                        role = SpeakerRole.GUEST

                    classifications.append(RoleClassification(
                        name=item.get("name", ""),
                        role=role,
                        confidence=float(item.get("confidence", 0.5)),
                        reasoning=item.get("reasoning", ""),
                    ))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_classification("", expected_names)

        return classifications

    def _fallback_classification(
        self,
        podcast_name: str,
        names: list[str],
    ) -> list[RoleClassification]:
        """
        Fallback heuristic classification when LLM unavailable.

        Uses simple rules:
        - First name is likely the host if podcast name matches
        - Other names are likely guests
        """
        classifications = []
        host_assigned = False
        podcast_lower = podcast_name.lower().replace("_", " ")

        for name in names:
            name_lower = name.lower()

            # check if name appears in podcast name.
            if not host_assigned and name_lower in podcast_lower:
                classifications.append(RoleClassification(
                    name=name,
                    role=SpeakerRole.HOST,
                    confidence=0.6,
                    reasoning="Name appears in podcast title (heuristic)",
                ))
                host_assigned = True
            else:
                classifications.append(RoleClassification(
                    name=name,
                    role=SpeakerRole.GUEST,
                    confidence=0.4,
                    reasoning="Default guest classification (heuristic)",
                ))

        return classifications


class HeuristicRoleClassifier:
    """Rule-based role classifier that doesn't require LLM."""

    def classify_roles(
        self,
        podcast_name: str,
        episode_title: str,
        extracted_names: list[ExtractedName],
        transcript_intro: str,
        known_host: str | None = None,
    ) -> list[RoleClassification]:
        """
        Classify roles using heuristic rules.

        Args:
            podcast_name: Podcast name.
            episode_title: Episode title.
            extracted_names: List of extracted names.
            transcript_intro: Introduction text.
            known_host: Known host name (from metadata).

        Returns:
            List of role classifications.
        """
        classifications = []
        podcast_lower = podcast_name.lower().replace("_", " ")
        episode_lower = episode_title.lower()

        # track if host assigned.
        host_assigned = False

        for name in extracted_names:
            name_str = name.name
            name_lower = name_str.lower()

            # rule 1: known host.
            if known_host and name_lower == known_host.lower():
                classifications.append(RoleClassification(
                    name=name_str,
                    role=SpeakerRole.HOST,
                    confidence=0.95,
                    reasoning="Matches known host from metadata",
                ))
                host_assigned = True
                continue

            # rule 2: name in podcast title.
            if not host_assigned and name_lower in podcast_lower:
                classifications.append(RoleClassification(
                    name=name_str,
                    role=SpeakerRole.HOST,
                    confidence=0.8,
                    reasoning="Name appears in podcast title",
                ))
                host_assigned = True
                continue

            # rule 3: name in episode title (likely guest).
            if name_lower in episode_lower:
                classifications.append(RoleClassification(
                    name=name_str,
                    role=SpeakerRole.GUEST,
                    confidence=0.75,
                    reasoning="Name appears in episode title",
                ))
                continue

            # rule 4: self-introduction pattern.
            intro_patterns = ["i'm " + name_lower, "my name is " + name_lower]
            if any(p in transcript_intro.lower() for p in intro_patterns):
                # could be host or guest, lean towards guest unless first.
                role = SpeakerRole.HOST if not host_assigned else SpeakerRole.GUEST
                classifications.append(RoleClassification(
                    name=name_str,
                    role=role,
                    confidence=0.7,
                    reasoning="Self-introduction detected",
                ))
                if role == SpeakerRole.HOST:
                    host_assigned = True
                continue

            # default: likely guest.
            classifications.append(RoleClassification(
                name=name_str,
                role=SpeakerRole.GUEST,
                confidence=0.5,
                reasoning="Default classification",
            ))

        return classifications


def create_role_classifier(
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
    use_local: bool = False,
    local_model: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> LLMRoleClassifier | LocalLLMRoleClassifier | HeuristicRoleClassifier:
    """
    Factory function to create role classifier.

    Args:
        use_llm: Whether to use LLM classifier (API-based).
        model: OpenAI model to use.
        use_local: Whether to use local transformers model with CUDA.
        local_model: HuggingFace model name for local inference.
        load_in_4bit: Use 4-bit quantization for local model.
        load_in_8bit: Use 8-bit quantization for local model.

    Returns:
        Configured classifier.

    Priority:
        1. Local CUDA model (if use_local=True and CUDA available)
        2. OpenAI API (if use_llm=True and API key available)
        3. Heuristic fallback
    """
    # option 1: local CUDA model.
    if use_local:
        if detect_cuda_available():
            logger.info("Using local CUDA model for role classification")
            return LocalLLMRoleClassifier(
                model_name=local_model,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
        else:
            logger.warning("CUDA not available, falling back to API or heuristics")

    # option 2: OpenAI API.
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        logger.info(f"Using OpenAI API model: {model}")
        return LLMRoleClassifier(model=model)

    # option 3: heuristic fallback.
    logger.info("Using heuristic role classification")
    return HeuristicRoleClassifier()


def create_local_classifier(
    model_name: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> LocalLLMRoleClassifier:
    """
    Factory function to create local CUDA classifier.

    Args:
        model_name: HuggingFace model name. Auto-selects if None.
        load_in_4bit: Use 4-bit quantization.
        load_in_8bit: Use 8-bit quantization.

    Returns:
        LocalLLMRoleClassifier configured for available GPU.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not detect_cuda_available():
        raise RuntimeError("CUDA is not available. Install PyTorch with CUDA support.")

    return LocalLLMRoleClassifier(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
