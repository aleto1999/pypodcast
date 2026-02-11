"""MLX-based LLM annotator for Apple Silicon using llm CLI."""

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MLXModelConfig:
    """Configuration for MLX models."""

    model_name: str
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9


class MLXAnnotator:
    """
    LLM annotator using MLX models via llm CLI.

    Uses Simon Willison's llm CLI tool with MLX plugin for running
    quantized models efficiently on Apple Silicon.

    Optimized for:
    - Apple Silicon (M1/M2/M3/M4)
    - 4-bit quantized models (e.g., Llama-3.3-70B-Instruct-4bit)
    - Memory-efficient inference
    - Local execution without API calls
    """

    DEFAULT_MODEL = "mlx-community/Llama-3.3-70B-Instruct-4bit"

    def __init__(
        self,
        model_name: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        """
        initialize MLX annotator.

        Args:
            model_name: MLX model identifier (default: Llama-3.3-70B-Instruct-4bit).
            max_tokens: maximum tokens to generate.
            temperature: sampling temperature (0.0-1.0).
            top_p: nucleus sampling threshold.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # find llm command - prefer from current environment.
        self.llm_cmd = self._find_llm_command()

        # verify llm CLI is available.
        if not self.llm_cmd:
            raise RuntimeError(
                "llm CLI not found. Install with: uv sync --extra macos"
            )

        # verify model is available.
        if not self._check_model_available():
            logger.warning(
                f"model {self.model_name} not found in llm models. "
                f"it will be downloaded on first use."
            )

        logger.info(f"initialized MLX annotator with model: {self.model_name}")

    def _find_llm_command(self) -> str | None:
        """
        find llm command in current environment or system PATH.

        Returns:
            path to llm command or None if not found.
        """
        # check if llm is in PATH (works for uv-managed environments).
        llm_path = shutil.which("llm")
        if llm_path:
            return llm_path

        # if not found, return None.
        return None

    def _check_llm_cli(self) -> bool:
        """check if llm CLI is installed."""
        try:
            result = subprocess.run(
                [self.llm_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_model_available(self) -> bool:
        """check if model is available in llm models list."""
        try:
            result = subprocess.run(
                [self.llm_cmd, "models", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return self.model_name in result.stdout
            return False
        except subprocess.SubprocessError:
            return False

    def annotate(
        self,
        text: str,
        prompt_template: str,
        json_schema: dict | None = None,
    ) -> dict[str, Any]:
        """
        annotate text using MLX model.

        Args:
            text: text to annotate.
            prompt_template: prompt template with {text} placeholder.
            json_schema: optional JSON schema for structured output validation.

        Returns:
            dict with annotation results.

        Raises:
            RuntimeError: if annotation fails.
        """
        # build prompt from template.
        prompt = prompt_template.format(text=text)

        # prepare llm command.
        cmd = [
            self.llm_cmd,
            "prompt",
            prompt,
            "-m", self.model_name,
            "--option", f"max_tokens={self.max_tokens}",
            "--option", f"temperature={self.temperature}",
            "--option", f"top_p={self.top_p}",
        ]

        try:
            # run llm command.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout.
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"llm command failed: {result.stderr}"
                )

            # parse JSON response.
            response = result.stdout.strip()
            try:
                annotation = json.loads(response)
            except json.JSONDecodeError:
                # if response is not valid JSON, try to extract JSON.
                annotation = self._extract_json(response)

            # validate against schema if provided.
            if json_schema:
                self._validate_json(annotation, json_schema)

            return annotation

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "annotation timed out after 300 seconds"
            )
        except Exception as e:
            raise RuntimeError(
                f"annotation failed: {e}"
            ) from e

    def annotate_batch(
        self,
        texts: list[str],
        prompt_template: str,
        json_schema: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        annotate multiple texts.

        Note: MLX via llm CLI doesn't support true batching,
        so this processes items sequentially.

        Args:
            texts: list of texts to annotate.
            prompt_template: prompt template with {text} placeholder.
            json_schema: optional JSON schema for validation.

        Returns:
            list of annotation results.
        """
        results = []
        for text in texts:
            try:
                result = self.annotate(text, prompt_template, json_schema)
                results.append(result)
            except RuntimeError as e:
                logger.error(f"failed to annotate text: {e}")
                # add empty result to maintain alignment.
                results.append({})

        return results

    def _extract_json(self, response: str) -> dict[str, Any]:
        """
        extract JSON from response that may contain markdown or other text.

        Args:
            response: raw response text.

        Returns:
            extracted JSON object.

        Raises:
            ValueError: if no valid JSON found.
        """
        # try to find JSON in markdown code block.
        import re

        # look for ```json ... ``` blocks.
        json_blocks = re.findall(
            r"```json\s*\n(.*?)\n```",
            response,
            re.DOTALL,
        )

        if json_blocks:
            try:
                return json.loads(json_blocks[0])
            except json.JSONDecodeError:
                pass

        # look for any JSON object in the text.
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # if no JSON found, raise error.
        raise ValueError(
            f"could not extract valid JSON from response: {response[:200]}"
        )

    def _validate_json(self, data: dict, schema: dict) -> None:
        """
        validate JSON data against schema.

        Args:
            data: JSON data to validate.
            schema: JSON schema for validation.

        Raises:
            ValueError: if validation fails.
        """
        # basic validation - check required fields.
        if "required" in schema:
            required_fields = schema["required"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(
                        f"missing required field: {field}"
                    )

        # check field types if properties defined.
        if "properties" in schema:
            for field, field_schema in schema["properties"].items():
                if field in data:
                    expected_type = field_schema.get("type")
                    if expected_type:
                        self._check_type(data[field], expected_type, field)

    def _check_type(self, value: Any, expected_type: str, field_name: str) -> None:
        """
        check if value matches expected JSON schema type.

        Args:
            value: value to check.
            expected_type: expected type from schema.
            field_name: name of field for error messages.

        Raises:
            ValueError: if type doesn't match.
        """
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        python_type = type_mapping.get(expected_type)
        if python_type and not isinstance(value, python_type):
            raise ValueError(
                f"field {field_name} has wrong type: "
                f"expected {expected_type}, got {type(value).__name__}"
            )
