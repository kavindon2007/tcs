"""
agent/llm_client.py — Local Ollama wrapper with retry, caching, and JSON parsing.

All LLM/VLM calls go through this module.  No stage imports the model SDK
directly — this keeps API coupling in one place and makes swapping models
or providers a config-only change.

Backend: Ollama REST API (http://localhost:11434)
  - Text calls:   POST /api/generate  { model, prompt, stream: false }
  - Vision calls: POST /api/generate  { model, prompt, images: [base64], stream: false }
"""

import base64
import hashlib
import json
import re
import time
from pathlib import Path

import requests

from agent import config


class LLMClient:
    """Thin wrapper around the local Ollama REST API.

    Responsibilities:
    - Route text-only vs. image+text calls to /api/generate.
    - Retry with exponential backoff on transient errors.
    - Cache responses to disk (keyed by SHA-256 of model+prompt+image_path)
      so re-runs or retries of the full pipeline don't re-process images.
    - Extract and parse JSON from model responses, tolerating markdown fences.
    """

    def __init__(self, cache: bool = True):
        self._use_cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call_text(self, model: str, prompt: str) -> str:
        """Send a text-only prompt and return the raw response string."""
        return self._call(model=model, prompt=prompt, image_path=None)

    def call_with_image(self, model: str, prompt: str, image_path: str | Path) -> str:
        """Send a prompt with one image and return the raw response string."""
        return self._call(model=model, prompt=prompt, image_path=Path(image_path))

    def parse_json(self, raw: str) -> dict:
        """Extract a JSON object from a model response.

        Handles:
        - Plain JSON (as instructed in prompts)
        - Markdown code fences (```json ... ```)
        - JSON embedded in surrounding prose (fallback regex)
        """
        text = raw.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: find the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not parse JSON from model response "
            f"(first 300 chars): {raw[:300]!r}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(
        self,
        model: str,
        prompt: str,
        image_path: Path | None,
    ) -> str:
        cache_key = self._cache_key(model, prompt, image_path)

        if self._use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        response_text = self._call_with_retry(model, prompt, image_path)

        if self._use_cache:
            self._cache_set(cache_key, response_text)

        return response_text

    def _call_with_retry(
        self,
        model: str,
        prompt: str,
        image_path: Path | None,
    ) -> str:
        delay = config.RETRY_BASE_DELAY
        last_error: Exception | None = None

        for attempt in range(config.MAX_RETRIES):
            try:
                # Honour inter-call delay BEFORE every real API call
                # (not on cache hits — those bypass this path entirely)
                if config.INTER_CALL_DELAY_SECONDS > 0:
                    time.sleep(config.INTER_CALL_DELAY_SECONDS)
                return self._call_once(model, prompt, image_path)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < config.MAX_RETRIES - 1:
                    print(
                        f"[llm_client] attempt {attempt + 1} failed "
                        f"({type(exc).__name__}: {exc}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2

        raise RuntimeError(
            f"All {config.MAX_RETRIES} attempts failed. "
            f"Last error: {last_error}"
        ) from last_error

    def _call_once(
        self,
        model: str,
        prompt: str,
        image_path: Path | None,
    ) -> str:
        """Call the local Ollama REST API.

        Endpoint: POST {OLLAMA_BASE_URL}/api/generate
        For vision calls, the image is base64-encoded and passed in the
        'images' list field (supported by all multimodal Ollama models).
        """
        url = f"{config.OLLAMA_BASE_URL}/api/generate"
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        if image_path is not None:
            image_bytes = Path(image_path).read_bytes()
            payload["images"] = [base64.b64encode(image_bytes).decode("utf-8")]

        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["response"]

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(model: str, prompt: str, image_path: Path | None) -> str:
        content = f"{model}|{prompt}|{str(image_path) if image_path else ''}"
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def _cache_get(key: str) -> str | None:
        cache_file = config.CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)["response"]
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    @staticmethod
    def _cache_set(key: str, response: str) -> None:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = config.CACHE_DIR / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump({"response": response}, f)
