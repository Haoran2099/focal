"""Ollama LLM backend implementation."""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

import ollama

from major_tom.llm.base import EmbeddingResponse, LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """Wraps ollama.generate() and ollama.embeddings() with standardized responses."""

    def __init__(self):
        self.request_timeout_s = float(os.environ.get("MAJOR_TOM_LLM_TIMEOUT_S", "120"))
        self.max_retries = int(os.environ.get("MAJOR_TOM_LLM_RETRIES", "1"))
        self.retry_backoff_s = float(os.environ.get("MAJOR_TOM_LLM_RETRY_BACKOFF_S", "1.5"))

        self.client = None
        host = os.environ.get("OLLAMA_HOST")
        try:
            # Newer ollama clients accept httpx client kwargs such as timeout.
            self.client = ollama.Client(host=host, timeout=self.request_timeout_s)
        except TypeError:
            # Backward compatibility for older ollama package versions.
            self.client = ollama.Client(host=host)

    def _run_with_timeout(self, operation_name: str, fn):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            try:
                return future.result(timeout=self.request_timeout_s)
            except FuturesTimeoutError as e:
                future.cancel()
                raise TimeoutError(
                    f"{operation_name} exceeded timeout ({self.request_timeout_s:.1f}s)"
                ) from e

    def _call_with_retry(self, operation_name: str, fn):
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                sleep_s = self.retry_backoff_s * (2 ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                    operation_name,
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                    sleep_s,
                )
                time.sleep(sleep_s)
        raise last_exc if last_exc else RuntimeError(f"{operation_name} failed with unknown error")

    def _generate_once(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is not None:
            return self.client.generate(**kwargs)
        return ollama.generate(**kwargs)

    def _embed_once(self, model: str, prompt: str) -> Dict[str, Any]:
        if self.client is not None:
            return self.client.embeddings(model=model, prompt=prompt)
        return ollama.embeddings(model=model, prompt=prompt)

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[bytes]] = None,
        format: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if images is not None:
            kwargs["images"] = images
        if format is not None:
            kwargs["format"] = format
        if options is not None:
            kwargs["options"] = options
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        if think is not None:
            kwargs["think"] = think

        start = time.monotonic()
        res = self._call_with_retry(
            f"Ollama generate[{model}]",
            lambda: self._run_with_timeout(
                f"Ollama generate[{model}]",
                lambda: self._generate_once(kwargs),
            ),
        )
        latency = (time.monotonic() - start) * 1000

        prompt_tokens = res.get("prompt_eval_count", 0)
        completion_tokens = res.get("eval_count", 0)

        return LLMResponse(
            text=res["response"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model,
            latency_ms=latency,
            raw=res,
        )

    def embed(self, model: str, prompt: str) -> EmbeddingResponse:
        start = time.monotonic()
        res = self._call_with_retry(
            f"Ollama embed[{model}]",
            lambda: self._run_with_timeout(
                f"Ollama embed[{model}]",
                lambda: self._embed_once(model, prompt),
            ),
        )
        latency = (time.monotonic() - start) * 1000

        vector = res.get("embedding", [])
        # Estimate tokens: ~4 chars per token for English
        estimated_tokens = max(1, len(prompt) // 4)

        return EmbeddingResponse(
            vector=vector,
            prompt_tokens=estimated_tokens,
            model=model,
            latency_ms=latency,
        )
