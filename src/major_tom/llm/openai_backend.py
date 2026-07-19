"""OpenAI LLM backend implementation."""

import logging
import time
import base64
import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from major_tom.llm.base import EmbeddingResponse, LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """Wraps OpenAI API with standardized responses."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("OpenAI client not installed. Please run `pip install openai`.")

        self.request_timeout_s = float(os.environ.get("MAJOR_TOM_LLM_TIMEOUT_S", "120"))
        self.max_retries = int(os.environ.get("MAJOR_TOM_LLM_RETRIES", "1"))
        self.retry_backoff_s = float(os.environ.get("MAJOR_TOM_LLM_RETRY_BACKOFF_S", "1.5"))
        
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL")
        )

    def _call_with_retry(self, operation_name: str, call):
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return call()
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
    ) -> LLMResponse:
        messages = []
        
        if images is None:
            messages.append({"role": "user", "content": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
            for img_bytes in images:
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            messages.append({"role": "user", "content": content})

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "timeout": self.request_timeout_s,
        }
        
        if format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        elif format is not None:
             # Try to map other formats if possible or ignore
             pass

        if options:
            # Map common options
            if "temperature" in options:
                kwargs["temperature"] = options["temperature"]
            if "top_p" in options:
                kwargs["top_p"] = options["top_p"]
            if "num_predict" in options:
                kwargs["max_tokens"] = options["num_predict"]
            # ... add more mappings as needed

        start = time.monotonic()
        try:
            completion = self._call_with_retry(
                f"OpenAI generate[{model}]",
                lambda: self.client.chat.completions.create(**kwargs),
            )
            latency = (time.monotonic() - start) * 1000

            response_content = completion.choices[0].message.content
            prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0
            completion_tokens = completion.usage.completion_tokens if completion.usage else 0
            total_tokens = completion.usage.total_tokens if completion.usage else 0

            return LLMResponse(
                text=response_content or "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model,
                latency_ms=latency,
                raw=completion.model_dump(),
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def embed(self, model: str, prompt: str) -> EmbeddingResponse:
        start = time.monotonic()
        try:
            # Explicitly allow the configured model, fallback to a sensible default if model is "nomic"
            # But since we updated the config to text-embedding-3-large, it will be passed here directly.
            if "nomic" in model and "openai" not in model:
               # Just a safety fallback if old config is used
                model = "text-embedding-3-large"
                
            res = self._call_with_retry(
                f"OpenAI embed[{model}]",
                lambda: self.client.embeddings.create(
                    model=model,
                    input=prompt,
                    timeout=self.request_timeout_s,
                ),
            )
            latency = (time.monotonic() - start) * 1000

            return EmbeddingResponse(
                vector=res.data[0].embedding,
                prompt_tokens=res.usage.prompt_tokens,
                model=model,
                latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"OpenAI Embedding error: {e}")
            raise
