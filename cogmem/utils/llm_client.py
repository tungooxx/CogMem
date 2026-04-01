import os
import time

from openai import OpenAI, RateLimitError


class LLMClient:
    PROVIDER_BASES = {
        "ollama": "http://localhost:11434/v1",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
    }

    def __init__(
        self,
        provider: str,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.api_base = api_base or self.PROVIDER_BASES[provider]
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "none")
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        self._retry_base_delay = 2.0
        self._max_retries = 5

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0,
    ) -> str:
        for attempt in range(self._max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except RateLimitError:
                wait = self._retry_base_delay * (2**attempt)
                time.sleep(wait)
            except Exception:
                raise
        raise RuntimeError(f"Max retries ({self._max_retries}) exceeded")
