from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypeVar, Union

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

# Request timeout, rate limiting, 5xx gateway errors and 529 (provider overloaded).
_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 529})

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff on transient failures: the first retry waits `backoff_initial`, the n-th
    `backoff_factor * 2 ** (n - 1)`, capped at `backoff_max`. The defaults suit a judge call during evaluation.
    """

    retries: int = 10
    backoff_initial: float = 0.0
    backoff_factor: float = 1.5
    backoff_max: float = 120.0

    def backoff(self, failures: int) -> float:
        if failures <= 1:
            return self.backoff_initial
        return min(self.backoff_max, self.backoff_factor * 2 ** (failures - 1))


# Pipeline calls ride out long outages: 288 retries at the 300s cap span about a day.
PIPELINE_RETRY_POLICY = RetryPolicy(retries=288, backoff_initial=10, backoff_factor=2, backoff_max=300)


def _is_transient(error: Exception) -> bool:
    if isinstance(error, APIStatusError):
        return error.status_code in _TRANSIENT_STATUS_CODES
    return isinstance(error, APIConnectionError)  # timeouts and dropped connections


def _retry_after(error: Exception) -> Optional[float]:
    if not isinstance(error, APIStatusError):
        return None
    try:
        return float(error.response.headers["retry-after"])
    except KeyError, ValueError:
        return None


async def call_with_retries(call: Callable[[], Awaitable[T]], policy: RetryPolicy) -> T:
    failures = 0
    while True:
        try:
            return await call()
        except Exception as error:
            failures += 1
            if failures > policy.retries or not _is_transient(error):
                raise
            delay = _retry_after(error) or policy.backoff(failures)
            logger.warning("Model call failed (%s), retry %d/%d in %.1fs", error, failures, policy.retries, delay)
            await asyncio.sleep(delay)


def _get_client() -> AsyncOpenAI:
    """An OpenAI-compatible client configured by OPENAI_BASE_URL and OPENAI_API_KEY."""
    return AsyncOpenAI(max_retries=0)  # retries follow the RetryPolicy instead


@dataclass
class CompletionResponse:
    choices: list

    @property
    def completions(self) -> list[str]:
        return [choice.message.content for choice in self.choices]


async def generate_completions(
    model: str,
    prompts: list[Union[str, list[dict]]],
    max_tokens: int = 1024,
    temperature: float = 0.6,
    n: int = 1,
    max_concurrent: int = 20,
    retry_policy: RetryPolicy = PIPELINE_RETRY_POLICY,
    **kwargs,
) -> list[CompletionResponse]:
    client = _get_client()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _complete(prompt: Union[str, list[dict]]) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        async with semaphore:
            response = await call_with_retries(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                ),
                retry_policy,
            )
        return CompletionResponse(choices=response.choices)

    return list(await asyncio.gather(*[_complete(p) for p in prompts]))
