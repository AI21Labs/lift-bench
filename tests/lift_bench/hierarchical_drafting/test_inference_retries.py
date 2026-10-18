"""Model calls retry transient failures with exponential backoff."""

import httpx
import openai
import pytest

from lift_bench.hierarchical_drafting.utils import inference
from lift_bench.hierarchical_drafting.utils.inference import (
    PIPELINE_RETRY_POLICY,
    RetryPolicy,
    call_with_retries,
)


def _status_error(status: int, headers: dict | None = None) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://gateway.test/v1/chat/completions")
    return openai.APIStatusError("error", response=httpx.Response(status, headers=headers, request=request), body=None)


@pytest.fixture
def sleeps(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(inference.asyncio, "sleep", fake_sleep)
    return slept


def _flaky(errors: list[Exception]):
    async def call():
        if errors:
            raise errors.pop(0)
        return "ok"

    return call


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
async def test_transient_status_codes_are_retried(status, sleeps):
    assert await call_with_retries(_flaky([_status_error(status)]), RetryPolicy()) == "ok"
    assert len(sleeps) == 1


@pytest.mark.usefixtures("sleeps")
async def test_dropped_connections_are_retried():
    request = httpx.Request("POST", "https://gateway.test")
    errors = [openai.APIConnectionError(request=request), openai.APITimeoutError(request=request)]

    assert await call_with_retries(_flaky(errors), RetryPolicy()) == "ok"


@pytest.mark.parametrize("error", [_status_error(400), _status_error(404), ValueError("bug")])
async def test_other_failures_are_raised_at_once(error, sleeps):
    with pytest.raises(type(error)):
        await call_with_retries(_flaky([error]), RetryPolicy())
    assert sleeps == []


async def test_the_last_failure_is_raised_once_retries_run_out(sleeps):
    errors = [_status_error(503) for _ in range(4)]

    with pytest.raises(openai.APIStatusError):
        await call_with_retries(_flaky(errors), RetryPolicy(retries=3))
    assert len(sleeps) == 3


async def test_a_retry_after_header_sets_the_wait(sleeps):
    await call_with_retries(_flaky([_status_error(429, {"retry-after": "7"})]), RetryPolicy())

    assert sleeps == [7.0]


def test_backoff_starts_at_the_initial_wait_then_doubles_up_to_the_cap():
    assert [PIPELINE_RETRY_POLICY.backoff(n) for n in range(1, 10)] == [10, 4, 8, 16, 32, 64, 128, 256, 300]
    assert [RetryPolicy().backoff(n) for n in range(1, 4)] == [0, 3.0, 6.0]
