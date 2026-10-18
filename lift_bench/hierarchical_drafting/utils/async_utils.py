import asyncio
import logging
import os
from dataclasses import dataclass
from functools import partial
from time import time

import tqdm
import tqdm.asyncio


async def _wrap_coroutines_with_progress_logging(
    coros, *, log_name: str = "Async Gather", log_level: int, log_interval: int, logger_name: str
):
    @dataclass
    class _ProgressObject:
        progress_bar: tqdm.tqdm = 0
        last_log_message_time: float = 0.0

    progress_obj = _ProgressObject(
        last_log_message_time=time(),
        progress_bar=tqdm.tqdm(total=len(coros), desc=log_name, file=open(os.devnull, "w")),
    )

    async def log_coro(coro):
        retval = await coro
        progress_obj.progress_bar.update(1)
        t = time()
        if t - progress_obj.last_log_message_time > log_interval:
            logging.getLogger(logger_name).log(log_level, str(progress_obj.progress_bar))
            progress_obj.last_log_message_time = t
        return retval

    return [log_coro(c) for c in coros]


async def gather_with_concurrency(
    max_concurrent: int,
    *coros,
    use_tqdm: bool = False,
    log_name: str | None = None,
    log_level: int = logging.DEBUG,
    log_interval: int = 10,
    logger_name: str = "alignment",
    tqdm_kwargs: dict = None,
):
    """asyncio.gather with a concurrency cap and optional tqdm/logging progress."""
    semaphore = asyncio.Semaphore(max_concurrent)
    if tqdm_kwargs is None:
        tqdm_kwargs = {}

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    if log_name:
        coros = await _wrap_coroutines_with_progress_logging(
            coros, log_name=log_name, log_level=log_level, log_interval=log_interval, logger_name=logger_name
        )

    gather_func = partial(tqdm.asyncio.tqdm_asyncio.gather, **tqdm_kwargs) if use_tqdm else asyncio.gather
    return await gather_func(*(sem_coro(c) for c in coros))
