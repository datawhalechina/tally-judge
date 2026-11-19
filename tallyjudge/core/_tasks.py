"""Ownership of tasks that outlive their caller on a particular event loop."""

import asyncio
import logging
from typing import Any

from tallyjudge.core.errors import UsageScopeClosed

logger = logging.getLogger(__name__)


def detached_tasks(loop: asyncio.AbstractEventLoop) -> set[asyncio.Task[Any]]:
    """Retain abandoned tasks until they finish or their owning loop closes."""
    # A process-wide set would retain closed loops through Task.get_loop().
    return vars(loop).setdefault("_tallyjudge_detached_tasks", set())


def consume_task(task: asyncio.Task[Any]) -> None:
    """Retrieve the exception from a finished background task."""
    if not task.cancelled():
        error = task.exception()
        if error is not None and not isinstance(error, UsageScopeClosed):
            logger.error(
                "Background task failed after cleanup",
                exc_info=(type(error), error, error.__traceback__),
            )


def detach_task(task: asyncio.Task[Any]) -> None:
    """Keep a late task alive and observe its terminal exception."""
    detached = detached_tasks(task.get_loop())
    detached.add(task)
    task.add_done_callback(detached.discard)
    task.add_done_callback(consume_task)
