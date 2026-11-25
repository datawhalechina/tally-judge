import asyncio
import logging
from typing import Any

from tallyjudge.sdk.errors import UsageScopeClosed

logger = logging.getLogger(__name__)


def detached_tasks(loop: asyncio.AbstractEventLoop) -> set[asyncio.Task[Any]]:
    return vars(loop).setdefault("_tallyjudge_detached_tasks", set())


def consume_task(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        error = task.exception()
        if error is not None and not isinstance(error, UsageScopeClosed):
            logger.error(
                "Background task failed after cleanup",
                exc_info=(type(error), error, error.__traceback__),
            )


def detach_task(task: asyncio.Task[Any]) -> None:
    detached = detached_tasks(task.get_loop())
    detached.add(task)
    task.add_done_callback(detached.discard)
    task.add_done_callback(consume_task)
