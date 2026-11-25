import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from tallyjudge.sdk._tasks import consume_task, detached_tasks

logger = logging.getLogger(__name__)


def run_sync[T](
    factory: Callable[[], Coroutine[Any, Any, T]],
    async_name: str,
    *,
    cleanup_timeout: float = 5.0,
) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            f"An event loop is already running; use await {async_name}()"
        )
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(factory())
    finally:
        detached = detached_tasks(loop)
        pending = asyncio.all_tasks(loop) - detached
        for task in pending:
            task.cancel()
        if pending:
            done, unfinished = loop.run_until_complete(
                asyncio.wait(pending, timeout=cleanup_timeout)
            )
            for task in done:
                consume_task(task)
            if unfinished:
                logger.warning(
                    "Synchronous loop shutdown left %d unfinished tasks",
                    len(unfinished),
                )
        finalizers = [
            loop.create_task(loop.shutdown_asyncgens()),
            loop.create_task(loop.shutdown_default_executor()),
        ]
        done, unfinished = loop.run_until_complete(
            asyncio.wait(finalizers, timeout=cleanup_timeout)
        )
        for task in done:
            consume_task(task)
        for task in unfinished:
            task.cancel()
        loop.close()
        detached.clear()
