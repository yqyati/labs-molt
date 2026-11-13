import errno
import logging
import re
import time


logger = logging.getLogger(__name__)


def is_storage_full(exc):
    nested = getattr(exc, "failures", None)
    errors = [failure[0] for failure in nested.values()] if nested else [exc]
    return all(
        (isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT))
        or bool(re.search(r"\[Errno (28|122)\]|no space left on device|disk quota exceeded", str(error), re.I))
        for error in errors
    )


def retry_on_storage_full(operation, target, *, retry_interval=30):
    """Retry a write forever when storage is full; propagate all other errors."""
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if not is_storage_full(exc):
                raise
            attempt += 1
            logger.error(
                "Storage full while saving %s; retry %d in %ds after space is available: %s",
                target,
                attempt,
                retry_interval,
                exc,
            )
            time.sleep(retry_interval)
