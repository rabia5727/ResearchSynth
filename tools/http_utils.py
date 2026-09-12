"""Shared HTTP helper: enforces a hard wall-clock timeout around a GET
request, on top of requests' own timeout parameter.

Why this exists: a real hang was observed in production where a search
request stayed stuck for minutes, well past its intended `timeout=20`.
requests/urllib3's connect+read timeout doesn't necessarily bound every
failure mode on every platform - DNS resolution in particular can stall
outside of it. Running the request in a worker thread and enforcing our
own outer deadline guarantees a caller can never hang indefinitely, no
matter what's happening underneath.

Trade-off: if the hard timeout fires, the abandoned request keeps running
in its worker thread until it finishes or its own timeout fires - Python
can't forcibly kill a thread. The thread pool is bounded so this can't
grow without limit even if hangs happen repeatedly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import requests

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="http-get")


def get_with_hard_timeout(
    url: str,
    *,
    params: dict | None = None,
    timeout: int = 20,
    hard_timeout: int | None = None,
) -> requests.Response:
    """requests.get() that can never block a caller longer than
    hard_timeout seconds, even if requests' own timeout fails to bound a
    stall. Raises requests.Timeout on either kind of timeout, so callers
    only need to handle one exception type."""
    hard_timeout = hard_timeout if hard_timeout is not None else timeout + 10

    future = _executor.submit(
        requests.get, url, params=params, timeout=timeout,
        headers={"User-Agent": "ResearchSynth/0.1"},
    )
    try:
        return future.result(timeout=hard_timeout)
    except FutureTimeoutError as exc:
        raise requests.Timeout(f"Hard timeout after {hard_timeout}s waiting for {url}") from exc
