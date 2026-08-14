"""ConcurrencyGate — bounds simultaneous entries; no dspy import."""

from __future__ import annotations

import threading
import time

import pytest

from gluomem.llm.concurrency import ConcurrencyGate


def test_gate_never_exceeds_limit() -> None:
    """N threads contend for a gate of C slots; peak concurrency must stay <= C
    and every thread must complete (the release path always runs)."""
    gate = ConcurrencyGate(3)
    lock = threading.Lock()
    active = 0
    peak = 0
    done = 0

    def worker() -> None:
        nonlocal active, peak, done
        with gate:
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)  # hold the slot so contention is real
            with lock:
                active -= 1
        with lock:
            done += 1

    threads = [threading.Thread(target=worker) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak <= 3, f"peak concurrency {peak} exceeded gate limit 3"
    assert peak >= 2, "gate should permit real concurrency, not serialize"
    assert done == 24  # all slots released, no leak


def test_gate_releases_on_exception() -> None:
    """A body that raises still releases its slot (no permanent slot leak)."""
    gate = ConcurrencyGate(1)
    with pytest.raises(RuntimeError):
        with gate:
            raise RuntimeError("boom")
    # If the slot leaked, this second acquire would block forever; a short-lived
    # thread proves it is immediately available.
    acquired = threading.Event()

    def reacquire() -> None:
        with gate:
            acquired.set()

    t = threading.Thread(target=reacquire)
    t.start()
    t.join(timeout=1.0)
    assert acquired.is_set(), "slot was not released after an exception"


def test_gate_rejects_bad_limit() -> None:
    with pytest.raises(ValueError, match="limit must be >= 1"):
        ConcurrencyGate(0)
