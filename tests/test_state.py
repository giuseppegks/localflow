#!/usr/bin/env python3
"""Regression test for DictationState: atomic transitions, single-winner
finish, and PROCESSING self-expiry (a stuck busy flag used to eat every
hotkey until app restart).

Run: .venv/bin/python tests/test_state.py
Exit 0 = all clean.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "localflow"))
from app import DictationState  # noqa: E402


def test_transitions():
    s = DictationState()
    assert s.get() == (DictationState.IDLE, None)
    assert s.try_begin_recording("toggle")
    assert s.get() == (DictationState.RECORDING, "toggle")
    assert not s.try_begin_recording("hold"), "double start must lose"
    assert s.try_finish()
    assert not s.try_finish(), "second finish must lose"
    assert s.get()[0] == DictationState.PROCESSING
    s.to_idle()
    assert s.get() == (DictationState.IDLE, None)


def test_expiry():
    s = DictationState()
    s.EXPIRE_S = 0.2
    assert s.try_begin_recording("toggle") and s.try_finish()
    assert s.get()[0] == DictationState.PROCESSING
    time.sleep(0.3)
    assert s.get()[0] == DictationState.IDLE, "expired PROCESSING must reset"
    assert s.try_begin_recording("hold"), "hotkey must work again after expiry"


def test_hammer():
    """8 threads race full cycles; invariant: per cycle exactly one
    start-winner and one finish-winner, never more."""
    s = DictationState()
    errors = []

    def worker():
        for _ in range(200):
            if s.try_begin_recording("toggle"):
                if not s.try_finish():
                    errors.append("winner of start lost finish")
                s.to_idle()
            else:
                # losers may only observe, never transition
                s.get()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
        assert not t.is_alive(), "deadlock in state machine"
    assert not errors, errors


def main():
    test_transitions()
    print("transitions OK")
    test_expiry()
    print("expiry OK")
    test_hammer()
    print("hammer OK (8 threads x 200 cycles, no deadlock, single winners)")
    sys.exit(0)


if __name__ == "__main__":
    main()
