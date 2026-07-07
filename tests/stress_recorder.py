#!/usr/bin/env python3
"""Stress + regression test for Recorder start/stop after the 2026-07-07
CoreAudio deadlock (PortAudio stop vs HAL notification, AB-BA on the HAL
mutex, froze the app + killed the event tap).

Two layers:
1. wedge simulation (deterministic): a stream whose stop() blocks forever
   must NOT block Recorder.stop() longer than the 3s abandon-timeout, and
   the already-captured frames must survive.
2. live cycles: N real microphone start/stop cycles, each guarded by a
   10s watchdog; a hanging cycle counts as failure.

Run: .venv/bin/python tests/stress_recorder.py [live_cycles]
Exit 0 = all clean.
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "localflow"))
from app import DEFAULT_CONFIG, Recorder  # noqa: E402


def test_wedged_stop():
    rec = Recorder(dict(DEFAULT_CONFIG))

    class Wedged:
        def stop(self):
            time.sleep(3600)

        def close(self):
            pass

    rec.stream = Wedged()
    rec.recording = True
    rec.frames = [np.zeros((1600, 1), dtype="float32")]
    t0 = time.time()
    audio = rec.stop()
    dt = time.time() - t0
    assert dt < 4.5, f"stop blocked {dt:.1f}s"
    assert audio is not None and len(audio) == 1600, "captured audio lost"
    assert rec.recording is False and rec.stream is None
    assert rec.wedged is True, "wedge must poison the recorder (app restarts on it)"
    return dt


def live_cycle(i, rec):
    assert rec.start(), f"start returned False (run {i})"
    time.sleep(0.4)
    audio = rec.stop()
    assert audio is not None and len(audio) > 1000, f"no audio (run {i})"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    dt = test_wedged_stop()
    print(f"wedge-recovery OK ({dt:.2f}s, limit 4.5s)")

    rec = Recorder(dict(DEFAULT_CONFIG))
    failures = 0
    for i in range(1, n + 1):
        box = {}

        def run(i=i, rec=rec):
            try:
                live_cycle(i, rec)
            except Exception as e:
                box["err"] = e

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(10)
        if th.is_alive():
            print(f"run {i}: HANG (>10s)")
            failures += 1
            rec = Recorder(dict(DEFAULT_CONFIG))  # abandon wedged recorder
        elif "err" in box:
            print(f"run {i}: FAIL {box['err']}")
            failures += 1
        else:
            print(f"run {i}: OK")

    print(f"{n - failures}/{n} live cycles clean")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
