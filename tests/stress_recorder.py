#!/usr/bin/env python3
"""Stress + regression test for the process-isolated capture path
(CaptureClient + capture_worker.py), successor of the 2026-07-07
in-process wedge test.

Layers:
1. wedge simulation (deterministic, via LOCALFLOW_TEST_WEDGE=1 in the
   child): stop() must return the captured frames < 4.5s, AND — the new
   headline assertion — the very NEXT start() must record live audio
   again on a respawned child. A wedge no longer poisons the process.
2. kill -9 mid-recording: partial audio survives, auto-respawn works.
3. live cycles: N real microphone start/stop cycles, each guarded by a
   10s watchdog; a hanging cycle counts as failure.

Run: .venv/bin/python tests/stress_recorder.py [live_cycles]
Exit 0 = all clean.
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "localflow"))
from app import DEFAULT_CONFIG, CaptureClient  # noqa: E402


def record_briefly(rec, seconds=0.6):
    assert rec.start(), "start returned False"
    time.sleep(seconds)
    return rec.stop()


def test_wedged_stop_then_recover():
    os.environ["LOCALFLOW_TEST_WEDGE"] = "1"
    try:
        rec = CaptureClient(dict(DEFAULT_CONFIG))
        assert rec.start(), "start failed"
        time.sleep(0.6)
        t0 = time.time()
        audio = rec.stop()
        dt = time.time() - t0
        assert dt < 4.5, f"stop blocked {dt:.1f}s"
        assert audio is not None and len(audio) > 1000, "captured audio lost"
        assert rec.wedge_count == 1, "wedge not detected"
    finally:
        os.environ.pop("LOCALFLOW_TEST_WEDGE", None)

    # Headline: the next dictation works without an app restart.
    time.sleep(1.0)  # let the background respawn land
    audio2 = record_briefly(rec)
    assert audio2 is not None and len(audio2) > 1000, \
        "recording after wedge failed — wedge still poisons the process"
    rec.shutdown()
    return dt


def test_kill9_recovery():
    rec = CaptureClient(dict(DEFAULT_CONFIG))
    assert rec.start(), "start failed"
    time.sleep(0.6)
    rec._proc.kill()  # simulate a hard child death mid-recording
    time.sleep(0.2)
    audio = rec.stop()
    assert audio is not None and len(audio) > 1000, "partial audio lost"
    time.sleep(1.0)
    audio2 = record_briefly(rec)
    assert audio2 is not None and len(audio2) > 1000, \
        "recording after kill -9 failed"
    rec.shutdown()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    dt = test_wedged_stop_then_recover()
    print(f"wedge-recovery OK ({dt:.2f}s stop, next start records again)")

    test_kill9_recovery()
    print("kill-9-recovery OK (partial audio kept, respawn records again)")

    rec = CaptureClient(dict(DEFAULT_CONFIG))
    failures = 0
    for i in range(1, n + 1):
        box = {}

        def run(i=i, rec=rec):
            try:
                audio = record_briefly(rec, 0.4)
                assert audio is not None and len(audio) > 1000, \
                    f"no audio (run {i})"
            except Exception as e:
                box["err"] = e

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(10)
        if th.is_alive():
            print(f"run {i}: HANG (>10s)")
            failures += 1
            rec.shutdown()
            rec = CaptureClient(dict(DEFAULT_CONFIG))
        elif "err" in box:
            print(f"run {i}: FAIL {box['err']}")
            failures += 1
        else:
            print(f"run {i}: OK")
    rec.shutdown()

    print(f"{n - failures}/{n} live cycles clean")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
