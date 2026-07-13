#!/usr/bin/env python3
"""LocalFlow capture worker: audio capture isolated in a child process.

PortAudio's stream.stop() can deadlock against CoreAudio's HAL mutex
(AB-BA), and once that happens every Pa_OpenStream in the process blocks
forever. Running capture here means a wedge poisons only this child:
the parent SIGKILLs it and respawns a fresh one in ~0.3s while the STT
model stays loaded in the parent.

Protocol (stdin, line-based commands): START | STOP | QUIT
Protocol (stdout, binary packets):     4-byte big-endian length + 1 type
byte + payload. Type 'A' = raw f32le mono audio chunk, type 'J' = UTF-8
JSON control message ({"event": "started"|"stopped", ...}).

On a wedged STOP this process logs to stderr and exits with code 70;
the parent already holds every frame (streamed live), so nothing is lost.
Env LOCALFLOW_TEST_WEDGE=1 makes STOP block forever (test hook).
"""

import json
import os
import queue
import struct
import sys
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
WEDGE_EXIT = 70

_out_lock = threading.Lock()


def send(kind, payload):
    data = kind + payload
    with _out_lock:
        sys.stdout.buffer.write(struct.pack(">I", len(data)) + data)
        sys.stdout.buffer.flush()


def send_json(obj):
    send(b"J", json.dumps(obj).encode("utf-8"))


class Capture:
    def __init__(self):
        self.stream = None
        # The RT audio callback must never block on the stdout pipe:
        # chunks go through a queue, a writer thread drains it.
        self.chunks = queue.Queue()
        threading.Thread(target=self._writer, daemon=True).start()

    def _writer(self):
        while True:
            chunk = self.chunks.get()
            send(b"A", chunk.tobytes())

    def _callback(self, indata, frame_count, time_info, status):
        self.chunks.put(indata[:, 0].copy())

    def start(self):
        if self.stream is not None:
            return
        try:
            dev = sd.query_devices(kind="input")
            device_name = dev.get("name", "?")
            device_rate = dev.get("default_samplerate", 0)
        except Exception:
            device_name, device_rate = "?", 0
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._callback)
        self.stream.start()
        send_json({"event": "started", "device": device_name,
                   "rate": device_rate})

    def stop(self):
        stream, self.stream = self.stream, None
        if stream is None:
            send_json({"event": "stopped"})
            return

        def _close():
            if os.environ.get("LOCALFLOW_TEST_WEDGE") == "1":
                time.sleep(3600)  # simulate the CoreAudio deadlock
            stream.stop()
            stream.close()

        t = threading.Thread(target=_close, daemon=True)
        t.start()
        t.join(3.0)
        if t.is_alive():
            print("capture_worker: stream stop wedged (CoreAudio deadlock), "
                  "exiting for respawn", file=sys.stderr, flush=True)
            os._exit(WEDGE_EXIT)
        send_json({"event": "stopped"})


def main():
    cap = Capture()
    for line in sys.stdin:
        cmd = line.strip()
        try:
            if cmd == "START":
                cap.start()
            elif cmd == "STOP":
                cap.stop()
            elif cmd == "QUIT":
                cap.stop()
                return
        except Exception as e:
            print(f"capture_worker error ({cmd}): {e}",
                  file=sys.stderr, flush=True)
            send_json({"event": "error", "error": str(e)})


if __name__ == "__main__":
    main()
