#!/usr/bin/env python3
"""LocalFlow: private, fully local voice dictation for macOS.

Hold Right-Option (or toggle with Ctrl+Alt+Space), speak, release.
Audio -> whisper.cpp (local) -> Ollama cleanup (local) -> pasted at cursor.
Nothing leaves the machine.
"""

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests
import rumps
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Controller, Key

import ctypes

import Foundation
import objc
from AppKit import (NSBackingStoreBuffered, NSColor, NSFont, NSImage,
                    NSImageView, NSPanel, NSScreen, NSTextField, NSView,
                    NSWindowStyleMaskBorderless,
                    NSWindowStyleMaskNonactivatingPanel, NSWorkspace,
                    NSWorkspaceDidWakeNotification,
                    NSWorkspaceWillSleepNotification)
from PyObjCTools import AppHelper


def boost_thread_qos():
    """Promote the calling thread to USER_INTERACTIVE so macOS schedules it
    on performance cores. Menu-bar apps default lower and App Nap makes
    inference ~6x slower (measured 3s -> 19.5s)."""
    try:
        libc = ctypes.CDLL(None)
        libc.pthread_set_qos_class_self_np(0x21, 0)  # QOS_CLASS_USER_INTERACTIVE
    except Exception:
        pass

# GUI-launched apps miss the Homebrew PATH; parakeet needs ffmpeg from there.
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

# GUI-launched apps also run with an ASCII locale, which makes every
# open() without encoding= (e.g. parakeet reading its config) crash on
# non-ASCII bytes. Force UTF-8 for the whole process.
import locale  # noqa: E402
try:
    locale.setlocale(locale.LC_CTYPE, "en_US.UTF-8")
except locale.Error:
    pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
DICT_PATH = ROOT / "dictionary.txt"
LOG_PATH = ROOT / "localflow.log"
STDERR_LOG = ROOT / "localflow.stderr.log"
LOG_MAX_BYTES = 512 * 1024

# The .app bundle we are running from (None in dev mode). Needed to
# relaunch ourselves after a CoreAudio wedge.
APP_BUNDLE = next((str(p) for p in Path(sys.executable).parents
                   if p.suffix == ".app"), None)

# GUI-launched (open -a): stdout/stderr go nowhere, so interpreter
# tracebacks vanish. Route both into a file. Dev mode keeps the terminal.
if APP_BUNDLE:
    try:
        if STDERR_LOG.exists() and STDERR_LOG.stat().st_size > LOG_MAX_BYTES:
            STDERR_LOG.replace(STDERR_LOG.with_suffix(".log.1"))
        _stderr_f = open(STDERR_LOG, "ab", buffering=0)
        os.dup2(_stderr_f.fileno(), 1)
        os.dup2(_stderr_f.fileno(), 2)
    except Exception:
        pass

DEFAULT_CONFIG = {
    "hold_key": "alt_r",
    "key_mode": "toggle",  # "toggle": tap to start/stop; "hold": push-to-talk
    "toggle_hotkey": "<ctrl>+<alt>+<space>",
    "language": "auto",
    # LLM cleanup off by default: qwen resident + parakeet resident thrashes
    # 8 GB unified memory (measured: 1s -> 106s STT). Python post-processing
    # (fillers, dictionary names) covers the common cases for free.
    "cleanup": False,
    "ollama_model": "qwen2.5:3b",
    "ollama_url": "http://127.0.0.1:11434",
    "stt_engine": "parakeet",  # "parakeet" (fast) or "whisper" (dict-prompt)
    "parakeet_model": "mlx-community/parakeet-tdt-0.6b-v3",
    "whisper_port": 8178,
    "whisper_model": str(ROOT / "models" / "ggml-medium-q5_0.bin"),
    "whisper_server_bin": "/opt/homebrew/bin/whisper-server",
    "whisper_flags": ["-bs", "1", "-fa"],
    "sample_rate": 16000,
    "max_seconds": 180,
    "sounds": True,
    "cleanup_min_words": 6,
    # After >1h idle + wake, run one silent clip through the model so the
    # compressed-out weights page back in off the user's critical path.
    "rewarm_on_wake": True,
}

SAMPLE_RATE = 16000

LANG_NAMES = {"de": "German", "german": "German",
              "nl": "Dutch", "dutch": "Dutch",
              "en": "English", "english": "English"}

CLEANUP_SYSTEM_PROMPT = (
    "You clean up dictated text. Fix punctuation and capitalization. "
    "Remove filler words (um, uh, äh, ähm, ehm, im as filler, nou ja, "
    "dus as filler, like as filler). "
    "Fix obvious speech-recognition errors. {lang_clause}"
    "NEVER translate. Do NOT add content. "
    "Do NOT answer questions contained in the text. "
    "Keep everything else word-for-word. {names_clause}"
    "Output ONLY the cleaned text, no quotes, no commentary."
)

# Cheap stopword-based language detection for the cleanup clause
# (parakeet does not report a language, unlike whisper).
LANG_MARKERS = {
    "de": {"und", "ich", "nicht", "das", "ist", "mit", "für", "ein", "eine",
           "der", "die", "morgen", "bitte", "aber", "auch", "wir", "noch"},
    "nl": {"en", "ik", "niet", "het", "met", "voor", "een", "de", "dat",
           "maar", "ook", "wordt", "naar", "nog", "wel", "graag", "moet"},
    "en": {"and", "the", "not", "with", "for", "that", "but", "also", "to",
           "of", "it", "we", "you", "please", "tomorrow", "have", "will"},
}


# Standalone filler tokens that are safe to strip in any of the three
# languages. Deliberately NOT "um"/"im"/"eh" — real words in DE/NL.
FILLERS = {"ähm", "äh", "ehm", "uhm", "uh", "hmm", "mhm", "ähm.", "ähm,"}


def post_process(text, dictionary_words):
    """Free, instant cleanup: strip fillers, fix proper nouns via fuzzy
    match against dictionary.txt. Replaces the LLM pass by default."""
    import difflib
    names = [w.strip() for w in dictionary_words.split(",") if w.strip()]
    out = []
    for raw in text.split():
        core = raw.strip(".,!?;:")
        if core.lower() in FILLERS or core.lower().strip(".,") in FILLERS:
            continue
        if names and core and core[0].isupper() and core not in names:
            match = difflib.get_close_matches(core, names, n=1, cutoff=0.8)
            if match:
                raw = raw.replace(core, match[0])
        out.append(raw)
    cleaned = " ".join(out)
    # A stripped leading filler can leave ", rest..." behind.
    cleaned = cleaned.lstrip(",. ")
    return cleaned[:1].upper() + cleaned[1:] if cleaned else text


def detect_lang(text):
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    scores = {code: len(words & markers)
              for code, markers in LANG_MARKERS.items()}
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] == 0 or ranked[0] == ranked[1]:
        return ""
    return best


def log(msg):
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > LOG_MAX_BYTES:
            LOG_PATH.replace(LOG_PATH.with_suffix(".log.1"))
    except Exception:
        pass
    with open(LOG_PATH, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except Exception as e:
            log(f"config parse error, using defaults: {e}")
    else:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return cfg


def load_dictionary():
    if DICT_PATH.exists():
        words = [w.strip() for w in DICT_PATH.read_text().splitlines()
                 if w.strip() and not w.startswith("#")]
        return ", ".join(words)
    return ""


def play_sound(name):
    subprocess.Popen(["afplay", f"/System/Library/Sounds/{name}.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


WORKER_PATH = Path(__file__).resolve().parent / "capture_worker.py"

# The capture child MUST run on the venv interpreter. Inside the .app
# bundle sys.executable is py2app's launcher, which has no venv
# site-packages: the child then dies on `import numpy` and every
# recording fails (hit live 2026-07-13).
_VENV_PY = ROOT / ".venv" / "bin" / "python"
CHILD_PY = str(_VENV_PY) if _VENV_PY.exists() else sys.executable


class CaptureClient:
    """Drop-in replacement for the old in-process Recorder: audio capture
    runs in a killable child process (capture_worker.py). PortAudio's
    stop() can deadlock against CoreAudio's HAL mutex and poison every
    later Pa_OpenStream in the process — in the child that costs a ~0.3s
    SIGKILL+respawn instead of a full app relaunch, and the dictation
    survives because frames stream live into the parent."""

    START_TIMEOUT = 4.0
    STOP_TIMEOUT = 3.0
    MAX_CHILD_AGE = 15 * 60  # stale HAL view after idle: respawn first

    def __init__(self, cfg):
        self.cfg = cfg
        self.frames = []
        self.recording = False
        self.level = 0.0
        self.on_chunk = None  # live feed into the streaming STT engine
        self.device_name = "?"
        self.device_rate = 0
        self.open_ms = -1
        self.wedge_count = 0
        self._spawn_lock = threading.Lock()
        self._proc = None
        self._spawned_at = 0.0
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._error = None
        self._max_samples = int(SAMPLE_RATE * cfg["max_seconds"])
        self._nsamples = 0

    # -- child lifecycle --
    def _spawn_locked(self):
        self._proc = subprocess.Popen(
            [CHILD_PY, "-u", str(WORKER_PATH)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(STDERR_LOG, "ab"))
        self._spawned_at = time.time()
        threading.Thread(target=self._reader, args=(self._proc,),
                         daemon=True).start()

    def _kill_locked(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def ensure_child(self, max_age=None):
        with self._spawn_lock:
            alive = self._proc is not None and self._proc.poll() is None
            stale = (alive and max_age is not None
                     and time.time() - self._spawned_at > max_age)
            if alive and not stale:
                return
            if stale:
                log(f"event=child_stale_respawn "
                    f"age_s={int(time.time() - self._spawned_at)}")
            self._kill_locked()
            self._spawn_locked()

    def respawn(self, reason):
        log(f"event=child_respawn reason=\"{reason}\"")
        with self._spawn_lock:
            self._kill_locked()
            self._spawn_locked()

    def child_alive(self):
        return self._proc is not None and self._proc.poll() is None

    def child_age(self):
        return time.time() - self._spawned_at if self._proc else -1

    def _send(self, cmd):
        try:
            self._proc.stdin.write((cmd + "\n").encode())
            self._proc.stdin.flush()
            return True
        except Exception:
            return False

    # -- packet reader (one thread per child; exits on child EOF) --
    def _reader(self, proc):
        import struct
        f = proc.stdout
        while True:
            head = f.read(4)
            if len(head) < 4:
                return
            n = struct.unpack(">I", head)[0]
            data = f.read(n)
            if len(data) < n:
                return
            kind, payload = data[:1], data[1:]
            if kind == b"A":
                if not self.recording:
                    continue
                chunk = np.frombuffer(payload, dtype=np.float32)
                self.frames.append(chunk)
                self._nsamples += len(chunk)
                self.level = float(np.sqrt(np.mean(chunk ** 2)))
                if self.on_chunk is not None:
                    self.on_chunk(chunk)
                if self._nsamples > self._max_samples:
                    # _rec_ticker notices and finishes the dictation,
                    # mirroring the old CallbackStop behavior.
                    self.recording = False
            elif kind == b"J":
                try:
                    msg = json.loads(payload)
                except Exception:
                    continue
                ev = msg.get("event")
                if ev == "started":
                    self.device_name = msg.get("device", "?")
                    self.device_rate = msg.get("rate", 0) or 0
                    self._started.set()
                elif ev == "stopped":
                    self._stopped.set()
                elif ev == "error":
                    self._error = msg.get("error", "?")
                    log(f"capture child error: {self._error}")
                    self._started.set()
                    self._stopped.set()

    # -- Recorder-compatible interface --
    def start(self):
        if self.recording:
            return False
        self.ensure_child(max_age=self.MAX_CHILD_AGE)
        for attempt in (1, 2):
            self.frames = []
            self._nsamples = 0
            self._error = None
            self._started.clear()
            self._stopped.clear()
            t0 = time.time()
            if (self._send("START") and self._started.wait(self.START_TIMEOUT)
                    and not self._error):
                self.open_ms = int((time.time() - t0) * 1000)
                self.recording = True
                return True
            if attempt == 1:
                self.respawn("start timeout/error")
        log("capture start failed after respawn")
        return False

    def stop(self):
        self.recording = False
        frames, self.frames = self.frames, []
        self._stopped.clear()
        ok = self._send("STOP") and self._stopped.wait(self.STOP_TIMEOUT)
        if not ok:
            self.wedge_count += 1
            log(f"event=child_wedge_respawn device=\"{self.device_name}\" "
                f"wedges={self.wedge_count}")
            threading.Thread(target=self.respawn, args=("stop wedge",),
                             daemon=True).start()
        if not frames:
            return None
        return np.concatenate(frames)

    def shutdown(self):
        with self._spawn_lock:
            if self._proc is not None:
                self._send("QUIT")
                self._kill_locked()


class ParakeetEngine:
    """In-process NVIDIA Parakeet v3 via MLX. ~3x faster than whisper-medium
    on M-series (1s warm for a short dictation) with comparable DE/NL/EN
    quality. No prompt biasing: proper nouns are fixed by the cleanup LLM
    using dictionary.txt instead."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.jobs = queue.Queue()
        self.ready = threading.Event()
        self.error = None
        self._acc = []       # small audio chunks, flushed per ~1s
        self._acc_len = 0

    def start(self):
        # MLX streams are bound to the thread that created them: loading
        # and ALL inference MUST happen on the same dedicated worker thread,
        # otherwise "There is no Stream(gpu, 0) in current thread".
        threading.Thread(target=self._worker, daemon=True).start()
        self.ready.wait()
        if self.error:
            raise RuntimeError(self.error)

    def _worker(self):
        boost_thread_qos()
        try:
            import mlx.core as mx
            try:
                # Wire the weights into memory so idle periods don't evict
                # them; otherwise the first dictation after a pause is slow.
                mx.set_wired_limit(1600 * 1024 * 1024)
            except Exception:
                pass
            from parakeet_mlx import from_pretrained
            # Resolve the cached snapshot offline: from_pretrained's own hub
            # call fails in GUI-launched contexts and then misreads the repo
            # id as a local directory.
            try:
                from huggingface_hub import snapshot_download
                model_ref = snapshot_download(self.cfg["parakeet_model"],
                                              local_files_only=True)
            except Exception:
                model_ref = self.cfg["parakeet_model"]  # first run: download
            self.model = from_pretrained(model_ref)
            # First inference compiles the graph (~5s); absorb it at boot
            # with a short silent clip so the first real dictation is fast.
            warm = "/tmp/localflow_warmup.wav"
            with wave.open(warm, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"\x00\x00" * SAMPLE_RATE)
            self.model.transcribe(warm)
            os.unlink(warm)
        except Exception as e:
            self.error = str(e)
            self.ready.set()
            return
        self.ready.set()
        log("parakeet ready")

        # Command loop. Streaming transcribes DURING recording, so stopping
        # a dictation only finalizes — long recordings finish near-instantly.
        stream = None

        def close_stream(s):
            try:
                s.__exit__(None, None, None)
            except Exception:
                pass

        while True:
            item = self.jobs.get()
            kind = item[0]
            if kind == "quit":
                return
            try:
                if kind == "start":
                    if stream is not None:
                        close_stream(stream)
                    stream = self.model.transcribe_stream(
                        context_size=(256, 256))
                    stream.__enter__()
                elif kind == "audio":
                    if stream is not None:
                        stream.add_audio(mx.array(item[1]))
                elif kind == "finish":
                    out = item[1]
                    if stream is None:
                        out.put(("", None))
                    else:
                        text = " ".join(stream.result.text.split())
                        close_stream(stream)
                        stream = None
                        out.put((text, None))
                elif kind == "abort":
                    if stream is not None:
                        close_stream(stream)
                        stream = None
                elif kind == "file":
                    r = self.model.transcribe(item[1], chunk_duration=30.0,
                                              overlap_duration=5.0)
                    item[2].put((" ".join(r.text.split()), None))
            except Exception as e:
                log(f"parakeet worker error ({kind}): {e}")
                if kind == "finish":
                    item[1].put((None, e))
                elif kind == "file":
                    item[2].put((None, e))
                if stream is not None:
                    close_stream(stream)
                    stream = None

    # -- streaming API (thread-safe: everything goes through the job queue) --
    def stream_start(self):
        self._acc = []
        self._acc_len = 0
        self.jobs.put(("start",))

    def feed(self, chunk):
        """Called from the audio callback; batches ~4s before handing off.
        Chunk size dominates streaming speed: 1s chunks run at 0.79x
        realtime (falls behind under load), 4s chunks at 0.19x (5x margin).
        """
        self._acc.append(chunk)
        self._acc_len += len(chunk)
        if self._acc_len >= SAMPLE_RATE * 4:
            self.jobs.put(("audio", np.concatenate(self._acc)))
            self._acc = []
            self._acc_len = 0

    def backlog(self):
        return self.jobs.qsize()

    def stream_finish(self):
        if self._acc:
            self.jobs.put(("audio", np.concatenate(self._acc)))
            self._acc = []
            self._acc_len = 0
        out = queue.Queue()
        self.jobs.put(("finish", out))
        text, err = out.get()
        if err:
            raise err
        return text

    def stream_abort(self):
        self._acc = []
        self._acc_len = 0
        # Drop queued audio first — a stale backlog is exactly why we abort;
        # without this, follow-up jobs would wait behind it anyway.
        try:
            while True:
                self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.jobs.put(("abort",))

    def transcribe(self, wav_path, language, prompt):
        """Batch fallback. Returns (text, ""): parakeet does not report lang."""
        out = queue.Queue()
        self.jobs.put(("file", wav_path, out))
        text, err = out.get()
        if err:
            raise err
        return text, ""

    def stop(self):
        self.jobs.put(("quit",))
        self.model = None


class WhisperServer:
    """Keeps whisper.cpp loaded in RAM via whisper-server for sub-second latency."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = None
        self.url = f"http://127.0.0.1:{cfg['whisper_port']}/inference"

    def start(self):
        if self.is_up():
            return
        self.proc = subprocess.Popen(
            [self.cfg["whisper_server_bin"],
             "-m", self.cfg["whisper_model"],
             "--host", "127.0.0.1",
             "--port", str(self.cfg["whisper_port"]),
             *self.cfg.get("whisper_flags", [])],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            if self.is_up():
                log("whisper-server up")
                return
            time.sleep(0.5)
        raise RuntimeError("whisper-server failed to start")

    def is_up(self):
        try:
            requests.get(f"http://127.0.0.1:{self.cfg['whisper_port']}/", timeout=1)
            return True
        except requests.RequestException:
            return False

    def transcribe(self, wav_path, language, prompt):
        """Returns (text, detected_language_code)."""
        data = {"response_format": "verbose_json", "temperature": "0.0"}
        if language and language != "auto":
            data["language"] = language
        else:
            data["language"] = "auto"
        if prompt:
            data["prompt"] = prompt
        with open(wav_path, "rb") as f:
            r = requests.post(self.url, files={"file": f}, data=data, timeout=120)
        r.raise_for_status()
        j = r.json()
        text = " ".join(j.get("text", "").split())
        lang = (j.get("language") or "").strip().lower()
        return text, lang

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None


def ollama_cleanup(cfg, text, lang=""):
    """Local LLM pass: filler removal, punctuation, formatting. Falls back to raw text."""
    lang_name = LANG_NAMES.get(lang, lang)
    if lang_name:
        lang_clause = (f"The dictated text is in {lang_name}. "
                       f"Your output MUST be in {lang_name}. ")
    else:
        lang_clause = "Keep the ORIGINAL language of the text. "
    names = load_dictionary()
    if names:
        names_clause = (f"Known proper nouns: {names}. If a word is a close "
                        f"phonetic match to one of these names, replace it "
                        f"with the exact spelling. ")
    else:
        names_clause = ""
    try:
        r = requests.post(
            f"{cfg['ollama_url']}/api/chat",
            json={
                "model": cfg["ollama_model"],
                "stream": False,
                # Short keep_alive: a resident 3b model starves parakeet
                # on 8 GB. Each LLM cleanup pays the load cost instead.
                "keep_alive": "2m",
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system",
                     "content": CLEANUP_SYSTEM_PROMPT.format(
                         lang_clause=lang_clause, names_clause=names_clause)},
                    {"role": "user", "content": text},
                ],
            },
            timeout=60,
        )
        r.raise_for_status()
        cleaned = r.json()["message"]["content"].strip().strip('"').strip()
        # Guard against a chatty small model: if output balloons, keep raw.
        if not cleaned or len(cleaned) > 2 * len(text) + 60:
            log("cleanup output rejected (empty or ballooned), using raw")
            return text
        return cleaned
    except Exception as e:
        log(f"ollama cleanup failed, using raw: {e}")
        return text


def paste_text(text):
    """Insert text at cursor: clipboard swap + Cmd+V, then restore clipboard."""
    old = subprocess.run(["pbpaste"], capture_output=True).stdout
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))
    time.sleep(0.06)
    kb = Controller()
    with kb.pressed(Key.cmd):
        kb.press("v")
        kb.release("v")
    time.sleep(0.15)
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(old)


class HUD:
    """Floating pill at the bottom of the screen, styled after VoiceInk:
    black capsule, gray seal badge left, live waveform bars center,
    gold sparkles right. Non-activating: focus stays in the target app.
    All UI mutations are marshalled onto the main thread via callAfter.
    """

    W, H = 232, 52
    NBARS = 14
    BAR_W = 4.0
    BAR_MIN = 5.0
    BAR_MAX = 24.0

    def __init__(self):
        rect = Foundation.NSMakeRect(0, 0, self.W, self.H)
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setLevel_(25)  # NSStatusWindowLevel: above normal windows
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setHidesOnDeactivate_(False)
        # canJoinAllSpaces | fullScreenAuxiliary
        self.panel.setCollectionBehavior_((1 << 0) | (1 << 8))
        content = self.panel.contentView()
        content.setWantsLayer_(True)
        content.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.96).CGColor())
        content.layer().setCornerRadius_(self.H / 2.0)

        def symbol(name, tint, x, size=22):
            iv = NSImageView.alloc().initWithFrame_(
                Foundation.NSMakeRect(x, (self.H - size) / 2.0, size, size))
            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                name, None)
            iv.setImage_(img)
            iv.setContentTintColor_(tint)
            content.addSubview_(iv)
            return iv

        self.badge = symbol("checkmark.seal.fill",
                            NSColor.systemGrayColor(), 16)
        self.sparkle = symbol("sparkles",
                              NSColor.systemYellowColor(), self.W - 38)

        # Waveform bars: tiny layer-backed views, heights follow mic level.
        area_x0, area_x1 = 52.0, self.W - 52.0
        step = (area_x1 - area_x0 - self.BAR_W) / (self.NBARS - 1)
        self.bars = []
        for i in range(self.NBARS):
            bar = NSView.alloc().initWithFrame_(Foundation.NSMakeRect(
                area_x0 + i * step, (self.H - self.BAR_MIN) / 2.0,
                self.BAR_W, self.BAR_MIN))
            bar.setWantsLayer_(True)
            bar.layer().setBackgroundColor_(
                NSColor.colorWithCalibratedWhite_alpha_(0.9, 0.95).CGColor())
            bar.layer().setCornerRadius_(self.BAR_W / 2.0)
            content.addSubview_(bar)
            self.bars.append(bar)

        # Error label, hidden unless something goes wrong.
        self.label = NSTextField.alloc().initWithFrame_(
            Foundation.NSMakeRect(46, (self.H - 16) / 2.0 - 1,
                                  self.W - 92, 16))
        self.label.setBezeled_(False)
        self.label.setDrawsBackground_(False)
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setAlignment_(2)  # NSTextAlignmentCenter (macOS)
        self.label.setFont_(NSFont.systemFontOfSize_weight_(12, 0.3))
        self.label.setTextColor_(NSColor.whiteColor())
        self.label.setHidden_(True)
        content.addSubview_(self.label)

        screen = NSScreen.mainScreen().frame()
        self.panel.setFrameOrigin_(
            Foundation.NSMakePoint((screen.size.width - self.W) / 2.0, 140))
        # Dead-man switch: every state sets a deadline, the janitor hides
        # the panel once it passes. Active tickers keep extending it, so a
        # stuck pill (thread race, killed worker, whatever) self-clears.
        self._deadline = 0.0
        threading.Thread(target=self._janitor, daemon=True).start()

    def _janitor(self):
        while True:
            time.sleep(0.5)
            if self._deadline and time.time() > self._deadline:
                self._deadline = 0.0
                AppHelper.callAfter(self._reset_and_hide)

    # -- main-thread appliers --
    def _apply_levels(self, levels):
        self.label.setHidden_(True)
        for bar, v in zip(self.bars, levels):
            h = self.BAR_MIN + (self.BAR_MAX - self.BAR_MIN) * max(0.0, min(1.0, v))
            f = bar.frame()
            bar.setFrame_(Foundation.NSMakeRect(
                f.origin.x, (self.H - h) / 2.0, self.BAR_W, h))
            bar.setHidden_(False)
        self.panel.orderFrontRegardless()

    def _apply_msg(self, text):
        for bar in self.bars:
            bar.setHidden_(True)
        self.label.setStringValue_(text)
        self.label.setHidden_(False)
        self.panel.orderFrontRegardless()

    def _apply_badge(self, color):
        self.badge.setContentTintColor_(color)

    # -- thread-safe API --
    def set_levels(self, levels):
        levels = (list(levels) + [0.0] * self.NBARS)[:self.NBARS]
        self._deadline = time.time() + 1.5
        AppHelper.callAfter(self._apply_levels, levels)

    def flash_ok(self, seconds=1.0):
        self._deadline = time.time() + seconds
        AppHelper.callAfter(self._apply_badge, NSColor.systemGreenColor())
        AppHelper.callAfter(self._apply_levels, [0.15] * self.NBARS)

    def flash_msg(self, text, seconds=1.4):
        self._deadline = time.time() + seconds
        AppHelper.callAfter(self._apply_msg, text)

    def _reset_and_hide(self):
        self.badge.setContentTintColor_(NSColor.systemGrayColor())
        self.label.setHidden_(True)
        self.panel.orderOut_(None)

    def badge_recording(self, on):
        """Red badge while the mic is live — a silent recording must never
        look like an idle or stuck pill."""
        color = NSColor.systemRedColor() if on else NSColor.systemGrayColor()
        AppHelper.callAfter(self._apply_badge, color)

    def hide(self):
        self._deadline = 0.0
        AppHelper.callAfter(self._reset_and_hide)


class DictationState:
    """Single lock-guarded state machine replacing the old scattered
    hold_active/toggle_active/_busy bools, which were mutated from many
    threads and could stick (a stuck _busy silently ate every hotkey).
    PROCESSING self-expires: even if every safety net fails, the hotkey
    comes back after EXPIRE_S."""

    IDLE, RECORDING, PROCESSING = "idle", "recording", "processing"
    EXPIRE_S = 95

    def __init__(self):
        self._lock = threading.Lock()
        self._state = self.IDLE
        self._mode = None  # "hold" | "toggle" while RECORDING
        self._expires_at = 0.0

    def _expire_locked(self):
        """Returns True if an expired PROCESSING was reset (log outside)."""
        if self._state == self.PROCESSING and time.time() > self._expires_at:
            self._state = self.IDLE
            self._mode = None
            return True
        return False

    def get(self):
        with self._lock:
            expired = self._expire_locked()
            state, mode = self._state, self._mode
        if expired:
            log("event=busy_expired (processing state auto-reset)")
        return state, mode

    def is_recording(self):
        return self.get()[0] == self.RECORDING

    def is_processing(self):
        return self.get()[0] == self.PROCESSING

    def mode(self):
        return self.get()[1]

    def try_begin_recording(self, mode):
        with self._lock:
            expired = self._expire_locked()
            ok = self._state == self.IDLE
            if ok:
                self._state = self.RECORDING
                self._mode = mode
        if expired:
            log("event=busy_expired (processing state auto-reset)")
        return ok

    def try_finish(self):
        """RECORDING -> PROCESSING, exactly one winner (double-taps,
        tickers and sleep handler race for this)."""
        with self._lock:
            if self._state != self.RECORDING:
                return False
            self._state = self.PROCESSING
            self._mode = None
            self._expires_at = time.time() + self.EXPIRE_S
            return True

    def to_idle(self):
        with self._lock:
            self._state = self.IDLE
            self._mode = None


class _WakeObserver(Foundation.NSObject):
    """NSWorkspace sleep/wake bridge. Notifications arrive on the main
    run loop; hand off to a thread immediately, never block here."""

    def initWithApp_(self, app):
        self = objc.super(_WakeObserver, self).init()
        if self is None:
            return None
        self.app = app
        return self

    def onWake_(self, note):
        threading.Thread(target=self.app._on_wake, daemon=True).start()

    def onSleep_(self, note):
        threading.Thread(target=self.app._on_sleep, daemon=True).start()


class LocalFlowApp(rumps.App):
    IDLE = "\U0001f3a4"       # microphone
    REC = "\U0001f534"        # red circle
    BUSY = "⚙️"     # gear
    WAIT = "⏳"     # model loading

    def __init__(self):
        super().__init__(self.IDLE, quit_button=None)
        self.cfg = load_config()
        self.recorder = CaptureClient(self.cfg)
        if self.cfg.get("stt_engine", "parakeet") == "parakeet":
            self.stt = ParakeetEngine(self.cfg)
            self.recorder.on_chunk = self.stt.feed
        else:
            self.stt = WhisperServer(self.cfg)
        self.last_text = ""
        self.state = DictationState()
        self._job_id = 0
        self._killed_job = -1
        self.ready = False
        self.wake_count = 0
        self._boot_ts = time.time()
        self._last_use_ts = 0.0
        self._start_fail_count = 0
        # Opt out of App Nap for the whole app lifetime.
        self._activity = Foundation.NSProcessInfo.processInfo(
        ).beginActivityWithOptions_reason_(
            Foundation.NSActivityUserInitiated
            | Foundation.NSActivityLatencyCritical,
            "LocalFlow real-time dictation")
        self.hud = HUD()

        self.status_item = rumps.MenuItem("Status: startet ...")
        self.cleanup_item = rumps.MenuItem("AI-Cleanup", callback=self.toggle_cleanup)
        self.cleanup_item.state = self.cfg["cleanup"]
        self.lang_menu = rumps.MenuItem("Sprache")
        self.lang_items = {}
        for code, label in [("auto", "Auto-Detect"), ("de", "Deutsch"),
                            ("nl", "Nederlands"), ("en", "English")]:
            item = rumps.MenuItem(label, callback=self.set_language)
            item.code = code
            item.state = self.cfg["language"] == code
            self.lang_menu.add(item)
            self.lang_items[code] = item
        self.copy_item = rumps.MenuItem("Letztes Diktat kopieren", callback=self.copy_last)
        self.dict_item = rumps.MenuItem("Wörterbuch öffnen", callback=self.open_dict)
        self.diag_item = rumps.MenuItem("Diagnose", callback=self.diagnose)
        self.menu = [
            self.status_item, None,
            self.cleanup_item, self.lang_menu, None,
            self.copy_item, self.dict_item, self.diag_item, None,
            rumps.MenuItem("Beenden", callback=self.quit_app),
        ]

        # Sleep/wake: CoreAudio device state goes stale across sleep and
        # was the prime trigger for first-memo-after-idle failures.
        self._wake_obs = _WakeObserver.alloc().initWithApp_(self)
        nc = NSWorkspace.sharedWorkspace().notificationCenter()
        nc.addObserver_selector_name_object_(
            self._wake_obs, b"onWake:", NSWorkspaceDidWakeNotification, None)
        nc.addObserver_selector_name_object_(
            self._wake_obs, b"onSleep:", NSWorkspaceWillSleepNotification,
            None)

        threading.Thread(target=self._boot, daemon=True).start()
        threading.Thread(target=self._hotkeys, daemon=True).start()

    # ---- sleep/wake ----
    def _on_wake(self):
        self.wake_count += 1
        log(f"event=wake wake_count={self.wake_count}")
        # Fresh child = fresh PortAudio/HAL view, BEFORE the first hotkey.
        if not self.recorder.recording:
            self.recorder.respawn("post-wake fresh HAL view")
        # Optional re-warm: after a long sleep the wired limit does not
        # stop macOS from compressing the model pages; one silent clip
        # pages them back in off the user's critical path.
        if (self.cfg.get("rewarm_on_wake", True) and self.ready
                and isinstance(self.stt, ParakeetEngine)):
            idle = (time.time() - self._last_use_ts
                    if self._last_use_ts else float("inf"))
            if idle > 3600:
                time.sleep(25)
                st, _ = self.state.get()
                if st == DictationState.IDLE:
                    try:
                        warm = "/tmp/localflow_rewarm.wav"
                        with wave.open(warm, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(b"\x00\x00" * SAMPLE_RATE)
                        t0 = time.time()
                        self.stt.transcribe(warm, "", "")
                        os.unlink(warm)
                        log(f"event=rewarm took_s={time.time()-t0:.1f}")
                    except Exception as e:
                        log(f"rewarm error: {e}")

    def _on_sleep(self):
        log("event=sleep")
        if self.state.try_finish():
            self.finish_recording()

    # ---- boot ----
    def _boot(self):
        try:
            self.title = self.WAIT
            self.status_item.title = "Status: Modell lädt …"
            t0 = time.time()
            self.stt.start()
            self.ready = True
            self.title = self.IDLE
            self.status_item.title = "Status: bereit (⌥ rechts = start/stop)"
            self.hud.flash_msg("✓ LocalFlow bereit", seconds=2.0)
            if self.cfg["sounds"]:
                play_sound("Glass")
            log(f"boot complete in {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"boot error: {e}")
            self.status_item.title = f"Status: FEHLER – {e}"
            self.hud.flash_msg("✕ Start-Fehler · siehe Log", seconds=5.0)

    # ---- hotkeys ----
    # Single pynput Listener for both hold-to-talk and the toggle combo.
    # Two concurrent listeners abort in pynput's darwin keycode_context.
    def _hotkeys(self):
        hold_key = getattr(Key, self.cfg["hold_key"], Key.alt_r)
        pressed = set()

        # start/finish run on throwaway threads: this callback runs inside
        # the CGEventTap; if it blocks (e.g. audio stop wedging against
        # CoreAudio), macOS disables the tap and every hotkey dies.
        # State transitions are atomic (DictationState), so double-taps
        # and races can only ever win once.
        def on_toggle():
            st, mode = self.state.get()
            if st == DictationState.PROCESSING or mode == "hold":
                return
            if st == DictationState.RECORDING:
                if self.state.try_finish():
                    threading.Thread(target=self.finish_recording,
                                     daemon=True).start()
            elif self.state.try_begin_recording("toggle"):
                threading.Thread(target=self.start_recording,
                                 daemon=True).start()

        key_mode = self.cfg.get("key_mode", "toggle")

        first_event = [True]
        key_down_at = [0.0]
        key_was_combo = [False]

        def on_press(key):
            if first_event[0]:
                first_event[0] = False
                log("key events flowing (input monitoring OK)")
            pressed.add(key)
            if key == hold_key:
                if not self.ready:
                    self.hud.flash_msg("⏳ Modell lädt noch …")
                elif key_mode == "hold":
                    if self.state.try_begin_recording("hold"):
                        threading.Thread(target=self.start_recording,
                                         daemon=True).start()
                else:
                    key_down_at[0] = time.time()
                    key_was_combo[0] = False
            else:
                # Option used as a modifier (é, special chars, shortcuts):
                # that is typing, not a dictation toggle.
                if hold_key in pressed:
                    key_was_combo[0] = True
                if key == Key.space:
                    ctrl = Key.ctrl in pressed or Key.ctrl_l in pressed or Key.ctrl_r in pressed
                    alt = Key.alt in pressed or Key.alt_l in pressed
                    if ctrl and alt:
                        if not self.ready:
                            self.hud.flash_msg("⏳ Modell lädt noch …")
                        else:
                            on_toggle()

        def on_release(key):
            pressed.discard(key)
            if key != hold_key:
                return
            if key_mode == "hold" and self.state.mode() == "hold":
                if self.state.try_finish():
                    threading.Thread(target=self.finish_recording,
                                     daemon=True).start()
            elif key_mode == "toggle" and self.ready:
                # Clean tap only: nothing else pressed while down, held
                # briefly. Kills accidental starts from key combos.
                if not key_was_combo[0] \
                        and time.time() - key_down_at[0] < 0.5:
                    on_toggle()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        log(f"hotkey listener started (mode={key_mode})")
        listener.join()

    # ---- record / transcribe / paste ----
    def _audio_reset(self):
        """Last resort only: capture-child respawns failed repeatedly,
        something above the child is broken. Relaunch the app."""
        if getattr(self, "_resetting", False):
            return
        self._resetting = True
        log("audio broken beyond child respawn: restarting LocalFlow")
        self.hud.flash_msg("Audio-Reset · Neustart …", seconds=3.0)

        def _go():
            time.sleep(1.2)  # let the flash render
            if APP_BUNDLE:
                subprocess.Popen(
                    ["sh", "-c", f'sleep 1; open "{APP_BUNDLE}"'])
            os._exit(0)

        threading.Thread(target=_go, daemon=True).start()

    def start_recording(self):
        try:
            if isinstance(self.stt, ParakeetEngine):
                self.stt.stream_start()
            # CaptureClient.start() carries its own deadlines (4s ack,
            # kill+respawn+retry once) — it can fail but never block long.
            if self.recorder.start():
                self._start_fail_count = 0
                idle_gap = (int(time.time() - self._last_use_ts)
                            if self._last_use_ts else -1)
                self._last_use_ts = time.time()
                log(f"event=record_start "
                    f"device=\"{self.recorder.device_name}\" "
                    f"rate={self.recorder.device_rate:.0f} "
                    f"open_ms={self.recorder.open_ms} "
                    f"idle_gap_s={idle_gap} "
                    f"child_age_s={int(self.recorder.child_age())} "
                    f"wake_count={self.wake_count}")
                self.title = self.REC
                self.hud.badge_recording(True)
                if self.cfg["sounds"]:
                    play_sound("Tink")
                threading.Thread(target=self._rec_ticker, daemon=True).start()
            else:
                self._start_fail_count += 1
                if isinstance(self.stt, ParakeetEngine):
                    self.stt.stream_abort()
                self.state.to_idle()
                if self._start_fail_count >= 2:
                    self._audio_reset()
                    return
                self.hud.flash_msg("Mikrofon-Fehler")
                self.title = self.IDLE
        except Exception as e:
            log(f"record start error: {e}")
            self.state.to_idle()
            self.hud.flash_msg("Mikrofon-Fehler")
            self.title = self.IDLE

    def _rec_ticker(self):
        """Scrolling live waveform while recording."""
        history = [0.0] * HUD.NBARS
        t0 = time.time()
        peak = 0.0
        while self.recorder.recording:
            lvl = self.recorder.level
            peak = max(peak, lvl)
            history = history[1:] + [min(1.0, lvl * 14)]
            self.hud.set_levels(history)
            # Accidental-start guard (toggle mode): 12s recording without
            # any speech means nobody meant to dictate — discard quietly.
            if self.state.mode() == "toggle" and peak < 0.015 \
                    and time.time() - t0 > 12 and self.state.try_finish():
                self.recorder.stop()
                if isinstance(self.stt, ParakeetEngine):
                    self.stt.stream_abort()
                self.state.to_idle()
                self.hud.badge_recording(False)
                self.hud.flash_msg("nichts gehört · gestoppt", seconds=2.5)
                if self.cfg["sounds"]:
                    play_sound("Basso")
                self.title = self.IDLE
                log(f"event=silent_discard peak={peak:.4f} "
                    f"device=\"{self.recorder.device_name}\"")
                return
            time.sleep(0.06)
        self.hud.badge_recording(False)
        # Recording can end without user action (max_seconds hit in the
        # capture reader). Finish the dictation instead of freezing the HUD.
        if self.state.try_finish():
            self.finish_recording()

    def _busy_ticker(self):
        """Gentle uniform pulse while transcribing/formatting."""
        t0 = time.time()
        while self.state.is_processing():
            v = 0.22 + 0.14 * math.sin((time.time() - t0) * 6.0)
            self.hud.set_levels([v] * HUD.NBARS)
            time.sleep(0.06)

    def finish_recording(self):
        # Caller has already won state.try_finish(): state is PROCESSING.
        audio = self.recorder.stop()
        if audio is None or len(audio) < SAMPLE_RATE * 0.3:
            if isinstance(self.stt, ParakeetEngine):
                self.stt.stream_abort()
            self.state.to_idle()
            self.hud.flash_msg("zu kurz · verworfen", seconds=2.5)
            if self.cfg["sounds"]:
                play_sound("Basso")
            self.title = self.IDLE
            log(f"event=too_short samples={0 if audio is None else len(audio)}")
            return
        if self.cfg["sounds"]:
            play_sound("Pop")
        self.title = self.BUSY
        self._job_id += 1
        job = self._job_id
        threading.Thread(target=self._busy_ticker, daemon=True).start()
        threading.Thread(target=self._process, args=(audio, job),
                         daemon=True).start()
        threading.Timer(90, self._watchdog, args=(job,)).start()

    def _watchdog(self, job):
        # The STT call cannot be interrupted, but the UI must never stay
        # stuck: reset state and tell the user instead of pulsing forever.
        if self.state.is_processing() and self._job_id == job:
            log("watchdog: processing >90s, resetting UI")
            self._killed_job = job
            self.state.to_idle()
            time.sleep(0.1)
            self.hud.flash_msg("Timeout · siehe Log", seconds=3)
            self.title = self.IDLE

    def _process(self, audio, job):
        boost_thread_qos()
        try:
            t0 = time.time()
            streamed = (isinstance(self.stt, ParakeetEngine)
                        and self.stt.backlog() <= 3)
            if streamed:
                # Audio was transcribed live during recording; this only
                # finalizes the tail — fast regardless of length.
                text, lang = self.stt.stream_finish(), ""
            else:
                # Backlogged stream (system under heavy load) or whisper:
                # drop the stream and batch-process the full recording,
                # which is faster than draining a stale queue.
                if isinstance(self.stt, ParakeetEngine):
                    log(f"event=stream_backlog_batch "
                        f"backlog={self.stt.backlog()}")
                    self.stt.stream_abort()
                wav_path = "/tmp/localflow_rec.wav"
                pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm.tobytes())
                text, lang = self.stt.transcribe(wav_path,
                                                 self.cfg["language"],
                                                 load_dictionary())
                os.unlink(wav_path)
            t1 = time.time()
            if not text or text.lower().strip(" .!?") in ("you", "thank you", ""):
                self.state.to_idle()
                time.sleep(0.1)
                self.hud.flash_msg("nichts erkannt", seconds=2.5)
                if self.cfg["sounds"]:
                    play_sound("Basso")
                self.title = self.IDLE
                log(f"event=no_text raw=\"{(text or '')[:40]}\"")
                return
            if self.cfg["language"] != "auto":
                lang = self.cfg["language"]
            elif not lang:
                lang = detect_lang(text)
            text = post_process(text, load_dictionary())
            # Very short dictations carry no fillers; skip the LLM pass.
            if self.cfg["cleanup"] and len(text.split()) >= self.cfg.get(
                    "cleanup_min_words", 6):
                text = ollama_cleanup(self.cfg, text, lang)
            t2 = time.time()
            self.last_text = text
            if self._killed_job == job:
                log("skipping paste: job was killed by watchdog "
                    "(text im Menü unter 'Letztes Diktat kopieren')")
                return
            self.state.to_idle()
            time.sleep(0.1)  # let the busy ticker exit before the flash
            paste_text(text)
            self.hud.flash_ok()
            idle_gap = (int(t0 - self._last_use_ts)
                        if self._last_use_ts else -1)
            self._last_use_ts = time.time()
            log(f"dictated {len(text)} chars, lang={lang} "
                f"(stt {t1-t0:.1f}s, cleanup {t2-t1:.1f}s) "
                f"event=dictated streamed={streamed} "
                f"device=\"{self.recorder.device_name}\" "
                f"idle_gap_s={idle_gap} "
                f"child_age_s={int(self.recorder.child_age())} "
                f"wake_count={self.wake_count}")
        except Exception as e:
            log(f"process error: {e}")
            self.state.to_idle()
            time.sleep(0.1)
            self.hud.flash_msg("Fehler · siehe Log")
        finally:
            self.title = self.IDLE
            self.state.to_idle()

    # ---- menu callbacks ----
    def toggle_cleanup(self, sender):
        sender.state = not sender.state
        self.cfg["cleanup"] = bool(sender.state)
        CONFIG_PATH.write_text(json.dumps(self.cfg, indent=2))

    def set_language(self, sender):
        self.cfg["language"] = sender.code
        for item in self.lang_items.values():
            item.state = item.code == sender.code
        CONFIG_PATH.write_text(json.dumps(self.cfg, indent=2))

    def copy_last(self, _):
        if self.last_text:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(self.last_text.encode("utf-8"))

    def open_dict(self, _):
        subprocess.Popen(["open", "-t", str(DICT_PATH)])

    def diagnose(self, _):
        st, mode = self.state.get()
        info = {
            "ready": self.ready,
            "state": st,
            "mode": mode,
            "recording": self.recorder.recording,
            "device": self.recorder.device_name,
            "child_alive": self.recorder.child_alive(),
            "child_age_s": int(self.recorder.child_age()),
            "wedge_count": self.recorder.wedge_count,
            "wake_count": self.wake_count,
            "uptime_s": int(time.time() - self._boot_ts),
            "last_use_ago_s": (int(time.time() - self._last_use_ts)
                               if self._last_use_ts else -1),
            "engine": type(self.stt).__name__,
            "backlog": (self.stt.backlog()
                        if isinstance(self.stt, ParakeetEngine) else 0),
        }
        dump = json.dumps(info)
        log(f"event=diagnose {dump}")
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        p.communicate(dump.encode("utf-8"))
        self.hud.flash_msg("Diagnose → Log + Zwischenablage", seconds=2.0)

    def quit_app(self, _):
        self.recorder.shutdown()
        self.stt.stop()
        rumps.quit_application()


if __name__ == "__main__":
    LocalFlowApp().run()
