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
from AppKit import (NSBackingStoreBuffered, NSColor, NSFont, NSImage,
                    NSImageView, NSPanel, NSScreen, NSTextField, NSView,
                    NSWindowStyleMaskBorderless,
                    NSWindowStyleMaskNonactivatingPanel)
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


class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frames = []
        self.stream = None
        self.recording = False
        self.level = 0.0
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.recording:
                return False
            self.frames = []
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=self._callback)
            self.stream.start()
            self.recording = True
            return True

    def _callback(self, indata, frame_count, time_info, status):
        self.frames.append(indata.copy())
        self.level = float(np.sqrt(np.mean(indata ** 2)))
        if len(self.frames) * frame_count / SAMPLE_RATE > self.cfg["max_seconds"]:
            self.recording = False
            raise sd.CallbackStop()

    def stop(self):
        with self._lock:
            if self.stream is None:
                return None
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.recording = False
            if not self.frames:
                return None
            audio = np.concatenate(self.frames, axis=0).flatten()
            self.frames = []
            return audio


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

    def start(self):
        # MLX streams are bound to the thread that created them: loading
        # and transcribing MUST happen on the same dedicated worker thread,
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
        while True:
            wav_path, out = self.jobs.get()
            if wav_path is None:
                return
            try:
                # chunk_duration bounds the attention window: without it,
                # long recordings blow up memory on 8 GB and never return.
                r = self.model.transcribe(wav_path, chunk_duration=60.0,
                                          overlap_duration=10.0)
                out.put((" ".join(r.text.split()), None))
            except Exception as e:
                out.put((None, e))

    def transcribe(self, wav_path, language, prompt):
        """Returns (text, ""): parakeet auto-detects but does not report."""
        out = queue.Queue()
        self.jobs.put((wav_path, out))
        text, err = out.get()
        if err:
            raise err
        return text, ""

    def stop(self):
        self.jobs.put((None, None))
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
        self._gen = 0

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
        self._gen += 1
        AppHelper.callAfter(self._apply_levels, levels)

    def flash_ok(self, seconds=1.0):
        self._gen += 1
        gen = self._gen
        AppHelper.callAfter(self._apply_badge, NSColor.systemGreenColor())
        AppHelper.callAfter(self._apply_levels, [0.15] * self.NBARS)
        threading.Timer(seconds, lambda: self._hide_if(gen)).start()

    def flash_msg(self, text, seconds=1.4):
        self._gen += 1
        gen = self._gen
        AppHelper.callAfter(self._apply_msg, text)
        threading.Timer(seconds, lambda: self._hide_if(gen)).start()

    def _hide_if(self, gen):
        if gen == self._gen:
            AppHelper.callAfter(self._reset_and_hide)

    def _reset_and_hide(self):
        self.badge.setContentTintColor_(NSColor.systemGrayColor())
        self.label.setHidden_(True)
        self.panel.orderOut_(None)

    def hide(self):
        self._gen += 1
        AppHelper.callAfter(self._reset_and_hide)


class LocalFlowApp(rumps.App):
    IDLE = "\U0001f3a4"       # microphone
    REC = "\U0001f534"        # red circle
    BUSY = "⚙️"     # gear

    def __init__(self):
        super().__init__(self.IDLE, quit_button=None)
        self.cfg = load_config()
        self.recorder = Recorder(self.cfg)
        if self.cfg.get("stt_engine", "parakeet") == "parakeet":
            self.stt = ParakeetEngine(self.cfg)
        else:
            self.stt = WhisperServer(self.cfg)
        self.last_text = ""
        self.hold_active = False
        self.toggle_active = False
        self._busy = False
        self._job_id = 0
        self._killed_job = -1
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
        self.menu = [
            self.status_item, None,
            self.cleanup_item, self.lang_menu, None,
            self.copy_item, self.dict_item, None,
            rumps.MenuItem("Beenden", callback=self.quit_app),
        ]

        threading.Thread(target=self._boot, daemon=True).start()
        threading.Thread(target=self._hotkeys, daemon=True).start()

    # ---- boot ----
    def _boot(self):
        try:
            self.status_item.title = "Status: Modell lädt …"
            self.stt.start()
            self.status_item.title = "Status: bereit (⌥ rechts = start/stop)"
        except Exception as e:
            log(f"boot error: {e}")
            self.status_item.title = f"Status: FEHLER – {e}"

    # ---- hotkeys ----
    # Single pynput Listener for both hold-to-talk and the toggle combo.
    # Two concurrent listeners abort in pynput's darwin keycode_context.
    def _hotkeys(self):
        hold_key = getattr(Key, self.cfg["hold_key"], Key.alt_r)
        pressed = set()

        def on_toggle():
            if self.hold_active or self._busy:
                return
            if self.toggle_active:
                self.toggle_active = False
                self.finish_recording()
            else:
                self.toggle_active = True
                self.start_recording()

        key_mode = self.cfg.get("key_mode", "toggle")

        first_event = [True]

        def on_press(key):
            if first_event[0]:
                first_event[0] = False
                log("key events flowing (input monitoring OK)")
            pressed.add(key)
            if key == hold_key:
                if key_mode == "toggle":
                    on_toggle()
                elif not self.toggle_active and not self._busy \
                        and not self.hold_active:
                    self.hold_active = True
                    self.start_recording()
            elif key == Key.space:
                ctrl = Key.ctrl in pressed or Key.ctrl_l in pressed or Key.ctrl_r in pressed
                alt = Key.alt in pressed or Key.alt_l in pressed
                if ctrl and alt:
                    on_toggle()

        def on_release(key):
            pressed.discard(key)
            if key_mode == "hold" and key == hold_key and self.hold_active:
                self.hold_active = False
                self.finish_recording()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        log(f"hotkey listener started (mode={key_mode})")
        listener.join()

    # ---- record / transcribe / paste ----
    def start_recording(self):
        try:
            if self.recorder.start():
                self.title = self.REC
                if self.cfg["sounds"]:
                    play_sound("Tink")
                threading.Thread(target=self._rec_ticker, daemon=True).start()
        except Exception as e:
            log(f"record start error: {e}")
            self.hud.flash_msg("Mikrofon-Fehler")
            self.title = self.IDLE

    def _rec_ticker(self):
        """Scrolling live waveform while recording."""
        history = [0.0] * HUD.NBARS
        while self.recorder.recording:
            history = history[1:] + [min(1.0, self.recorder.level * 14)]
            self.hud.set_levels(history)
            time.sleep(0.06)
        # Recording can end without user action (max_seconds hit in the
        # audio callback). Finish the dictation instead of freezing the HUD.
        if self.toggle_active and not self._busy:
            self.toggle_active = False
            self.finish_recording()

    def _busy_ticker(self):
        """Gentle uniform pulse while transcribing/formatting."""
        t0 = time.time()
        while self._busy:
            v = 0.22 + 0.14 * math.sin((time.time() - t0) * 6.0)
            self.hud.set_levels([v] * HUD.NBARS)
            time.sleep(0.06)

    def finish_recording(self):
        audio = self.recorder.stop()
        if audio is None or len(audio) < SAMPLE_RATE * 0.3:
            self.hud.flash_msg("zu kurz · verworfen")
            self.title = self.IDLE
            return
        if self.cfg["sounds"]:
            play_sound("Pop")
        self.title = self.BUSY
        self._busy = True
        self._job_id += 1
        job = self._job_id
        threading.Thread(target=self._busy_ticker, daemon=True).start()
        threading.Thread(target=self._process, args=(audio, job),
                         daemon=True).start()
        threading.Timer(90, self._watchdog, args=(job,)).start()

    def _watchdog(self, job):
        # The STT call cannot be interrupted, but the UI must never stay
        # stuck: reset state and tell the user instead of pulsing forever.
        if self._busy and self._job_id == job:
            log("watchdog: processing >90s, resetting UI")
            self._killed_job = job
            self._busy = False
            time.sleep(0.1)
            self.hud.flash_msg("Timeout · siehe Log", seconds=3)
            self.title = self.IDLE

    def _process(self, audio, job):
        boost_thread_qos()
        try:
            wav_path = "/tmp/localflow_rec.wav"
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())

            t0 = time.time()
            text, lang = self.stt.transcribe(wav_path, self.cfg["language"],
                                             load_dictionary())
            t1 = time.time()
            if not text or text.lower().strip(" .!?") in ("you", "thank you", ""):
                self._busy = False
                time.sleep(0.1)
                self.hud.flash_msg("nichts erkannt")
                self.title = self.IDLE
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
            self._busy = False
            time.sleep(0.1)  # let the busy ticker exit before the flash
            paste_text(text)
            self.hud.flash_ok()
            log(f"dictated {len(text)} chars, lang={lang} "
                f"(stt {t1-t0:.1f}s, cleanup {t2-t1:.1f}s)")
            os.unlink(wav_path)
        except Exception as e:
            log(f"process error: {e}")
            self._busy = False
            time.sleep(0.1)
            self.hud.flash_msg("Fehler · siehe Log")
        finally:
            self.title = self.IDLE
            self._busy = False

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

    def quit_app(self, _):
        self.stt.stop()
        rumps.quit_application()


if __name__ == "__main__":
    LocalFlowApp().run()
