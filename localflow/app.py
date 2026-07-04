#!/usr/bin/env python3
"""LocalFlow: private, fully local voice dictation for macOS.

Hold Right-Option (or toggle with Ctrl+Alt+Space), speak, release.
Audio -> whisper.cpp (local) -> Ollama cleanup (local) -> pasted at cursor.
Nothing leaves the machine.
"""

import json
import os
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

import Foundation
from AppKit import (NSBackingStoreBuffered, NSColor, NSFont, NSPanel,
                    NSScreen, NSTextField, NSWindowStyleMaskBorderless,
                    NSWindowStyleMaskNonactivatingPanel)
from PyObjCTools import AppHelper

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
DICT_PATH = ROOT / "dictionary.txt"
LOG_PATH = ROOT / "localflow.log"

DEFAULT_CONFIG = {
    "hold_key": "alt_r",
    "toggle_hotkey": "<ctrl>+<alt>+<space>",
    "language": "auto",
    "cleanup": True,
    "ollama_model": "qwen2.5:3b",
    "ollama_url": "http://127.0.0.1:11434",
    "whisper_port": 8178,
    "whisper_model": str(ROOT / "models" / "ggml-medium-q5_0.bin"),
    "whisper_server_bin": "/opt/homebrew/bin/whisper-server",
    "whisper_flags": ["-bs", "1", "-fa"],
    "sample_rate": 16000,
    "max_seconds": 180,
    "sounds": True,
}

SAMPLE_RATE = 16000

CLEANUP_SYSTEM_PROMPT = (
    "You clean up dictated text. Fix punctuation and capitalization. "
    "Remove filler words (um, uh, äh, ähm, ehm, im as filler, nou ja, "
    "dus as filler, like as filler). "
    "Fix obvious speech-recognition errors. Keep the ORIGINAL language "
    "(German, Dutch, or English). Do NOT translate. Do NOT add content. "
    "Do NOT answer questions contained in the text. "
    "Keep everything else word-for-word. "
    "Output ONLY the cleaned text, no quotes, no commentary."
)


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
        data = {"response_format": "json", "temperature": "0.0"}
        if language and language != "auto":
            data["language"] = language
        else:
            data["language"] = "auto"
        if prompt:
            data["prompt"] = prompt
        with open(wav_path, "rb") as f:
            r = requests.post(self.url, files={"file": f}, data=data, timeout=120)
        r.raise_for_status()
        return " ".join(r.json().get("text", "").split())

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None


def ollama_cleanup(cfg, text):
    """Local LLM pass: filler removal, punctuation, formatting. Falls back to raw text."""
    try:
        r = requests.post(
            f"{cfg['ollama_url']}/api/chat",
            json={
                "model": cfg["ollama_model"],
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
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
    time.sleep(0.15)
    kb = Controller()
    with kb.pressed(Key.cmd):
        kb.press("v")
        kb.release("v")
    time.sleep(0.4)
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(old)


class HUD:
    """Floating status pill at the bottom of the screen (VoiceInk-style).

    Shows recording/processing/done states so the user always sees what
    LocalFlow is doing. Non-activating: focus stays in the target app.
    All UI mutations are marshalled onto the main thread via callAfter.
    """

    W, H = 260, 44

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
            NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.92).CGColor())
        content.layer().setCornerRadius_(self.H / 2.0)
        self.label = NSTextField.alloc().initWithFrame_(
            Foundation.NSMakeRect(10, (self.H - 20) / 2.0 - 1, self.W - 20, 20))
        self.label.setBezeled_(False)
        self.label.setDrawsBackground_(False)
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setAlignment_(2)  # NSTextAlignmentCenter (macOS)
        self.label.setFont_(NSFont.systemFontOfSize_weight_(14, 0.3))
        self.label.setTextColor_(NSColor.whiteColor())
        content.addSubview_(self.label)
        screen = NSScreen.mainScreen().frame()
        self.panel.setFrameOrigin_(
            Foundation.NSMakePoint((screen.size.width - self.W) / 2.0, 140))
        self._gen = 0

    def _apply(self, text):
        self.label.setStringValue_(text)
        self.panel.orderFrontRegardless()

    def show(self, text):
        self._gen += 1
        AppHelper.callAfter(self._apply, text)

    def flash(self, text, seconds=1.4):
        self._gen += 1
        gen = self._gen
        AppHelper.callAfter(self._apply, text)
        threading.Timer(seconds, lambda: self._hide_if(gen)).start()

    def _hide_if(self, gen):
        if gen == self._gen:
            AppHelper.callAfter(self.panel.orderOut_, None)

    def hide(self):
        self._gen += 1
        AppHelper.callAfter(self.panel.orderOut_, None)


class LocalFlowApp(rumps.App):
    IDLE = "\U0001f3a4"       # microphone
    REC = "\U0001f534"        # red circle
    BUSY = "⚙️"     # gear

    def __init__(self):
        super().__init__(self.IDLE, quit_button=None)
        self.cfg = load_config()
        self.recorder = Recorder(self.cfg)
        self.whisper = WhisperServer(self.cfg)
        self.last_text = ""
        self.hold_active = False
        self.toggle_active = False
        self._busy = False
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
            self.whisper.start()
            self.status_item.title = "Status: bereit (⌥ rechts halten / ⌃⌥Space)"
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

        def on_press(key):
            pressed.add(key)
            if key == hold_key and not self.toggle_active and not self._busy:
                if not self.hold_active:
                    self.hold_active = True
                    self.start_recording()
            elif key == Key.space:
                ctrl = Key.ctrl in pressed or Key.ctrl_l in pressed or Key.ctrl_r in pressed
                alt = Key.alt in pressed or Key.alt_l in pressed
                if ctrl and alt:
                    on_toggle()

        def on_release(key):
            pressed.discard(key)
            if key == hold_key and self.hold_active:
                self.hold_active = False
                self.finish_recording()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
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
            self.hud.flash("✕ Mikrofon-Fehler")
            self.title = self.IDLE

    def _rec_ticker(self):
        t0 = time.time()
        while self.recorder.recording:
            self.hud.show(f"🔴 Aufnahme · {int(time.time() - t0)}s")
            time.sleep(0.5)

    def finish_recording(self):
        audio = self.recorder.stop()
        if audio is None or len(audio) < SAMPLE_RATE * 0.3:
            self.hud.flash("✕ Zu kurz · verworfen")
            self.title = self.IDLE
            return
        if self.cfg["sounds"]:
            play_sound("Pop")
        self.hud.show("⚙️ Transkribiere …")
        self.title = self.BUSY
        self._busy = True
        threading.Thread(target=self._process, args=(audio,), daemon=True).start()

    def _process(self, audio):
        try:
            wav_path = "/tmp/localflow_rec.wav"
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())

            t0 = time.time()
            text = self.whisper.transcribe(wav_path, self.cfg["language"],
                                           load_dictionary())
            t1 = time.time()
            if not text or text.lower().strip(" .!?") in ("you", "thank you", ""):
                self.hud.flash("✕ Nichts erkannt")
                self.title = self.IDLE
                self._busy = False
                return
            if self.cfg["cleanup"]:
                self.hud.show("⚙️ Formatiere …")
                text = ollama_cleanup(self.cfg, text)
            t2 = time.time()
            self.last_text = text
            paste_text(text)
            self.hud.flash("✓ Eingefügt")
            log(f"dictated {len(text)} chars (stt {t1-t0:.1f}s, cleanup {t2-t1:.1f}s)")
            os.unlink(wav_path)
        except Exception as e:
            log(f"process error: {e}")
            self.hud.flash("✕ Fehler · siehe Log")
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
        self.whisper.stop()
        rumps.quit_application()


if __name__ == "__main__":
    LocalFlowApp().run()
