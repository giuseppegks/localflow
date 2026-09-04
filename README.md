# LocalFlow

Private Wispr-Flow-Alternative. 100% lokal auf dem Mac: Audio, Transkription und Nachbearbeitung verlassen die Maschine nie.

## Installation (neuer Mac, 10 Minuten + Download)

Voraussetzungen: Mac mit Apple Silicon (M1 oder neuer), macOS 14+, [Homebrew](https://brew.sh) installiert. Alles andere holt das Skript selbst (uv, Python 3.12.12, alle Pakete exakt gepinnt, ffmpeg, Modell).

```bash
git clone https://github.com/giuseppegks/localflow.git ~/repos/localflow
cd ~/repos/localflow
./install.sh
```

Danach in **Systemeinstellungen → Datenschutz & Sicherheit** für LocalFlow freigeben: **Mikrofon**, **Bedienungshilfen**, **Eingabemonitoring**. Dann die App einmal neu starten (`open -a ~/Applications/LocalFlow.app`).

Test: Textfeld anklicken, rechte ⌥ tippen, sprechen, nochmal ⌥ tippen. Fertig.

Hinweise:
- Der Repo-Ordner darf danach nicht verschoben werden (die App zeigt auf `~/repos/localflow` + `.venv`).
- Eigennamen, die falsch erkannt werden: in `dictionary.txt` eintragen (ein Name pro Zeile), App neu starten.
- Update: `git pull && ./install.sh` (das Skript ist wiederholbar).
- Ollama ist optional (nur für den AI-Cleanup-Schalter im Menü); ohne Ollama läuft alles.

## Pipeline

```
Rechte ⌥ (Toggle) → Mikrofon → Parakeet v3 (MLX, in-process) → Python-Postprocess → Cmd+V an Cursor-Position
```

- **STT**: NVIDIA `parakeet-tdt-0.6b-v3` via MLX, resident im RAM, dedizierter Worker-Thread (MLX-Streams sind thread-gebunden). Chunking bei 60s gegen Speicher-Explosion. DE/NL/EN automatisch.
- **Postprocess** (`post_process()` in app.py): Füllwörter raus, Eigennamen-Fuzzy-Korrektur gegen `dictionary.txt`, Großschreibung. Kostenlos und instant.
- **LLM-Cleanup (opt-in)**: Menü → "AI-Cleanup". qwen2.5:3b via Ollama, `keep_alive 2m`. NICHT dauerhaft anlassen: resident 3b-LLM + Parakeet = Thrashing auf 8 GB (gemessen 1s → 106s).
- **Einfügen**: Clipboard-Swap + simuliertes Cmd+V, altes Clipboard wird wiederhergestellt (nur Text).

## Bedienung

| Aktion | Auslöser |
|---|---|
| Aufnahme start/stop | **Rechte ⌥ tippen** (Toggle; `key_mode: "hold"` für Push-to-talk) |
| Alternativ | **Ctrl+Alt+Space** |
| HUD-Pill (unten mittig) | Live-Waveform beim Aufnehmen, Puls beim Verarbeiten, grünes Badge bei Erfolg, Fehlertext bei Problemen |

Menü (🎤 in der Menüleiste): AI-Cleanup an/aus, Sprache erzwingen (auto/de/nl/en), letztes Diktat kopieren, Wörterbuch öffnen.

## Start & Autostart

Die App läuft als **`~/Applications/LocalFlow.app`** (py2app alias mode: referenziert dieses Repo + `.venv`, Repo nicht verschieben). Autostart: LaunchAgent `com.giu.localflow` (`open -a LocalFlow.app` bei Login), plist liegt im Repo.

Entwicklung/Debug aus dem Terminal: `./run.sh` (Prozess läuft dann unter Terminal-TCC-Identität).

Bundle neu bauen nach setup.py-Änderung (app.py-Änderungen brauchen KEINEN Rebuild, nur App-Neustart):

```bash
.venv/bin/python setup.py py2app -A
rm -rf ~/Applications/LocalFlow.app && cp -R dist/LocalFlow.app ~/Applications/
codesign --force -s - ~/Applications/LocalFlow.app
```

## macOS-Freigaben (TCC)

**LocalFlow** braucht: Bedienungshilfen (Einfügen) + Eingabemonitoring (Hotkey) + Mikrofon. Nach einem Bundle-Rebuild kleben alte Grants an der alten Signatur. Dann:

```bash
tccutil reset Accessibility com.giu.localflow
tccutil reset ListenEvent com.giu.localflow
```

und in Systemeinstellungen → Datenschutz beide neu vergeben. Symptom für fehlende Freigabe: kein `key events flowing` in `localflow.log`.

## Konfiguration

`config.json` im Projektordner:

- `key_mode`: `"toggle"` (Default) oder `"hold"`.
- `cleanup`: `false` (Default), `true` schaltet LLM-Cleanup zu (~5-10s extra).
- `stt_engine`: `"parakeet"` (Default) oder `"whisper"` (whisper.cpp-Fallback, Modelle in `models/`).
- `language`: `"auto"` / `"de"` / `"nl"` / `"en"`.

`dictionary.txt`: ein Eigenname pro Zeile → Fuzzy-Korrektur im Postprocess ("Miratera" → "Mirathera").

## Performance-Lektionen (M2, 8 GB)

| Bremse | Wirkung | Fix (in app.py) |
|---|---|---|
| App Nap + E-Cores | 19,5s statt 3s | `NSProcessInfo.beginActivityWithOptions` + `pthread_set_qos_class_self_np(USER_INTERACTIVE)` |
| LLM resident neben Parakeet | 1s → 106s | Cleanup opt-in, `keep_alive 2m` |
| Lange Aufnahme ohne Chunking | hängt für immer | `chunk_duration=60` + 90s-Watchdog |
| Weights-Eviction im Idle | erstes Diktat nach Pause langsam | `mx.set_wired_limit` |

GUI-App-Fallen (PATH, ASCII-Locale, HF-Offline-Resolution, MLX-Thread-Affinität): siehe Kommentare in `app.py` und Vault-Note `Resources/tools/localflow.md`.

Diktat-Latenz real: ~2-4 Sek. je nach Länge und Systemlast. Timing pro Diktat in `localflow.log`.
