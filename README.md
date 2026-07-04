# LocalFlow

Private Wispr-Flow-Alternative. 100% lokal auf dem Mac: Audio, Transkription und AI-Cleanup verlassen die Maschine nie.

## Pipeline

```
Hotkey → Mikrofon → whisper.cpp (whisper-server, Metal) → Ollama-Cleanup → Cmd+V an Cursor-Position
```

- **STT**: `ggml-medium-q5_0` via `whisper-server` (bleibt im RAM, ~3 Sek. pro Diktat). Auto-Detect DE/NL/EN.
- **Cleanup**: `qwen2.5:3b` via Ollama (~1,5 Sek.): Füllwörter raus, Interpunktion, Großschreibung. Fällt bei Fehler auf Raw-Transkript zurück.
- **Einfügen**: Clipboard-Swap + simuliertes Cmd+V, altes Clipboard wird wiederhergestellt (nur Text).

## Bedienung

| Aktion | Auslöser |
|---|---|
| Hold-to-talk | **Rechte Option-Taste (⌥)** halten, sprechen, loslassen |
| Toggle | **Ctrl+Alt+Space** start, nochmal = stop + einfügen |
| Menüleiste | 🎤 idle, 🔴 aufnehmend, ⚙️ verarbeitet |

Menü: AI-Cleanup an/aus, Sprache erzwingen (auto/de/nl/en), letztes Diktat kopieren, Wörterbuch öffnen.

## Start

```bash
~/repos/localflow/run.sh
```

Beim ersten Start fragt macOS nach **Mikrofon** und **Bedienungshilfen** (Accessibility) für das Terminal bzw. Python. Beides erteilen, sonst funktionieren Hotkey und Einfügen nicht.

### Autostart (optional)

```bash
cp ~/repos/localflow/com.giu.localflow.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.giu.localflow.plist
```

## Konfiguration

`config.json` im Projektordner:

- `whisper_model`: Pfad zum GGML-Modell. Alternativen in `models/`: `ggml-large-v3-turbo-q5_0.bin` (beste Qualität, ~5 Sek. statt 3).
- `ollama_model`: `qwen2.5:3b` (Default) oder `qwen2.5:1.5b` (schneller, schlechter).
- `cleanup`: `false` = Raw-Transkript direkt einfügen.
- `hold_key` / `toggle_hotkey`: pynput-Namen.

`dictionary.txt`: ein Eigenname pro Zeile, wird whisper als Kontext-Prompt mitgegeben (so wird "Mirathera" statt "Mira Theram" erkannt).

## Benchmarks (M2, 8 GB, 8-Sek.-Diktat, warm)

| Whisper-Modell | Latenz | Qualität |
|---|---|---|
| large-v3-turbo q8/q5 | ~5 Sek. | beste |
| **medium q5_0 (Default)** | ~3 Sek. | gut |
| small q8_0 | ~0,8 Sek. | Fehler bei Namen/Wörtern |

Cleanup: qwen2.5:3b ~1,5 Sek. warm. Wichtig auf 8 GB: turbo-Whisper + 3b-LLM gleichzeitig resident = Speicherdruck (gemessen 19 Sek.). Deshalb medium + 3b als Default-Paar.
