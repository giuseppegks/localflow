#!/bin/zsh
# LocalFlow installer für einen frischen Mac (Apple Silicon, macOS 14+).
# Aufruf aus dem geklonten Repo:   ./install.sh
# Idempotent: nochmal ausführen ist ungefährlich (Update-Pfad).
#
# Was passiert:
#   1. Homebrew + ffmpeg prüfen/installieren
#   2. uv (Python-Manager) installieren
#   3. .venv mit exakt Python 3.12.12 + festgepinnten Paketen (requirements.txt)
#   4. Parakeet-Modell (~2,3 GB) in den HuggingFace-Cache laden
#   5. Selbsttest
#   6. LocalFlow.app bauen (py2app alias mode) -> ~/Applications
#   7. LaunchAgents (Autostart + Watchdog) installieren
#   8. App starten
#
# WICHTIG: alias mode = die App zeigt auf DIESES Repo + .venv.
# Repo-Ordner danach NICHT verschieben oder löschen.
# Mit LOCALFLOW_BUILD_ONLY=1 stoppt das Skript nach dem Build (kein Copy,
# keine LaunchAgents, kein Start).

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "FEHLER: LocalFlow läuft nur auf macOS mit Apple Silicon (M1/M2/M3/M4)."
  exit 1
fi

cd "$(dirname "$0")"
ROOT="$(pwd)"
echo "==> Repo: $ROOT"

# 1. Homebrew + ffmpeg
if ! command -v brew >/dev/null 2>&1; then
  echo "FEHLER: Homebrew fehlt. Erst installieren: https://brew.sh  (dann ./install.sh nochmal)"
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> ffmpeg installieren"
  brew install ffmpeg
fi

# 2. uv
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv installieren"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 3. venv + Pakete (exakt gepinnt)
PY_VERSION="$(cat .python-version)"
if [[ ! -x .venv/bin/python ]]; then
  echo "==> Python $PY_VERSION + .venv anlegen"
  uv venv --python "$PY_VERSION" .venv
fi
echo "==> Pakete installieren (requirements.txt)"
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c 'import sys; print("Python", sys.version.split()[0])'

# 4. Modell
echo "==> Parakeet-Modell laden (einmalig ~2,3 GB, später nur Cache-Check)"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("mlx-community/parakeet-tdt-0.6b-v3")
print("Modell:", p)
PY

# 5. Selbsttest
echo "==> Selbsttest"
.venv/bin/python tests/test_state.py
.venv/bin/python -c 'import rumps, pynput, sounddevice, mlx.core, parakeet_mlx, AppKit; print("Imports OK")'

# 6. App-Bundle
echo "==> LocalFlow.app bauen"
rm -rf build dist
.venv/bin/python -W ignore setup.py py2app -A >/dev/null 2>&1
test -d dist/LocalFlow.app || { echo "FEHLER: Build fehlgeschlagen"; exit 1; }

if [[ "${LOCALFLOW_BUILD_ONLY:-0}" == "1" ]]; then
  echo "==> BUILD_ONLY gesetzt, fertig nach Build: $ROOT/dist/LocalFlow.app"
  exit 0
fi

mkdir -p "$HOME/Applications"
if pgrep -f "LocalFlow.app/Contents/MacOS/" >/dev/null; then
  echo "==> laufende LocalFlow.app beenden"
  pkill -f "LocalFlow.app/Contents/MacOS/" || true
  sleep 1
fi
rm -rf "$HOME/Applications/LocalFlow.app"
cp -R dist/LocalFlow.app "$HOME/Applications/"
codesign --force -s - "$HOME/Applications/LocalFlow.app"
echo "==> installiert: $HOME/Applications/LocalFlow.app"

# 7. LaunchAgents
mkdir -p "$HOME/Library/LaunchAgents"
for plist in com.giu.localflow.plist com.giu.localflow.watchdog.plist; do
  dest="$HOME/Library/LaunchAgents/$plist"
  launchctl unload "$dest" 2>/dev/null || true
  cp "$plist" "$dest"
  launchctl load -w "$dest"
done
echo "==> Autostart + Watchdog aktiv"

# 8. Start
open -a "$HOME/Applications/LocalFlow.app"

cat <<'TXT'

==> FERTIG. Jetzt noch von Hand (einmalig, macOS verlangt das):
    Systemeinstellungen -> Datenschutz & Sicherheit:
      - Mikrofon:          LocalFlow erlauben
      - Bedienungshilfen:  LocalFlow erlauben   (zum Einfügen des Textes)
      - Eingabemonitoring: LocalFlow erlauben   (für die Hotkey-Taste)
    Danach LocalFlow einmal beenden (Menüleiste, Mikro-Symbol) und neu starten,
    oder einfach:  open -a ~/Applications/LocalFlow.app

    Test: Textfeld anklicken, rechte Option-Taste (⌥) tippen, sprechen,
    nochmal ⌥ tippen. Der Text erscheint an der Cursor-Position.
    Log bei Problemen:  tail -f localflow.log   (im Repo-Ordner)
TXT
