"""py2app build: python setup.py py2app -A  (alias mode, references repo+venv).

Alias mode keeps the heavy deps (MLX, numpy) in the venv instead of bundling
them; the app breaks if the repo moves. Good trade-off for a personal tool.
"""
from setuptools import setup

setup(
    app=["localflow/app.py"],
    name="LocalFlow",
    options={
        "py2app": {
            "plist": {
                "CFBundleIdentifier": "com.giu.localflow",
                "CFBundleName": "LocalFlow",
                "CFBundleShortVersionString": "1.0",
                "LSUIElement": True,
                "NSMicrophoneUsageDescription":
                    "LocalFlow nimmt Diktate über das Mikrofon auf.",
                # GUI apps launch without a UTF-8 locale or Homebrew PATH;
                # parakeet's config read and ffmpeg both need these.
                "LSEnvironment": {
                    "PYTHONUTF8": "1",
                    "LANG": "en_US.UTF-8",
                    "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
                },
            },
        },
    },
    setup_requires=["py2app"],
)
