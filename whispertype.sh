#!/bin/bash
export PYNPUT_BACKEND=uinput
exec python3 /app/bin/whisper_tray.py
