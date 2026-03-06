#!/bin/bash
export DISPLAY=${DISPLAY:-:0}
export PYNPUT_BACKEND=uinput
exec python3 /app/bin/whisper_tray.py
