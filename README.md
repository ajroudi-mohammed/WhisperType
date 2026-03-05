# 🎙 WhisperType

Push-to-talk voice transcription that types into any text field on your screen — powered by OpenAI Whisper, running fully locally. No cloud, no subscription, no data leaving your machine.

---

## What it does

Hold a hotkey, speak, release — your words appear instantly in whatever text field is focused. Works in any app: browsers, chat apps, terminals, editors, anywhere.

---

## Requirements

- **OS**: Linux (Ubuntu / Kubuntu / Debian-based)
- **Session**: Wayland or X11
- **Python**: 3.10+
- **GPU**: Nvidia GPU recommended (CUDA) — falls back to CPU automatically if not available

---

## Installation

Clone the repo and run the installer:

```bash
git clone https://github.com/ajroudi-mohammed/WhisperType.git
cd WhisperType
chmod +x install.sh
./install.sh
```

The installer will automatically:

- Install all system and Python dependencies
- Detect your GPU (Nvidia CUDA or CPU fallback)
- Select the best Whisper model for your hardware
- Detect your microphone
- Create a desktop launcher and app menu entry

> **First launch** will download the AI model (~1.6GB for CUDA, ~500MB for CPU). This only happens once.

---

## Usage

1. Launch **WhisperType** from your app menu or Desktop icon
2. Wait for the tray icon to turn **green** — model is loaded and ready
3. Click into any text field (browser, Slack, Discord, terminal...)
4. **Hold your hotkey** and speak
5. **Release** — your speech is transcribed and typed instantly

---

## Tray Icon States

| Icon | Color | Meaning |
|------|-------|---------|
| **…** | Grey | Loading model |
| **W** | Green | Ready — waiting for hotkey |
| **●** | Red | Recording |
| **⚙** | Orange | Transcribing |

---

## Changing the Hotkey

1. Right-click the tray icon
2. Click **⌨️ Hotkey: F9 — Change**
3. Click the button in the dialog and press your desired key
4. Click **Save**

Your hotkey is saved to `~/.config/whispertype.json` and persists across restarts.

> **Note:** The `Fn` key cannot be used as it is intercepted by keyboard firmware before reaching the OS.

---

## Quitting

Right-click the tray icon → **⏹ Quit / Unload model**

This fully unloads the AI model from memory and exits.

---

## Hardware Performance

| Hardware | Model Used | Transcription Speed |
|----------|------------|-------------------|
| Nvidia GPU (CUDA) | large-v3-turbo | ~0.5s |
| CPU only | small.en | ~3-5s |

---

## Troubleshooting

**Tray icon doesn't appear**
Make sure your desktop environment supports system tray icons. On KDE, check that the System Tray widget is added to your panel.

**No text appears after speaking**
- Check that a text field is focused before releasing the hotkey
- On Wayland, make sure `ydotool` is installed: `sudo apt install ydotool`
- Make sure passwordless sudo is configured for ydotool (the installer handles this automatically)

**Microphone not working**
Run this to test your mic level:
```bash
cd ~/whisper-type
source venv/bin/activate
python3 test_audio.py
```

**Wrong text / hallucinations**
This usually means the mic volume is too low or the recording is too short. Speak clearly and hold the hotkey for at least 1 second before speaking.

---

## Built With

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — fast Whisper inference
- [PyQt6](https://pypi.org/project/PyQt6/) — system tray UI
- [sounddevice](https://python-sounddevice.readthedocs.io/) — audio capture
- [ydotool](https://github.com/ReimuNotMoe/ydotool) — Wayland keyboard input
- [pynput](https://pynput.readthedocs.io/) — global hotkey listener

---

## License

MIT
