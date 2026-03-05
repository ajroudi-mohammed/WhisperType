#!/bin/bash
set -e

echo ""
echo "🎙  WhisperType Installer"
echo "=========================="
echo ""

# ── Check Linux ───────────────────────────────────────────────────────
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ This installer only supports Linux."
    exit 1
fi

# ── Check session type ────────────────────────────────────────────────
SESSION=$XDG_SESSION_TYPE
echo "🖥  Session type: $SESSION"

# ── System dependencies ───────────────────────────────────────────────
echo ""
echo "📦 Installing system dependencies..."
sudo apt update -q
sudo apt install -y python3-pip python3-venv portaudio19-dev \
     xdotool xclip wl-clipboard ydotool

# ── Project folder ────────────────────────────────────────────────────
INSTALL_DIR="$HOME/whisper-type"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# ── Copy files ────────────────────────────────────────────────────────
echo ""
echo "📁 Copying files..."
cp "$(dirname "$0")/whisper_tray.py" "$INSTALL_DIR/" 2>/dev/null || true

# ── Virtual environment ───────────────────────────────────────────────
echo ""
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ── Detect GPU ────────────────────────────────────────────────────────
echo ""
echo "🔍 Detecting GPU..."

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ Nvidia GPU detected — installing CUDA version"
    pip install --quiet torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121
    DEVICE="cuda"
    COMPUTE="float16"
    MODEL="large-v3-turbo"
    echo "   GPU:     Nvidia (CUDA)"
    echo "   Model:   large-v3-turbo"
    echo "   Speed:   ~0.5s per transcription"
else
    echo "⚠️  No Nvidia GPU found — falling back to CPU"
    pip install --quiet torch torchvision
    DEVICE="cpu"
    COMPUTE="int8"
    MODEL="small.en"
    echo "   Device:  CPU"
    echo "   Model:   small.en"
    echo "   Speed:   ~3-5s per transcription"
fi

# ── Python packages ───────────────────────────────────────────────────
echo ""
echo "📦 Installing Python packages..."
pip install --quiet faster-whisper sounddevice scipy \
    numpy pyperclip pynput PyQt6

# ── Patch device/model in script ─────────────────────────────────────
echo ""
echo "⚙️  Configuring script for this machine..."
sed -i "s/device=\"cuda\"/device=\"$DEVICE\"/g" "$INSTALL_DIR/whisper_tray.py"
sed -i "s/compute_type=\"float16\"/compute_type=\"$COMPUTE\"/g" "$INSTALL_DIR/whisper_tray.py"
sed -i "s/\"large-v3-turbo\"/\"$MODEL\"/g" "$INSTALL_DIR/whisper_tray.py"

# ── Passwordless ydotool ──────────────────────────────────────────────
echo ""
echo "🔑 Setting up passwordless ydotool..."
SUDOERS_LINE="$USER ALL=(ALL) NOPASSWD: /usr/bin/ydotool"
if sudo grep -q "$USER.*ydotool" /etc/sudoers 2>/dev/null; then
    echo "   Already configured."
else
    echo "$SUDOERS_LINE" | sudo tee -a /etc/sudoers > /dev/null
    echo "   Done."
fi

# ── Auto-detect audio device ──────────────────────────────────────────
echo ""
echo "🎤 Detecting best audio input device..."
echo "   Please speak into your microphone for a few seconds..."
echo ""

AUDIO_DEVICE=$("$INSTALL_DIR/venv/bin/python3" << 'PYEOF' 2>/dev/null
import sounddevice as sd
import numpy as np

candidates = ['pulse', 'pipewire', 'default']
best_device = 'pulse'
best_level = 0

for name in candidates:
    try:
        audio = sd.rec(int(44100 * 2), samplerate=44100, channels=1,
                       dtype='float32', device=name)
        sd.wait()
        level = float(np.max(np.abs(audio)))
        import sys
        print(f'   {name}: level={level:.4f}', file=sys.stderr)
        if level > best_level:
            best_level = level
            best_device = name
    except Exception as e:
        import sys
        print(f'   {name}: unavailable', file=sys.stderr)

print(best_device)
PYEOF
)
