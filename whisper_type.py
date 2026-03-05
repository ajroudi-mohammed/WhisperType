import tempfile
import os
import subprocess
import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import pyperclip
from pynput import keyboard
from faster_whisper import WhisperModel

print("Loading model...")
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
print("✅ Model ready! Hold F9 to talk, ESC to quit.")

SAMPLE_RATE = 44100
recording = False
stop_event = threading.Event()

def transcribe_and_type(audio_data):
    try:
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        wav.write(tmp_path, SAMPLE_RATE, audio_int16)

        segments, _ = model.transcribe(tmp_path, beam_size=5, language="en")
        text = " ".join([s.text for s in segments]).strip()
        os.unlink(tmp_path)

        if text:
            print(f"📝 {text}")
            subprocess.run(["wl-copy", text])
            subprocess.run(["sudo", "ydotool", "type", "--delay", "50", "--", text])
    except Exception as e:
        print(f"Transcription error: {e}")

def record_audio():
    global recording
    chunks = []
    print("🎙 Recording...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while recording:
            chunk, _ = stream.read(1024)
            chunks.append(chunk)
    if chunks:
        audio_data = np.concatenate(chunks, axis=0)
        print("⚙️  Transcribing...")
        threading.Thread(target=transcribe_and_type, args=(audio_data,), daemon=True).start()

record_thread = None

def on_press(key):
    global recording, record_thread
    if key == keyboard.Key.f9 and not recording:
        recording = True
        record_thread = threading.Thread(target=record_audio, daemon=True)
        record_thread.start()

def on_release(key):
    global recording
    if key == keyboard.Key.f9 and recording:
        recording = False

    if key == keyboard.Key.esc:
        return False

print("Listening for F9...")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
