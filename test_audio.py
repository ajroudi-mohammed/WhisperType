import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

SAMPLE_RATE = 44100
print("Recording 5 seconds... speak now!")

audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

print(f"Max volume level: {np.max(np.abs(audio))}")
wav.write("test.wav", SAMPLE_RATE, (audio * 32767).astype(np.int16))
print("Saved to test.wav - play it back with:")
print("aplay test.wav")
