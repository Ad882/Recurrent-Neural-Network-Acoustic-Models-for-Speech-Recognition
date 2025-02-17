import sounddevice as sd
import numpy as np

sd.default.device = 5  # Replace by the index of the mic to test

SAMPLE_RATE = 16000
DURATION = 3

print("Speak now...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()
print("Average audio level:", np.mean(np.abs(recording)))