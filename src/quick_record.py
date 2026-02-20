#!/usr/bin/env python
"""Test audio capture and save WAV file for inspection."""
import sounddevice as sd
import soundfile as sf
import numpy as np
import sys

SAMPLE_RATE = 16000
RECORD_SECS = 4
DEVICE = 8

print("Recording 4 seconds...", file=sys.stderr, flush=True)
audio = sd.rec(int(RECORD_SECS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=DEVICE)
sd.wait()

audio = audio.flatten()
print(f"Captured: {len(audio)} samples", file=sys.stderr, flush=True)
print(f"Max amplitude: {np.max(np.abs(audio)):.4f}", file=sys.stderr, flush=True)
print(f"RMS: {np.sqrt(np.mean(audio**2)):.6f}", file=sys.stderr, flush=True)

sf.write('test_audio.wav', audio, SAMPLE_RATE)
print("Saved to test_audio.wav", file=sys.stderr, flush=True)

