
import os, sys, numpy as np

try:
    import sounddevice as sd
except:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

SAMPLE_RATE = 16000
RECORD_SECS = 3
MIN_RMS_THRESHOLD = 0.002
TARGET_RMS = 0.08

def list_devices():
    """List all input devices"""
    print("\n" + "="*70)
    print("AVAILABLE INPUT DEVICES")
    print("="*70)
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0:
                name = dev.get('name', f'Device {idx}')
                channels = dev.get('max_input_channels', 0)
                print(f"  {idx}: {name}")
                print(f"     Channels: {channels}")
    except Exception as e:
        print(f"  Error listing devices: {e}")

def find_default():
    """Find default input device"""
    try:
        info = sd.query_devices(kind='input')
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev.get('name') == info.get('name'):
                return idx
        default = sd.default.device
        if isinstance(default, (list, tuple)):
            return default[0] if default[0] is not None else None
        return default
    except:
        return None

def record_audio(device_idx=None):
    """Record 3 seconds of audio"""
    print("\n" + "="*70)
    print("RECORDING TEST")
    print("="*70)

    if device_idx is None:
        device_idx = find_default()
        print(f"Using default device: {device_idx}")
    else:
        print(f"Using selected device: {device_idx}")

    total_samples = int(RECORD_SECS * SAMPLE_RATE)
    print(f"Recording {RECORD_SECS} seconds at {SAMPLE_RATE} Hz...")
    print("Speak normally and clearly!\n")

    try:
        audio = sd.rec(total_samples, samplerate=SAMPLE_RATE, channels=1,
                      dtype='float32', device=device_idx, blocking=False)
        for i in range(RECORD_SECS, 0, -1):
            print(f"  {i}... ", end='', flush=True)
            import time
            time.sleep(1)
        print("\nDone!")
        sd.wait()
        data = audio.flatten().astype(np.float32).copy()
        return data
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def analyze_audio(data):
    """Analyze audio properties"""
    print("\n" + "="*70)
    print("AUDIO ANALYSIS")
    print("="*70)

    rms_raw = float(np.sqrt(np.mean(data ** 2)))
    peak = float(np.max(np.abs(data)))

    print(f"\nBEFORE Normalization:")
    print(f"  RMS Level:           {rms_raw:.6f}")
    print(f"  Peak Level:          {peak:.6f}")
    print(f"  Above threshold:     {'✓ YES' if rms_raw > MIN_RMS_THRESHOLD else '✗ NO'}")

    if rms_raw < MIN_RMS_THRESHOLD:
        print(f"\n  ⚠️  WARNING: Signal too quiet!")
        print(f"  Minimum RMS needed: {MIN_RMS_THRESHOLD:.6f}")
        print(f"  Try: Speaking louder, moving closer to mic, or increasing system volume")
        return None

    # Apply normalization
    normalized = data / (rms_raw + 1e-8)
    normalized = normalized * TARGET_RMS
    normalized = np.clip(normalized, -1.0, 1.0)

    rms_norm = float(np.sqrt(np.mean(normalized ** 2)))
    peak_norm = float(np.max(np.abs(normalized)))

    print(f"\nAFTER Normalization:")
    print(f"  RMS Level:           {rms_norm:.6f} (target: {TARGET_RMS})")
    print(f"  Peak Level:          {peak_norm:.6f}")
    print(f"  Clipped to [-1, 1]:  ✓")

    print(f"\nNormalization Check:")
    if abs(rms_norm - TARGET_RMS) < 0.01:
        print(f"  ✓ RMS normalized correctly to ~{TARGET_RMS}")
    else:
        print(f"  ✗ RMS doesn't match target")

    if peak_norm <= 1.0:
        print(f"  ✓ Peak within [-1, 1] range")
    else:
        print(f"  ✗ Peak exceeds range (will be clipped)")

    return normalized

def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        Voice Spoof Detector - Audio Diagnostics Tool              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # Step 1: List devices
    list_devices()

    # Step 2: Ask user which device to use
    default = find_default()
    device_idx = default

    user_input = input(f"\nEnter device number (default: {default}): ").strip()
    if user_input:
        try:
            device_idx = int(user_input)
        except:
            device_idx = default

    # Step 3: Record audio
    audio = record_audio(device_idx)
    if audio is None:
        print("\nFailed to record audio")
        return

    # Step 4: Analyze
    normalized = analyze_audio(audio)
    if normalized is None:
        return

    # Step 5: Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  ✓ Recording captured successfully")
    print(f"  ✓ Audio normalized to training distribution")
    print(f"  ✓ Ready for spoof detection")
    print(f"\nNext: Run the app with this device selected:")
    print(f"  python app.py")

if __name__ == '__main__':
    main()

