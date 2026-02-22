import torch
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'notebooks', 'models', 'checkpoints', 'transformer', 'transformer_best_model.pt')

print(f"Checkpoint path: {CHECKPOINT_PATH}")
print(f"Exists: {os.path.exists(CHECKPOINT_PATH)}")

if not os.path.exists(CHECKPOINT_PATH):
    print("ERROR: Checkpoint not found!")
    sys.exit(1)

ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
print(f"\nCheckpoint keys: {list(ckpt.keys())}")

if 'model_state_dict' in ckpt:
    sd = ckpt['model_state_dict']
    print(f"\nModel has {len(sd)} parameter tensors")
    print("\nFirst 10 layer names:")
    for i, k in enumerate(list(sd.keys())[:10]):
        print(f"  {k}: {sd[k].shape}")

    has_transformer = any('transformer' in k for k in sd.keys())
    has_lstm = any('lstm' in k.lower() for k in sd.keys())
    has_cls_token = any('cls_token' in k for k in sd.keys())

    print(f"\nModel type detection:")
    print(f"  Has 'transformer' layers: {has_transformer}")
    print(f"  Has 'lstm' layers: {has_lstm}")
    print(f"  Has 'cls_token': {has_cls_token}")

    if has_transformer and has_cls_token and not has_lstm:
        print("\n✓ This is the TRANSFORMER model (correct)")
    elif has_lstm:
        print("\n✗ This is the BiLSTM model (WRONG!)")
    else:
        print("\n? Unknown model architecture")

print("\n--- Testing inference ---")

from app import TransformerSpoofDetector, LFCCExtractor, _process_audio_for_inference

device = torch.device('cpu')
model = TransformerSpoofDetector(input_dim=60, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.3)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

lfcc_extractor = LFCCExtractor()

np.random.seed(42)
fake_audio = np.random.randn(64000).astype(np.float32) * 0.1
processed = _process_audio_for_inference(fake_audio)

waveform = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
lfcc = lfcc_extractor(waveform)
lfcc = lfcc.transpose(1, 2)

with torch.no_grad():
    logits = model(lfcc)
    probs = torch.softmax(logits, dim=1)
    spoof_prob = probs[0, 1].item()

print(f"Random noise test:")
print(f"  Logits: {logits[0].tolist()}")
print(f"  Probs [bonafide, spoof]: {probs[0].tolist()}")
print(f"  Spoof probability: {spoof_prob:.4f}")

if spoof_prob > 0.5:
    print("  Result: SPOOF")
else:
    print("  Result: AUTHENTIC")

print("\n--- Testing with silence ---")
silent_audio = np.zeros(64000, dtype=np.float32)
processed_silent = _process_audio_for_inference(silent_audio)
waveform_silent = torch.from_numpy(processed_silent).float().unsqueeze(0).unsqueeze(0)

if waveform_silent.shape[-1] < 64000:
    waveform_silent = torch.nn.functional.pad(waveform_silent, (0, 64000 - waveform_silent.shape[-1]))

lfcc_silent = lfcc_extractor(waveform_silent)
lfcc_silent = lfcc_silent.transpose(1, 2)

with torch.no_grad():
    logits_silent = model(lfcc_silent)
    probs_silent = torch.softmax(logits_silent, dim=1)
    spoof_prob_silent = probs_silent[0, 1].item()

print(f"Silent audio test:")
print(f"  Spoof probability: {spoof_prob_silent:.4f}")

print("\nDone.")

