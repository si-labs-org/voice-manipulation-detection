# Voice Manipulation Detection

Deep learning system for detecting synthetic and manipulated speech using Transformer-based classification on LFCC features, trained on the ASVspoof2019 LA dataset.

**Authors:** Tahiri Soufiane & Akhchine Ismail
**Program:** Master ISOC (Intelligence et Sécurité des Objets Connectés)

---

## Problem Statement

Voice spoofing attacks — text-to-speech synthesis, voice conversion, and replay attacks — pose a serious threat to speaker verification systems. This project implements and compares three deep learning architectures for binary classification of audio as **authentic** (bonafide human speech) or **spoof** (synthetic/manipulated).

## Approach

### Feature Extraction

All models use **LFCC (Linear Frequency Cepstral Coefficients)** as input features:

- 60 cepstral coefficients
- 512-point FFT, 160-sample hop length
- Log-scaled power spectrogram → DCT projection
- Fixed input length: 64,000 samples (4 seconds @ 16 kHz)
- Output tensor shape: `[batch, 401, 60]`

### Models Trained

| Model | Architecture | Parameters | Test Accuracy | Test EER | Overfit Ratio |
|-------|-------------|------------|---------------|----------|---------------|
| CNN | 5 residual blocks + BatchNorm + MaxPool | ~50K | 28.04% | 0.3485 | 54.5x |
| BiLSTM | 2-layer BiLSTM + attention pooling | ~400K | 76.11% | 0.3500 | 18.8x |
| **Transformer** | 2-layer encoder + CLS token + SpecAugment | **~300K** | **86.47%** | **0.3496** | **7.5x** |

### Production Model: Transformer

The Transformer was selected for deployment based on:

- Lowest overfit ratio (7.5x val→test degradation vs 54.5x for CNN)
- Highest recall on spoofed samples: 88.98%
- Best F1 score: 0.9218
- SpecAugment regularization prevents memorization on the 43% data subset

**Confusion matrix (test set, 71,237 samples):**

|  | Predicted Bonafide | Predicted Spoof |
|--|-------------------|-----------------|
| **Actual Bonafide** | 4,755 | 2,600 |
| **Actual Spoof** | 7,041 | 56,841 |

### Why CNN and BiLSTM Failed

- **CNN** memorized the validation set entirely (val EER 0.0064, test EER 0.3485). It predicts almost everything as bonafide on unseen data — 19.75% recall is useless.
- **BiLSTM** generalizes better (76.11% accuracy) but the attention mechanism is insufficient without aggressive data augmentation. It still degrades 18.8x from validation to test.

The Transformer wins because SpecAugment (random masking of frequency/time bands) acts as a strong regularizer on limited data, and positional encoding captures temporal artifacts that characterize spoofed speech.

## Application

A desktop GUI application (`src/app.py`) provides real-time spoof detection:

- Record audio directly from any microphone (auto-detects system default)
- Load audio files in any format (WAV, FLAC, MP3, OGG, M4A, AAC, AIFF, OPUS)
- Automatic resampling to 16 kHz mono
- Peak normalization before LFCC extraction
- Transformer inference with spoof probability score and animated result display

### Running the App

```bash
python src/app.py
```

### Testing with TTS Attacks

Generate synthetic speech samples to verify spoof detection:

```bash
python src/gen_tts.py
```

This creates 5 TTS-synthesized WAV files using Microsoft Edge voices. Load them in the app via "LOAD FILE" — the model should classify them as spoof.

### Recording Test

```bash
python src/quick_record.py
```

Records 5 seconds from the system default microphone, normalizes, and saves as `test_audio.wav`.

## Project Structure

```
voice-manipulation-detection/
├── src/
│   ├── app.py                      Main GUI application (Transformer only)
│   ├── gen_tts.py                   TTS attack generator (edge-tts)
│   ├── generate_tts_attack.py       TTS generator with fallback backends
│   └── quick_record.py              CLI recording utility
│
├── notebooks/
│   ├── 00_setup_and_data.ipynb      Dataset exploration and preprocessing
│   ├── 01_train_cnn.ipynb           CNN baseline training
│   ├── 02_train_rnn.ipynb           BiLSTM training
│   ├── 03_train_transformer.ipynb   Transformer training
│   ├── 04_evaluate.ipynb            Test set evaluation (all 3 models)
│   ├── 05_compare_models.ipynb      Comparative analysis and plots
│   ├── models/checkpoints/          Saved model weights
│   ├── lfcc_cache/                  Precomputed LFCC features
│   └── experiments/                 Training curves and figures
│
├── requirements.txt
├── RESULTS_SUMMARY.md               Detailed training results
├── setup.py
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd voice-manipulation-detection
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- PyTorch + torchaudio
- sounddevice, soundfile
- numpy, scipy, librosa
- edge-tts (for TTS attack generation)
- pydub (fallback audio decoder for MP3/M4A)

### Hardware

- CUDA GPU recommended for training (15-20 min on RTX 3050)
- CPU inference works fine for the app (~200ms per prediction)

## Dataset

**ASVspoof2019 Logical Access (LA)**

- Training: 10,913 samples (43% subset) — 1,078 bonafide / 9,835 spoof
- Validation: 10,683 samples (full dev set)
- Test: 71,237 samples (full eval set)
- Spoofing attacks: 13 TTS and voice conversion algorithms (A01–A19)
- Reference: https://www.asvspoof.org/asvspoof2019/

## Decision Boundary

The model outputs P(spoof) ∈ [0, 1]:

- P(spoof) > 0.5 → **SPOOF DETECTED**
- P(spoof) ≤ 0.5 → **AUTHENTIC**

## Known Limitations

1. Fixed 4-second input window (pad or truncate)
2. No streaming/real-time continuous analysis
3. Trained on 43% of the dataset — full dataset training would improve performance
4. EER is high (0.35) because the train subset is small and class-imbalanced (1:9)
5. Mono audio only (stereo auto-downmixed)

## References

1. ASVspoof 2019 — https://www.asvspoof.org/asvspoof2019/
2. Vaswani et al., "Attention Is All You Need," NeurIPS 2017
3. Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR," Interspeech 2019
4. LFCC features for anti-spoofing — Sahidullah & Kinnunen, 2015

---

Master ISOC — Deep Learning Project, February 2026
