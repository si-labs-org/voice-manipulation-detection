# Voice Manipulation Detection

**Deep Learning-Based Speaker Spoofing Detection System**

Master: ISOC (Intelligence et Sécurité des Objets Connectés)
**Authors:** Tahiri Soufiane & Akhchine Ismail 

---

## Overview

Voice Manipulation Detection is a deep learning application designed to detect synthetic, manipulated, or spoofed speech in real-time. The system uses a Transformer-based neural network trained on the **ASVspoof2019 LA** dataset to classify audio as either **AUTHENTIC** (genuine human speech) or **SPOOF** (synthetic/manipulated).

### Key Features

-  **Real-time Recording** - Capture voice directly from your microphone with live waveform visualization
-  **Multi-Format Support** - Load WAV, FLAC, OGG, MP3, M4A, AAC, and other audio formats
-  **Transformer Model** - 86.47% test accuracy on ASVspoof2019 LA dataset
-  **Professional UI** - Dark theme interface with real-time audio visualization
-  **Microphone Selection** - Choose between multiple input devices with device refresh
-  **Visual Feedback** - Animated progress bars, waveform display, and color-coded verdicts

---

## Model Performance

| Model | Test Accuracy | Test EER | Generalization | Status |
|-------|---------------|----------|----------------|--------|
| **Transformer** | **86.47%** | **0.3496** | **7.5x degradation** |  Production |
| BiLSTM | 76.11% | 0.3500 | 18.8x degradation | Advisory |
| CNN | 28.04% | 0.3485 | 54.5x degradation |  Disabled |

**Recommendation:** The Transformer model provides the best generalization with the lowest overfitting ratio. It reliably detects both bonafide (genuine) and spoofed speech in production environments.

---

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (recommended for GPU acceleration)
- Windows / Linux / macOS

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/si-labs-org/voice-manipulation-detection.git
   cd voice-manipulation-detection
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python src/app.py
   ```

---

## Usage

### Launch the Application

```bash
python src/app.py
```

### Recording Voice

1. Click **RECORD AUDIO** button
2. Speak clearly for the next 4 seconds
3. The waveform will display in real-time
4. Transformer will analyze and return **AUTHENTIC** or **SPOOF DETECTED**

### Loading Audio Files

1. Click **LOAD FILE** button
2. Select an audio file (WAV, MP3, FLAC, OGG, M4A, etc.)
3. The app automatically:
   - Resamples to 16 kHz
   - Converts stereo to mono
   - Extracts LFCC features
4. Get instant prediction

### Microphone Selection

- Use the **Mic dropdown** to select input device
- Click **Refresh** to reload available devices
- Default: Intel Microphone Array (device 8)

---

## Architecture

### Feature Extraction

**LFCC (Linear Frequency Cepstral Coefficients):**
- 60 coefficients extracted via DCT (Discrete Cosine Transform)
- 512-point FFT with 160-sample hop length
- Captures both spectral and temporal dynamics

### Model: Transformer Encoder

```
Input: [B, T, 60] LFCC features
  ↓
Input Projection: Linear + LayerNorm + Dropout
  ↓
Positional Encoding (max_len=420)
  ↓
Transformer Encoder (2 layers, 4 heads, 128 d_model)
  ↓
[CLS] Token Classification Head
  ↓
Output: P(spoof) ∈ [0, 1]
```

**Architecture Details:**
- **2 Transformer encoder layers**
- **4 attention heads**
- **128-dimensional embeddings**
- **256-dimensional feedforward networks**
- **GELU activation**
- **SpecAugment regularization** during training

### Preprocessing Pipeline

1. **Audio Loading** (soundfile + pydub fallback)
   - Handles WAV, FLAC, OGG natively
   - Falls back to pydub for MP3, M4A, AAC
   
2. **Resampling** (scipy.signal or numpy fallback)
   - Target: 16 kHz
   - Method: Resample or linear interpolation
   
3. **LFCC Extraction** (custom PyTorch implementation)
   - Log-scale power spectrogram
   - DCT transformation
   - Output: [1, T', 60] tensor

4. **Inference**
   - Batch size: 1
   - Device: CUDA (GPU) or CPU (auto-detect)
   - Softmax → P(spoof)

---

## Project Structure

```
voice-manipulation-detection/
├── src/
│   ├── app.py                    # Main GUI application
│   ├── test_capture.py           # Audio capture testing utility
│   ├── list_devices.py           # List available microphones
│   ├── gen_tts.py                # TTS attack sample generator
│   └── quick_record.py           # Quick recording utility
│
├── notebooks/
│   ├── 00_setup_and_data.ipynb   # Data exploration & setup
│   ├── 01_train_cnn.ipynb        # CNN baseline training
│   ├── 02_train_rnn.ipynb        # BiLSTM training
│   ├── 03_train_transformer.ipynb # Transformer training (best)
│   ├── 04_evaluate.ipynb         # Evaluation & metrics
│   ├── 05_compare_models.ipynb   # Model comparison
│   │
│   ├── models/checkpoints/
│   │   ├── rnn/rnn_best_model.pt
│   │   ├── transformer/transformer_best_model.pt
│   │   └── cnn/cnn_best_model.pt (disabled)
│   │
│   ├── lfcc_cache/               # Cached LFCC features (train/dev/eval)
│   └── experiments/              # Plots, confusion matrices, results
│
├── requirements.txt
├── .gitignore
├── RESULTS_SUMMARY.md            # Detailed training results
└── README.md
```

---

## Training Data

**ASVspoof2019 LA (Logical Access)**
- **Total samples:** 10,913 training (43% subset)
- **Classes:** Bonafide (genuine) & Spoofed (synthetic)
- **Sample rate:** 16 kHz
- **Spoofing attacks:** TTS (Text-to-Speech), Voice Conversion
- **Reference:** https://www.asvspoof.org/asvspoof2019/

---

## Decision Threshold

The model outputs **P(spoof)** ∈ [0, 1]:
- **P(spoof) > 0.5** → **SPOOF DETECTED** 
- **P(spoof) ≤ 0.5** → **AUTHENTIC** 

For strict security applications, consider raising the threshold to 0.6-0.7 to reduce false positives.

---

## Known Limitations

1. **Transformer only:** BiLSTM model disabled due to calibration issues
2. **Fixed 4-second input:** Pad/crop audio to 64,000 samples (4 sec @ 16kHz)
3. **No real-time streaming:** Processes fixed-length audio chunks
4. **Intel microphone focus:** Optimized for Intel Smart Sound Technology devices
5. **Monophonic audio:** Stereo auto-converted to mono

---

## Troubleshooting

### "Failed to load audio: DLL load failed importing _propack"
- **Cause:** scipy/librosa dependency issue on Windows
- **Fix:** Already handled with pydub fallback. No action needed.

### "sounddevice not installed"
- Run: `pip install sounddevice`

### "No models loaded"
- Verify checkpoint files exist in `notebooks/models/checkpoints/`
- Check file paths in `src/app.py`

### Microphone not detected
- Click **Refresh** button in the app
- Run `python src/list_devices.py` to list all devices
- Update `MIC_DEVICE` in `src/app.py` if needed

---

## Development

### Running Jupyter Notebooks

```bash
cd notebooks
jupyter notebook
```

**Training a model takes 15-20 minutes on RTX 3050 GPU.**

### Testing

```bash
python src/test_capture.py  # Test mic input
python src/list_devices.py  # List audio devices
```

---

## References

1. **ASVspoof 2019** - https://www.asvspoof.org/asvspoof2019/
2. **Transformer-based Spoofing Detection** - Müller et al., INTERSPEECH 2021
3. **LFCC Features** - Linear Frequency Cepstral Coefficients for speech recognition

---

## License

Academic project for Master's ISOC program. 

---

## Authors

- **Tahiri Soufiane** 
- **Akhchine Ismail** 

Master's Program: ISOC 

---

**Last Updated:** February 2026

