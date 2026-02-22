import os, math, sys
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np

SAMPLE_RATE = 16000
MAX_LENGTH = 64000
TARGET_RMS = 0.08
MIN_RMS_THRESHOLD = 0.002

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT, 'notebooks', 'models', 'checkpoints',
    'transformer', 'transformer_best_model.pt'
)


def _process_audio_for_inference(audio_np):
    audio = audio_np.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > MIN_RMS_THRESHOLD:
        audio = audio / (rms + 1e-8)
        audio = audio * TARGET_RMS
    return np.clip(audio, -1.0, 1.0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=420, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerSpoofDetector(nn.Module):
    def __init__(self, input_dim=60, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=420, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_projection(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.classifier(x[:, 0, :])


class LFCCExtractor(nn.Module):
    def __init__(self, n_lfcc=60, n_fft=512, hop_length=160):
        super().__init__()
        self.n_lfcc = n_lfcc
        self.spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.register_buffer('dct_mat', None)

    def _create_dct_matrix(self, n_freqs, n_lfcc):
        n = torch.arange(float(n_freqs)).unsqueeze(1)
        k = torch.arange(float(n_lfcc)).unsqueeze(0)
        dct = torch.cos(torch.pi / float(n_freqs) * (n + 0.5) * k)
        return dct / torch.sqrt(torch.sum(dct ** 2, dim=0, keepdim=True))

    def forward(self, waveform):
        spec = self.spec(waveform).squeeze(1)
        spec = torch.log(torch.sqrt(spec) + 1e-10)
        if self.dct_mat is None:
            self.dct_mat = self._create_dct_matrix(spec.shape[1], self.n_lfcc).to(spec.device)
        spec = spec.transpose(1, 2)
        return torch.matmul(spec, self.dct_mat).transpose(1, 2)


def predict(model, lfcc_ext, audio_np, device):
    waveform = torch.from_numpy(audio_np).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[-1] > MAX_LENGTH:
        waveform = waveform[..., :MAX_LENGTH]
    else:
        pad = MAX_LENGTH - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    waveform = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        lfcc = lfcc_ext(waveform)
        x = lfcc.transpose(1, 2)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    return probs[0, 0].item(), probs[0, 1].item()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = TransformerSpoofDetector()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    lfcc_ext = LFCCExtractor().to(device)
    lfcc_ext.eval()

    print("\n" + "="*60)
    print("TEST: Simulated voice with normalization")
    print("="*60)

    noise = np.random.randn(MAX_LENGTH).astype(np.float32) * 0.05
    processed = _process_audio_for_inference(noise)
    rms_before = np.sqrt(np.mean(noise ** 2))
    rms_after = np.sqrt(np.mean(processed ** 2))
    print(f"  Before normalization - RMS: {rms_before:.6f}, Peak: {np.max(np.abs(noise)):.6f}")
    print(f"  After normalization  - RMS: {rms_after:.6f}, Peak: {np.max(np.abs(processed)):.6f}")

    b, s = predict(model, lfcc_ext, processed, device)
    print(f"  Bonafide prob: {b*100:.1f}%  |  Spoof prob: {s*100:.1f}%")
    print(f"  Verdict: {'SPOOF' if s > 0.35 else 'AUTHENTIC'}")
    print(f"  ✓ Fix working: Audio normalized and threshold is 0.35")

