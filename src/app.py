import os, math, time, threading, queue
import tkinter as tk
from tkinter import filedialog, font
from typing import Literal, Optional, cast
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchaudio.transforms as T
    TORCH_OK = True
except ImportError:
    torch = None
    nn = None
    T = None
    TORCH_OK = False
    print("ERROR: torch not installed. Run: pip install torch torchaudio")

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    sd = None
    SD_OK = False
    print("WARNING: sounddevice not installed. Run: pip install sounddevice")

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    sf = None
    SF_OK = False
    print("WARNING: soundfile not installed. Run: pip install soundfile")

SAMPLE_RATE  = 16000
MAX_LENGTH   = 64000
RECORD_SECS  = 4

MIC_DEVICE   = 8

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINTS = {
    'BiLSTM':      os.path.join(_PROJECT_ROOT, 'notebooks', 'models', 'checkpoints', 'rnn', 'rnn_best_model.pt'),
    'Transformer': os.path.join(_PROJECT_ROOT, 'notebooks', 'models', 'checkpoints', 'transformer', 'transformer_best_model.pt'),
}

BG          = '#0d1117'
BG2         = '#161b22'
BG3         = '#21262d'
ACCENT      = '#58a6ff'
ACCENT2     = '#79c0ff'
GREEN       = '#3fb950'
GREEN_DIM   = '#238636'
RED         = '#f85149'
RED_DIM     = '#da3633'
TEXT        = '#e6edf3'
TEXT_DIM    = '#8b949e'
BORDER      = '#30363d'
YELLOW      = '#d29922'
SURFACE     = '#1f2428'

class BiLSTMSpoofDetector(nn.Module):
    def __init__(self, input_size=60, hidden_size=128, num_layers=2,
                 num_classes=2, dropout=0.5, bidirectional=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_out, lstm_out // 2), nn.Tanh(),
            nn.Linear(lstm_out // 2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(lstm_out, lstm_out // 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(lstm_out // 2, num_classes)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn * lstm_out, dim=1)
        return self.classifier(context)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=420, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
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
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=420, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_encoder(x)
        x   = self.transformer_encoder(x)
        return self.classifier(x[:, 0, :])


class LFCCExtractor(nn.Module):
    def __init__(self, n_lfcc=60, n_fft=512, hop_length=160):
        super().__init__()
        self.n_lfcc = n_lfcc
        self.spec   = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.register_buffer('dct_mat', None)

    def _create_dct_matrix(self, n_freqs, n_lfcc):
        n   = torch.arange(float(n_freqs)).unsqueeze(1)
        k   = torch.arange(float(n_lfcc)).unsqueeze(0)
        dct = torch.cos(torch.pi / float(n_freqs) * (n + 0.5) * k)
        return dct / torch.sqrt(torch.sum(dct**2, dim=0, keepdim=True))

    def forward(self, waveform):
        spec = self.spec(waveform).squeeze(1)
        spec = torch.log(torch.sqrt(spec) + 1e-10)
        if self.dct_mat is None:
            self.dct_mat = self._create_dct_matrix(
                spec.shape[1], self.n_lfcc).to(spec.device)
        spec = spec.transpose(1, 2)
        return torch.matmul(spec, self.dct_mat).transpose(1, 2)


class InferenceEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lfcc   = LFCCExtractor().to(self.device)
        self.models = {}
        self.errors = []

    def load_models(self):
        constructors = {
            'BiLSTM':      lambda: BiLSTMSpoofDetector(
                input_size=60, hidden_size=128, num_layers=2, dropout=0.5),
            'Transformer': lambda: TransformerSpoofDetector(
                input_dim=60, d_model=128, nhead=4, num_layers=2,
                dim_feedforward=256, dropout=0.3),
        }
        for name, path in CHECKPOINTS.items():
            if not os.path.exists(path):
                self.errors.append(f'{name}: checkpoint not found at {path}')
                continue
            try:
                model = constructors[name]()
                ckpt  = torch.load(path, map_location=self.device,
                                   weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                model.to(self.device).eval()
                self.models[name] = model
            except Exception as e:
                self.errors.append(f'{name}: {e}')

    def preprocess(self, waveform_np):
        waveform = torch.from_numpy(waveform_np).float()
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[-1] > MAX_LENGTH:
            waveform = waveform[..., :MAX_LENGTH]
        else:
            pad = MAX_LENGTH - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        waveform = waveform.unsqueeze(0).to(self.device)

        lfcc = self.lfcc(waveform)

        lfcc = lfcc.transpose(1, 2)

        return lfcc

    @torch.no_grad()
    def predict(self, waveform_np):
        x = self.preprocess(waveform_np)

        results = {}
        for name, model in self.models.items():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            results[name] = prob
        return results


class SpoofDetectorApp:
    def __init__(self, root):
        self.root    = root
        self.engine  = InferenceEngine()
        self.q       = queue.Queue()
        self.recording     = False
        self.recorded_data = None
        self.anim_id       = None
        self.bar_anim_ids  = {}  # Track bar animation IDs per model

        self._setup_window()
        self._build_ui()
        self._load_models_async()
        self._poll_queue()

    def _setup_window(self):
        self.root.title('Voice Authentication System')
        self.root.configure(bg=BG)
        self.root.geometry('920x720')
        self.root.minsize(920, 720)
        self.root.resizable(True, True)
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f'+{x}+{y}')

    def _build_ui(self):
        self.f_title   = tk.font.Font(family='Segoe UI', size=11, weight='bold')
        self.f_big     = tk.font.Font(family='Segoe UI Semibold', size=18, weight='bold')
        self.f_label   = tk.font.Font(family='Segoe UI', size=10)
        self.f_small   = tk.font.Font(family='Segoe UI', size=9)
        self.f_verdict = tk.font.Font(family='Segoe UI Semibold', size=24, weight='bold')
        self.f_score   = tk.font.Font(family='Segoe UI', size=10, weight='bold')
        self.f_mono    = tk.font.Font(family='Consolas', size=9)

        hdr = tk.Frame(self.root, bg=BG, pady=20)
        hdr.pack(fill='x', padx=40)

        title_frame = tk.Frame(hdr, bg=BG)
        title_frame.pack(side='left')

        tk.Label(title_frame, text='VOICE AUTHENTICATION',
                 font=self.f_big, bg=BG, fg=TEXT).pack(anchor='w')
        tk.Label(title_frame, text='Deep Learning Spoofing Detection System',
                 font=self.f_small, bg=BG, fg=TEXT_DIM).pack(anchor='w')

        # Status indicator (right side)
        status_frame = tk.Frame(hdr, bg=BG)
        status_frame.pack(side='right')

        self.status_dot = tk.Canvas(status_frame, width=10, height=10,
                                     bg=BG, highlightthickness=0)
        self.status_dot.pack(side='left', padx=(0, 8))
        self._draw_status_dot(YELLOW)

        self.status_lbl = tk.Label(status_frame, text='Initializing...',
                                   font=self.f_mono, bg=BG, fg=TEXT_DIM)
        self.status_lbl.pack(side='left')

        # Divider
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill='x', padx=40)

        # ── MAIN CONTENT ──────────────────────────────────────────────────────
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill='both', expand=True, padx=40, pady=20)

        # ── WAVEFORM PANEL ────────────────────────────────────────────────────
        wave_panel = tk.Frame(content, bg=BG2, highlightbackground=BORDER,
                              highlightthickness=1)
        wave_panel.pack(fill='x', pady=(0, 20))

        wave_header = tk.Frame(wave_panel, bg=BG2)
        wave_header.pack(fill='x', padx=16, pady=(12, 0))

        tk.Label(wave_header, text='AUDIO WAVEFORM', font=self.f_small,
                 bg=BG2, fg=TEXT_DIM).pack(side='left')

        self.duration_lbl = tk.Label(wave_header, text='Duration: --',
                                     font=self.f_mono, bg=BG2, fg=TEXT_DIM)
        self.duration_lbl.pack(side='right')

        self.canvas = tk.Canvas(wave_panel, width=820, height=120,
                                bg=BG2, highlightthickness=0)
        self.canvas.pack(padx=16, pady=(8, 16))
        self._draw_flat_wave()

        # ── CONTROLS PANEL ────────────────────────────────────────────────────
        ctrl_panel = tk.Frame(content, bg=BG)
        ctrl_panel.pack(fill='x', pady=(0, 20))

        # Record button
        self.rec_btn = tk.Button(
            ctrl_panel, text='RECORD AUDIO',
            font=self.f_title, bg=ACCENT, fg='#ffffff',
            activebackground=ACCENT2, activeforeground='#ffffff',
            relief='flat', padx=32, pady=14, cursor='hand2',
            command=self._on_record, bd=0
        )
        self.rec_btn.pack(side='left', padx=(0, 12))

        # Load button
        self.load_btn = tk.Button(
            ctrl_panel, text='LOAD FILE',
            font=self.f_title, bg=BG3, fg=TEXT,
            activebackground=BORDER, activeforeground=TEXT,
            relief='flat', padx=32, pady=14, cursor='hand2',
            command=self._on_load, bd=0,
            highlightbackground=BORDER, highlightthickness=1
        )
        self.load_btn.pack(side='left', padx=(0, 12))

        # Recording time indicator
        self.countdown_lbl = tk.Label(ctrl_panel, text='',
                                      font=self.f_title, bg=BG, fg=YELLOW)
        self.countdown_lbl.pack(side='left', padx=(20, 0))

        # Mic device selector
        device_frame = tk.Frame(ctrl_panel, bg=BG)
        device_frame.pack(side='right')

        tk.Label(device_frame, text='Mic:', font=self.f_small,
                 bg=BG, fg=TEXT_DIM).pack(side='left', padx=(0, 6))

        self.device_var = tk.StringVar(value='Loading devices...')
        self.device_menu = tk.OptionMenu(device_frame, self.device_var, 'Loading devices...')
        self.device_menu.config(
            font=self.f_small, bg=BG3, fg=TEXT,
            activebackground=BORDER, activeforeground=TEXT,
            highlightthickness=1, relief='flat', padx=8
        )
        self.device_menu.pack(side='left')

        self.refresh_btn = tk.Button(
            device_frame, text='Refresh',
            font=self.f_small, bg=BG3, fg=TEXT,
            activebackground=BORDER, activeforeground=TEXT,
            relief='flat', padx=10, pady=2, cursor='hand2',
            command=self._refresh_devices, bd=0,
            highlightbackground=BORDER, highlightthickness=1
        )
        self.refresh_btn.pack(side='left', padx=(8, 0))

        self._refresh_devices()

        # ── RESULTS PANEL ─────────────────────────────────────────────────────
        results_panel = tk.Frame(content, bg=BG2, highlightbackground=BORDER,
                                 highlightthickness=1)
        results_panel.pack(fill='x', pady=(0, 20))

        results_header = tk.Frame(results_panel, bg=BG2)
        results_header.pack(fill='x', padx=16, pady=(12, 8))

        tk.Label(results_header, text='DETECTION RESULTS', font=self.f_small,
                 bg=BG2, fg=TEXT_DIM).pack(side='left')

        self.bars = {}
        self.score_labels = {}
        self.verdict_labels = {}

        for name in ['Transformer']:  # Only show Transformer
            row = tk.Frame(results_panel, bg=BG2, pady=8)
            row.pack(fill='x', padx=16)

            # Model name
            name_lbl = tk.Label(row, text='Spoofing Detection Score', font=self.f_score,
                                bg=BG2, fg=TEXT, width=25, anchor='w')
            name_lbl.pack(side='left')

            # Progress bar container - taller for visibility
            bar_container = tk.Frame(row, bg=BG3, height=20)
            bar_container.pack(side='left', fill='x', expand=True, padx=(12, 12))
            bar_container.pack_propagate(False)

            bar_fill = tk.Frame(bar_container, bg=TEXT_DIM)
            bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0)
            self.bars[name] = (bar_container, bar_fill)

            # Score percentage
            score_lbl = tk.Label(row, text='--', font=self.f_mono,
                                 bg=BG2, fg=TEXT_DIM, width=8, anchor='e')
            score_lbl.pack(side='left')
            self.score_labels[name] = score_lbl

            # Verdict tag
            verdict_lbl = tk.Label(row, text='', font=self.f_small,
                                   bg=BG2, fg=TEXT_DIM, width=10, anchor='w')
            verdict_lbl.pack(side='left', padx=(12, 0))
            self.verdict_labels[name] = verdict_lbl

        # Padding at bottom of results panel
        tk.Frame(results_panel, bg=BG2, height=12).pack()

        # ── VERDICT PANEL ─────────────────────────────────────────────────────
        verdict_panel = tk.Frame(content, bg=BG2, highlightbackground=BORDER,
                                 highlightthickness=1)
        verdict_panel.pack(fill='x')

        verdict_inner = tk.Frame(verdict_panel, bg=BG2, pady=24)
        verdict_inner.pack()

        tk.Label(verdict_inner, text='FINAL VERDICT', font=self.f_small,
                 bg=BG2, fg=TEXT_DIM).pack()

        self.verdict_big = tk.Label(verdict_inner, text='AWAITING INPUT',
                                    font=self.f_verdict, bg=BG2, fg=TEXT_DIM)
        self.verdict_big.pack(pady=(8, 0))

        self.verdict_desc = tk.Label(verdict_inner, text='Record or load audio to analyze',
                                     font=self.f_small, bg=BG2, fg=TEXT_DIM)
        self.verdict_desc.pack(pady=(4, 0))

        # ── FOOTER ────────────────────────────────────────────────────────────
        footer = tk.Frame(self.root, bg=BG, pady=12)
        footer.pack(side='bottom', fill='x')

        tk.Label(footer, text='ASVspoof2019 LA  |  LFCC Features  |  Transformer Model (86.47% Accuracy)',
                 font=self.f_mono, bg=BG, fg=TEXT_DIM).pack()

    def _draw_status_dot(self, color):
        self.status_dot.delete('all')
        self.status_dot.create_oval(1, 1, 9, 9, fill=color, outline=color)

    # ── WAVEFORM DRAWING ─────────────────────────────────────────────────────
    def _draw_flat_wave(self):
        self.canvas.delete('all')
        w, h = self.canvas.winfo_width() or 820, self.canvas.winfo_height() or 120
        mid = h // 2
        self.canvas.create_line(0, mid, w, mid, fill=BORDER, width=1)

    def _draw_waveform(self, data, color=ACCENT):
        self.canvas.delete('all')
        w, h = self.canvas.winfo_width() or 820, self.canvas.winfo_height() or 120
        mid = h // 2

        if len(data) == 0:
            return

        max_amp = np.max(np.abs(data))
        if max_amp > 0:
            normalized = data / max_amp
        else:
            normalized = data

        step = max(1, len(normalized) // w)
        pts = []

        for i in range(w):
            idx = i * step
            if idx < len(normalized):
                chunk = normalized[idx:idx+step]
                amp = float(np.mean(np.abs(chunk))) * mid * 0.9
                pts.extend([i, mid - amp])

        if len(pts) >= 4:
            self.canvas.create_line(pts, fill=color, width=2, smooth=True)

        self.canvas.create_line(0, mid, w, mid, fill=BORDER, width=1, dash=(4, 4))

        duration_sec = len(data) / SAMPLE_RATE
        self.duration_lbl.config(text=f'Duration: {duration_sec:.2f}s  |  Max Amp: {max_amp:.3f}')

    def _animate_recording(self, frame=0):
        self.canvas.delete('all')
        w, h = self.canvas.winfo_width() or 820, self.canvas.winfo_height() or 120
        mid = h // 2
        pts = []

        for i in range(w):
            phase = (i / w) * 4 * np.pi + frame * 0.3
            amp = (15 + 10 * np.sin(frame * 0.1)) * (0.5 * np.sin(phase) + 0.3 * np.sin(phase * 2))
            pts.extend([i, mid - amp])

        self.canvas.create_line(pts, fill=RED, width=2, smooth=True)
        self.canvas.create_line(0, mid, w, mid, fill=BORDER, width=1, dash=(4, 4))
        self.anim_id = self.root.after(40, self._animate_recording, frame + 1)

    def _stop_animation(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    # ── STATUS ────────────────────────────────────────────────────────────────
    def _set_status(self, text, color=TEXT_DIM):
        self.status_lbl.config(text=text, fg=color)
        self._draw_status_dot(color)

    def _set_buttons(self, enabled):
        state = cast(Literal["normal", "disabled"], "normal" if enabled else "disabled")
        self.rec_btn.config(state=state)
        self.load_btn.config(state=state)
        self.refresh_btn.config(state=state)
        self.device_menu.config(state=state)

    def _refresh_devices(self):
        devices = list_input_devices()
        menu = self.device_menu['menu']
        menu.delete(0, 'end')

        if not devices:
            self.device_var.set('No input devices')
            menu.add_command(label='No input devices', command=lambda: None)
            return

        self.device_map = {}
        for idx, name in devices:
            label = f'{idx}: {name}'
            self.device_map[label] = idx
            menu.add_command(label=label, command=lambda v=label: self.device_var.set(v))

        default_label = None
        for label, idx in self.device_map.items():
            if idx == MIC_DEVICE:
                default_label = label
                break
        if default_label is None:
            default_label = next(iter(self.device_map.keys()))
        self.device_var.set(default_label)

    def _get_selected_device_index(self) -> Optional[int]:
        label = self.device_var.get()
        if not hasattr(self, 'device_map'):
            return None
        return self.device_map.get(label)

    # ── MODEL LOADING ─────────────────────────────────────────────────────────
    def _load_models_async(self):
        def _load():
            self.engine.load_models()
            self.q.put(('models_loaded', None))
        threading.Thread(target=_load, daemon=True).start()

    # ── RECORDING ─────────────────────────────────────────────────────────────
    def _on_record(self):
        if not SD_OK or sd is None:
            self._set_status('sounddevice not installed!', RED)
            return
        self._set_buttons(False)
        self._reset_results()
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        try:
            self.q.put(('recording_start', None))

            total_frames = int(RECORD_SECS * SAMPLE_RATE)
            selected_idx = self._get_selected_device_index()
            device = resolve_input_device(selected_idx)

            if selected_idx is not None and device is None:
                raise RuntimeError('Invalid microphone device index. Select another device.')

            audio = sd.rec(total_frames, samplerate=SAMPLE_RATE, channels=1,
                           dtype='float32', device=device)

            for sec_remaining in range(RECORD_SECS, 0, -1):
                self.q.put(('countdown', sec_remaining))
                time.sleep(1)

            sd.stop()
            sd.wait()
            data = audio.flatten()
            self.q.put(('recording_done', data))
            self.q.put(('run_inference', data))
        except Exception as e:
            self.q.put(('error', f'Recording failed: {e}'))

    # ── LOAD FILE ─────────────────────────────────────────────────────────────
    def _on_load(self):
        path = filedialog.askopenfilename(
            title='Select Audio File',
            filetypes=[('Audio files', '*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.wma *.aiff *.opus'),
                       ('All files', '*.*')]
        )
        if not path:
            return
        self._set_buttons(False)
        self._reset_results()
        threading.Thread(target=self._load_thread, args=(path,), daemon=True).start()

    def _load_thread(self, path):
        try:
            data, sr = None, None

            if not SF_OK or sf is None:
                raise RuntimeError('soundfile not installed')

            try:
                data, sr = sf.read(path)
            except Exception as sf_err:
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(path)

                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    if audio.channels == 2:
                        samples = samples.reshape((-1, 2)).mean(axis=1)
                    else:
                        samples = samples.reshape((-1, audio.channels)).mean(axis=1) if audio.channels > 1 else samples

                    samples = samples / 32768.0
                    data = samples
                    sr = audio.frame_rate
                except Exception as pydub_err:
                    raise RuntimeError(f'Both soundfile and pydub failed: {sf_err} | {pydub_err}')

            if data is None:
                raise RuntimeError('Failed to load audio file')

            if len(data.shape) > 1:
                data = data.mean(axis=1)

            if sr != SAMPLE_RATE:
                try:
                    from scipy import signal
                    num_samples = int(len(data) * SAMPLE_RATE / sr)
                    data = signal.resample(data, num_samples)
                except ImportError:
                    indices = np.linspace(0, len(data) - 1, int(len(data) * SAMPLE_RATE / sr))
                    data = np.interp(indices, np.arange(len(data)), data)

            data = data.astype(np.float32)

            self.q.put(('file_loaded', data))
            self.q.put(('run_inference', data))
        except Exception as e:
            self.q.put(('error', f'Failed to load audio: {e}'))

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    def _inference_thread(self, data):
        try:
            if not self.engine.models:
                raise RuntimeError('No models loaded')
            results = self.engine.predict(data)
            self.q.put(('results', results))
        except Exception as e:
            self.q.put(('error', str(e)))

    # ── RESULTS DISPLAY ───────────────────────────────────────────────────────
    def _reset_results(self):
        # Cancel all bar animations
        for name in list(self.bar_anim_ids.keys()):
            self.root.after_cancel(self.bar_anim_ids[name])
        self.bar_anim_ids.clear()

        for name in ['Transformer']:
            bar_bg, bar_fill = self.bars[name]
            bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0)
            bar_fill.config(bg=TEXT_DIM)
            self.score_labels[name].config(text='--', fg=TEXT_DIM)
            self.verdict_labels[name].config(text='', fg=TEXT_DIM)
        self.verdict_big.config(text='ANALYZING...', fg=YELLOW)
        self.verdict_desc.config(text='Processing audio sample...', fg=TEXT_DIM)
        self.countdown_lbl.config(text='')

    def _animate_bar(self, name, target_pct, color, frame=0, max_frames=30):
        if name in self.bar_anim_ids:
            self.root.after_cancel(self.bar_anim_ids[name])

        _, bar_fill = self.bars[name]

        if frame >= max_frames:
            bar_fill.place(x=0, y=0, relheight=1.0, relwidth=target_pct)
            bar_fill.config(bg=color)
            if name in self.bar_anim_ids:
                del self.bar_anim_ids[name]
            return

        progress = frame / max_frames
        eased = 1 - (1 - progress) ** 2
        current_pct = target_pct * eased

        bar_fill.place(x=0, y=0, relheight=1.0, relwidth=max(0, current_pct))
        bar_fill.config(bg=color)

        self.bar_anim_ids[name] = self.root.after(16, self._animate_bar, name, target_pct, color, frame + 1, max_frames)

    def _show_results(self, results):
        self._stop_animation()

        # Only use Transformer (86.47% test accuracy — solid)
        transformer_score = results.get('Transformer', 0)
        is_spoof = transformer_score > 0.5
        bar_color = RED if is_spoof else GREEN
        pct_txt = f'{transformer_score*100:.1f}%'
        verdict = 'SPOOF' if is_spoof else 'AUTHENTIC'
        v_color = RED if is_spoof else GREEN

        self._animate_bar('Transformer', transformer_score, bar_color)
        self.score_labels['Transformer'].config(text=pct_txt, fg=bar_color)
        self.verdict_labels['Transformer'].config(text=verdict, fg=v_color)

        if is_spoof:
            self.verdict_big.config(text='SPOOF DETECTED', fg=RED)
            self.verdict_desc.config(text='Audio sample appears to be synthetic or manipulated', fg=RED_DIM)
        else:
            self.verdict_big.config(text='AUTHENTIC', fg=GREEN)
            self.verdict_desc.config(text='Audio sample appears to be genuine human speech', fg=GREEN_DIM)

        self._set_buttons(True)
        self._set_status(f'Analysis complete', GREEN)

    def _poll_queue(self):
        try:
            while True:
                msg, data = self.q.get_nowait()

                if msg == 'models_loaded':
                    if self.engine.models:
                        loaded = ', '.join(self.engine.models.keys())
                        self._set_status(f'Ready  ·  {loaded}', GREEN)
                        self._set_buttons(True)
                    else:
                        self._set_status('No models loaded', RED)
                        self._set_buttons(False)
                    if self.engine.errors:
                        err_text = '\n'.join(self.engine.errors)
                        print(f"Model loading errors:\n{err_text}")
                        self.verdict_big.config(text='MODEL ERROR', fg=RED)
                        self.verdict_desc.config(
                            text=self.engine.errors[0][:80], fg=RED_DIM)

                elif msg == 'recording_start':
                    self._set_status('Recording...', RED)
                    self._animate_recording()

                elif msg == 'countdown':
                    self.countdown_lbl.config(text=f'Recording: {data}s')

                elif msg == 'recording_done':
                    self._stop_animation()
                    self.countdown_lbl.config(text='Finished')  # Show confirmation
                    self._draw_waveform(data, color=ACCENT)
                    self._set_status('Running inference...', YELLOW)
                    self.root.after(500, lambda: self.countdown_lbl.config(text=''))

                elif msg == 'file_loaded':
                    self._draw_waveform(data, color=ACCENT2)
                    self._set_status('Running inference...', YELLOW)

                elif msg == 'run_inference':
                    threading.Thread(target=self._inference_thread,
                                     args=(data,), daemon=True).start()

                elif msg == 'results':
                    self._show_results(data)

                elif msg == 'error':
                    self._set_status(f'Error: {data}', RED)
                    self.verdict_big.config(text='ERROR', fg=RED)
                    self._set_buttons(True)
                    self._stop_animation()

        except queue.Empty:
            pass

        self.root.after(50, self._poll_queue)


def list_input_devices():
    if not SD_OK or sd is None:
        return []
    devices = []
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get('max_input_channels', 0) > 0:
                devices.append((idx, dev.get('name', f'Device {idx}')))
    except Exception:
        return []
    return devices


def resolve_input_device(device_index: Optional[int]) -> Optional[int]:
    if not SD_OK or sd is None:
        return None
    try:
        if device_index is None:
            return None
        devices = sd.query_devices()
        if not (0 <= device_index < len(devices)):
            return None
        if devices[device_index].get('max_input_channels', 0) <= 0:
            return None
        return device_index
    except Exception:
        return None

if __name__ == '__main__':
    if not TORCH_OK:
        print("Install PyTorch first: pip install torch torchaudio")
        exit(1)

    root = tk.Tk()
    app  = SpoofDetectorApp(root)
    root.mainloop()
