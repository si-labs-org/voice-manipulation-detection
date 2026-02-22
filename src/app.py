import os, math, time, threading, queue
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchaudio.transforms as T
    TORCH_OK = True
except ImportError:
    torch = None; nn = None; T = None; TORCH_OK = False

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    sd = None; SD_OK = False

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    sf = None; SF_OK = False

SAMPLE_RATE       = 16000
MAX_LENGTH        = 64000
RECORD_SECS       = 5
MIN_RMS_THRESHOLD = 0.002
TARGET_RMS        = 0.08
PRE_EMPHASIS_COEF = 0.97
SPOOF_THRESHOLD   = 0.35

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT, 'notebooks', 'models', 'checkpoints',
    'transformer', 'transformer_best_model.pt'
)

BG        = '#0f1318'
BG2       = '#181d25'
BG3       = '#232a35'
ACCENT    = '#4f8ff7'
ACCENT2   = '#6ba3fa'
GREEN     = '#34d399'
GREEN_DIM = '#22916a'
RED       = '#ef4444'
RED_DIM   = '#b91c1c'
TEXT      = '#e2e8f0'
TEXT_DIM  = '#7c8594'
BORDER    = '#2d3544'
YELLOW    = '#eab308'
SURFACE   = '#1e242e'


# ── ARCHITECTURE ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=420, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
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
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x   = self.input_projection(x)
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
            self.dct_mat = self._create_dct_matrix(spec.shape[1], self.n_lfcc).to(spec.device)
        spec = spec.transpose(1, 2)
        return torch.matmul(spec, self.dct_mat).transpose(1, 2)


# ── AUDIO ─────────────────────────────────────────────────────────────────────
def _process_audio_for_inference(audio_np):
    audio = audio_np.astype(np.float32)
    rms   = np.sqrt(np.mean(audio**2))
    if rms > MIN_RMS_THRESHOLD:
        audio = audio / (rms + 1e-8) * TARGET_RMS
    return np.clip(audio, -1.0, 1.0)


def _list_input_devices():
    if not SD_OK or sd is None:
        return []
    out = []
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get('max_input_channels', 0) > 0:
                out.append((idx, dev.get('name', f'Device {idx}')))
    except Exception:
        pass
    return out


def _find_default_input():
    if not SD_OK or sd is None:
        return None
    try:
        info    = sd.query_devices(kind='input')
        devices = sd.query_devices()
        if info is not None:
            for idx, dev in enumerate(devices):
                if dev.get('name') == info.get('name'):
                    return idx
        default = sd.default.device
        return default[0] if isinstance(default, (list, tuple)) else default
    except Exception:
        devs = _list_input_devices()
        return devs[0][0] if devs else None


# ── INFERENCE ─────────────────────────────────────────────────────────────────
class InferenceEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lfcc   = LFCCExtractor().to(self.device)
        self.model  = None
        self.error  = None

    def load_model(self):
        if not os.path.exists(CHECKPOINT_PATH):
            self.error = f'Checkpoint not found: {CHECKPOINT_PATH}'
            return
        try:
            model = TransformerSpoofDetector(
                input_dim=60, d_model=128, nhead=4,
                num_layers=2, dim_feedforward=256, dropout=0.3,
            )
            ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(self.device).eval()
            self.model = model
        except Exception as e:
            self.error = str(e)

    def preprocess(self, waveform_np):
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[-1] > MAX_LENGTH:
            waveform = waveform[..., :MAX_LENGTH]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, MAX_LENGTH - waveform.shape[-1]))
        waveform = waveform.unsqueeze(0).to(self.device)
        lfcc     = self.lfcc(waveform)
        return lfcc.transpose(1, 2)

    @torch.no_grad()
    def predict(self, waveform_np):
        if self.model is None:
            raise RuntimeError('Model not loaded')
        x = self.preprocess(waveform_np)
        return torch.softmax(self.model(x), dim=1)[0, 1].item()


# ── APP ───────────────────────────────────────────────────────────────────────
class SpoofDetectorApp:
    def __init__(self, root):
        self.root        = root
        self.engine      = InferenceEngine()
        self.q           = queue.Queue()
        self.recording   = False
        self.anim_id     = None
        self.bar_anim_id = None
        self.device_map  = {}
        self._setup_window()
        self._build_ui()
        self._load_model_async()
        self._poll_queue()

    def _setup_window(self):
        self.root.title('Voice Spoof Detector')
        self.root.configure(bg=BG)
        self.root.geometry('960x760')
        self.root.minsize(860, 700)
        self.root.resizable(True, True)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f'+{sw//2-480}+{sh//2-380}')

    def _build_ui(self):
        self.f_heading = tkfont.Font(family='Segoe UI', size=16, weight='bold')
        self.f_sub     = tkfont.Font(family='Segoe UI', size=10)
        self.f_btn     = tkfont.Font(family='Segoe UI', size=10, weight='bold')
        self.f_label   = tkfont.Font(family='Segoe UI', size=9)
        self.f_mono    = tkfont.Font(family='Consolas', size=9)
        self.f_verdict = tkfont.Font(family='Segoe UI', size=28, weight='bold')
        self.f_score   = tkfont.Font(family='Consolas', size=11, weight='bold')

        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill='both', expand=True, padx=36, pady=24)

        # Header
        hdr = tk.Frame(outer, bg=BG)
        hdr.pack(fill='x', pady=(0, 16))
        tk.Label(hdr, text='Voice Spoof Detector',
                 font=self.f_heading, bg=BG, fg=TEXT).pack(side='left')
        status_f = tk.Frame(hdr, bg=BG)
        status_f.pack(side='right')
        self.status_dot = tk.Canvas(status_f, width=10, height=10, bg=BG, highlightthickness=0)
        self.status_dot.pack(side='left', padx=(0, 6))
        self._dot(YELLOW)
        self.status_lbl = tk.Label(status_f, text='Loading model…',
                                   font=self.f_mono, bg=BG, fg=TEXT_DIM)
        self.status_lbl.pack(side='left')
        tk.Frame(outer, bg=BORDER, height=1).pack(fill='x', pady=(0, 16))

        # Waveform
        wave_frame = tk.Frame(outer, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        wave_frame.pack(fill='x', pady=(0, 14))
        wh = tk.Frame(wave_frame, bg=BG2)
        wh.pack(fill='x', padx=14, pady=(10, 0))
        tk.Label(wh, text='WAVEFORM', font=self.f_label, bg=BG2, fg=TEXT_DIM).pack(side='left')
        self.dur_lbl = tk.Label(wh, text='', font=self.f_mono, bg=BG2, fg=TEXT_DIM)
        self.dur_lbl.pack(side='right')
        self.canvas = tk.Canvas(wave_frame, height=110, bg=BG2, highlightthickness=0)
        self.canvas.pack(fill='x', padx=14, pady=(6, 14))
        self._flat_wave()

        # Controls
        ctrl = tk.Frame(outer, bg=BG)
        ctrl.pack(fill='x', pady=(0, 14))
        left_ctrl = tk.Frame(ctrl, bg=BG)
        left_ctrl.pack(side='left')
        self.rec_btn = tk.Button(
            left_ctrl, text='RECORD', font=self.f_btn,
            bg=ACCENT, fg='#ffffff', activebackground=ACCENT2, activeforeground='#ffffff',
            relief='flat', padx=28, pady=10, cursor='hand2', bd=0,
            command=self._on_record)
        self.rec_btn.pack(side='left', padx=(0, 10))
        self.load_btn = tk.Button(
            left_ctrl, text='LOAD FILE', font=self.f_btn,
            bg=BG3, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief='flat', padx=28, pady=10, cursor='hand2', bd=0,
            command=self._on_load)
        self.load_btn.pack(side='left', padx=(0, 10))
        self.rec_status = tk.Label(left_ctrl, text='', font=self.f_btn, bg=BG, fg=YELLOW)
        self.rec_status.pack(side='left', padx=(12, 0))

        right_ctrl = tk.Frame(ctrl, bg=BG)
        right_ctrl.pack(side='right')
        tk.Label(right_ctrl, text='Input:', font=self.f_label,
                 bg=BG, fg=TEXT_DIM).pack(side='left', padx=(0, 4))
        self.device_var  = tk.StringVar(value='(detecting)')
        self.device_menu = tk.OptionMenu(right_ctrl, self.device_var, '(detecting)')
        self.device_menu.config(
            font=self.f_label, bg=BG3, fg=TEXT,
            activebackground=BORDER, activeforeground=TEXT,
            highlightthickness=0, relief='flat', padx=6)
        self.device_menu['menu'].config(
            bg=BG3, fg=TEXT, activebackground=ACCENT, activeforeground='#fff')
        self.device_menu.pack(side='left')
        self._populate_devices()

        # Score bar
        res_frame = tk.Frame(outer, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        res_frame.pack(fill='x', pady=(0, 14))
        rh = tk.Frame(res_frame, bg=BG2)
        rh.pack(fill='x', padx=14, pady=(10, 4))
        tk.Label(rh, text='ANALYSIS', font=self.f_label, bg=BG2, fg=TEXT_DIM).pack(side='left')
        bar_row = tk.Frame(res_frame, bg=BG2, pady=6)
        bar_row.pack(fill='x', padx=14)
        tk.Label(bar_row, text='Spoof probability', font=self.f_sub,
                 bg=BG2, fg=TEXT, width=18, anchor='w').pack(side='left')
        bar_outer = tk.Frame(bar_row, bg=BG3, height=22)
        bar_outer.pack(side='left', fill='x', expand=True, padx=(8, 8))
        bar_outer.pack_propagate(False)
        self.bar_fill  = tk.Frame(bar_outer, bg=TEXT_DIM)
        self.bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0)
        self.bar_outer = bar_outer
        self.score_lbl = tk.Label(bar_row, text='--', font=self.f_score,
                                  bg=BG2, fg=TEXT_DIM, width=7, anchor='e')
        self.score_lbl.pack(side='left')
        self.tag_lbl   = tk.Label(bar_row, text='', font=self.f_btn,
                                  bg=BG2, fg=TEXT_DIM, width=12, anchor='w')
        self.tag_lbl.pack(side='left', padx=(10, 0))
        tk.Frame(res_frame, bg=BG2, height=4).pack()

        # Threshold indicator
        thr_row = tk.Frame(res_frame, bg=BG2)
        thr_row.pack(fill='x', padx=14, pady=(0, 10))
        tk.Label(thr_row,
                 text=f'Threshold: {SPOOF_THRESHOLD:.2f}  (score > threshold = SPOOF)',
                 font=self.f_mono, bg=BG2, fg=TEXT_DIM).pack(side='left')

        # Verdict
        vf = tk.Frame(outer, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        vf.pack(fill='x')
        vi = tk.Frame(vf, bg=BG2, pady=22)
        vi.pack()
        tk.Label(vi, text='VERDICT', font=self.f_label, bg=BG2, fg=TEXT_DIM).pack()
        self.verdict_lbl = tk.Label(vi, text='AWAITING INPUT',
                                    font=self.f_verdict, bg=BG2, fg=TEXT_DIM)
        self.verdict_lbl.pack(pady=(6, 0))
        self.verdict_sub = tk.Label(vi, text='Record or load an audio sample',
                                    font=self.f_sub, bg=BG2, fg=TEXT_DIM)
        self.verdict_sub.pack(pady=(2, 0))

        # Footer
        ft = tk.Frame(self.root, bg=BG, pady=10)
        ft.pack(side='bottom', fill='x')
        tk.Label(ft,
                 text='ASVspoof2019 LA  ·  LFCC  ·  Transformer (86.47% acc)  ·  Tahiri S. & Akhchine I.',
                 font=self.f_mono, bg=BG, fg=TEXT_DIM).pack()

    # ── HELPERS ───────────────────────────────────────────────────────────────
    def _dot(self, color):
        self.status_dot.delete('all')
        self.status_dot.create_oval(1, 1, 9, 9, fill=color, outline=color)

    def _set_status(self, txt, color=TEXT_DIM):
        self.status_lbl.config(text=txt, fg=color)
        self._dot(color)

    def _set_controls(self, on):
        state = 'normal' if on else 'disabled'
        self.rec_btn.config(state=state)
        self.load_btn.config(state=state)
        self.device_menu.config(state=state)

    def _flat_wave(self):
        self.canvas.delete('all')
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width() or 880
        h = self.canvas.winfo_height() or 110
        self.canvas.create_line(0, h//2, w, h//2, fill=BORDER, width=1)

    def _draw_wave(self, data, color=ACCENT):
        self.canvas.delete('all')
        self.canvas.update_idletasks()
        w, h = self.canvas.winfo_width() or 880, self.canvas.winfo_height() or 110
        mid  = h // 2
        if len(data) == 0:
            return
        peak = np.max(np.abs(data))
        norm = data / peak if peak > 0 else data
        step = max(1, len(norm) // w)
        pts_top, pts_bot = [], []
        for i in range(w):
            idx = i * step
            if idx >= len(norm): break
            chunk = norm[idx:idx+step]
            pts_top.extend([i, mid - float(np.max(chunk)) * mid * 0.85])
            pts_bot.extend([i, mid - float(np.min(chunk)) * mid * 0.85])
        if len(pts_top) >= 4:
            self.canvas.create_line(pts_top, fill=color, width=1, smooth=True)
        if len(pts_bot) >= 4:
            self.canvas.create_line(pts_bot, fill=color, width=1, smooth=True)
        self.canvas.create_line(0, mid, w, mid, fill=BORDER, width=1, dash=(4, 4))
        dur = len(data) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(data**2)))
        self.dur_lbl.config(text=f'{dur:.2f}s  peak={peak:.3f}  rms={rms:.4f}')

    def _anim_rec(self, frame=0):
        self.canvas.delete('all')
        self.canvas.update_idletasks()
        w, h = self.canvas.winfo_width() or 880, self.canvas.winfo_height() or 110
        mid  = h // 2
        pts  = []
        for i in range(w):
            phase = (i/w)*4*np.pi + frame*0.25
            amp   = (12 + 8*np.sin(frame*0.08)) * (0.5*np.sin(phase) + 0.3*np.sin(phase*2.3))
            pts.extend([i, mid - amp])
        if len(pts) >= 4:
            self.canvas.create_line(pts, fill=RED, width=2, smooth=True)
        self.canvas.create_line(0, mid, w, mid, fill=BORDER, width=1, dash=(4, 4))
        self.anim_id = self.root.after(40, self._anim_rec, frame+1)

    def _stop_anim(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    # ── DEVICE SELECTION — FIXED INDENTATION ──────────────────────────────────
    def _populate_devices(self):
        devs = _list_input_devices()
        menu = self.device_menu['menu']
        menu.delete(0, 'end')
        self.device_map = {}
        if not devs:
            self.device_var.set('No devices')
            return
        default_idx      = _find_default_input()
        chosen           = None
        microphone_array = None
        for idx, name in devs:
            label = f'{idx}: {name}'
            self.device_map[label] = idx
            menu.add_command(label=label, command=lambda v=label: self.device_var.set(v))
            name_lower = name.lower()
            if any(k in name_lower for k in ('array', 'stereo mix')):
                microphone_array = label
            if default_idx is not None and idx == default_idx:
                chosen = label
        if chosen is None and microphone_array is not None:
            chosen = microphone_array
        if chosen is None:
            for label in self.device_map:
                if any(k in label.lower() for k in ('microphone', 'mic', 'input')):
                    chosen = label
                    break
        if chosen is None:
            chosen = next(iter(self.device_map))
        self.device_var.set(chosen)

    def _get_device(self):
        return self.device_map.get(self.device_var.get())

    # ── MODEL LOAD ────────────────────────────────────────────────────────────
    def _load_model_async(self):
        def _bg():
            self.engine.load_model()
            self.q.put(('model_loaded', None))
        threading.Thread(target=_bg, daemon=True).start()

    # ── RECORD ────────────────────────────────────────────────────────────────
    def _on_record(self):
        if not SD_OK:
            self._set_status('sounddevice missing', RED); return
        self._set_controls(False)
        self._reset_results()
        threading.Thread(target=self._record_worker, daemon=True).start()

    def _record_worker(self):
        try:
            self.q.put(('rec_start', None))
            dev = self._get_device() or _find_default_input()
            if dev is None:
                self.q.put(('error', 'No input device available')); return
            total = int(RECORD_SECS * SAMPLE_RATE)
            sd.default.samplerate = SAMPLE_RATE
            sd.default.channels   = 1
            sd.default.dtype      = 'float32'
            audio = sd.rec(total, samplerate=SAMPLE_RATE, channels=1,
                           dtype='float32', device=dev, blocking=False)
            for remaining in range(RECORD_SECS, 0, -1):
                self.q.put(('tick', remaining))
                time.sleep(1)
            sd.wait()
            data    = audio.flatten().astype(np.float32).copy()
            rms_raw = float(np.sqrt(np.mean(data**2)))
            if rms_raw < MIN_RMS_THRESHOLD:
                self.q.put(('warn_low_signal', rms_raw))
                self._set_controls(True)
                return
            data = _process_audio_for_inference(data)
            self.q.put(('rec_done', data))
            self.q.put(('infer', data))
        except Exception as e:
            self.q.put(('error', f'Recording failed: {e}'))

    # ── LOAD FILE ─────────────────────────────────────────────────────────────
    def _on_load(self):
        path = filedialog.askopenfilename(
            title='Select Audio',
            filetypes=[
                ('Audio', '*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.wma *.aiff *.opus'),
                ('All', '*.*')])
        if not path: return
        self._set_controls(False)
        self._reset_results()
        threading.Thread(target=self._load_worker, args=(path,), daemon=True).start()

    def _load_worker(self, path):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f'File not found: {path}')
            data, sr = None, None
            if SF_OK and sf is not None:
                try: data, sr = sf.read(path, dtype='float32')
                except Exception: data = None
            if data is None:
                try:
                    from pydub import AudioSegment
                    seg     = AudioSegment.from_file(path)
                    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                    if seg.channels > 1:
                        samples = samples.reshape((-1, seg.channels)).mean(axis=1)
                    data, sr = samples / (2**15), seg.frame_rate
                except Exception: pass
            if data is None:
                raise RuntimeError('Cannot decode file. Try WAV or FLAC format.')
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            if sr != SAMPLE_RATE:
                try:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g    = gcd(sr, SAMPLE_RATE)
                    data = resample_poly(data, SAMPLE_RATE//g, sr//g).astype(np.float32)
                except ImportError:
                    idx  = np.linspace(0, len(data)-1, int(len(data)*SAMPLE_RATE/sr))
                    data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
            data = _process_audio_for_inference(data)
            self.q.put(('file_ok', data))
            self.q.put(('infer', data))
        except Exception as e:
            self.q.put(('error', f'Load error: {e}'))

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    def _infer_worker(self, data):
        try:
            score = self.engine.predict(data)
            self.q.put(('result', score))
        except Exception as e:
            self.q.put(('error', str(e)))

    # ── RESULTS ───────────────────────────────────────────────────────────────
    def _reset_results(self):
        if self.bar_anim_id:
            self.root.after_cancel(self.bar_anim_id)
            self.bar_anim_id = None
        self.bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0)
        self.bar_fill.config(bg=TEXT_DIM)
        self.score_lbl.config(text='--', fg=TEXT_DIM)
        self.tag_lbl.config(text='', fg=TEXT_DIM)
        self.verdict_lbl.config(text='ANALYZING…', fg=YELLOW)
        self.verdict_sub.config(text='Processing audio…', fg=TEXT_DIM)
        self.rec_status.config(text='')

    def _anim_bar(self, target, color, frame=0, total=30):
        if self.bar_anim_id:
            self.root.after_cancel(self.bar_anim_id)
        if frame >= total:
            self.bar_fill.place(x=0, y=0, relheight=1.0, relwidth=target)
            self.bar_fill.config(bg=color)
            self.bar_anim_id = None
            return
        ease = 1 - (1 - frame/total)**2
        self.bar_fill.place(x=0, y=0, relheight=1.0, relwidth=max(0, target*ease))
        self.bar_fill.config(bg=color)
        self.bar_anim_id = self.root.after(16, self._anim_bar, target, color, frame+1, total)

    def _show_result(self, spoof_prob):
        self._stop_anim()
        is_spoof = spoof_prob > SPOOF_THRESHOLD
        color    = RED if is_spoof else GREEN
        self._anim_bar(spoof_prob, color)
        self.score_lbl.config(text=f'{spoof_prob*100:.1f}%', fg=color)
        self.tag_lbl.config(text='SPOOF' if is_spoof else 'AUTHENTIC', fg=color)
        if is_spoof:
            self.verdict_lbl.config(text='SPOOF DETECTED', fg=RED)
            self.verdict_sub.config(text='Audio appears synthetic or manipulated', fg=RED_DIM)
        else:
            self.verdict_lbl.config(text='AUTHENTIC', fg=GREEN)
            self.verdict_sub.config(text='Audio appears to be genuine human speech', fg=GREEN_DIM)
        self._set_controls(True)
        self._set_status('Done', GREEN)

    # ── QUEUE ─────────────────────────────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                msg, payload = self.q.get_nowait()
                if msg == 'model_loaded':
                    if self.engine.model:
                        self._set_status('Ready', GREEN)
                        self._set_controls(True)
                    else:
                        self._set_status(f'Error: {self.engine.error}', RED)
                        self._set_controls(False)
                        self.verdict_lbl.config(text='MODEL ERROR', fg=RED)
                        self.verdict_sub.config(text=str(self.engine.error)[:90], fg=RED_DIM)
                elif msg == 'rec_start':
                    self._set_status('Recording…', RED)
                    self._anim_rec()
                elif msg == 'tick':
                    self.rec_status.config(text=f'{payload}s')
                elif msg == 'warn_low_signal':
                    self._set_status(f'Too quiet (rms={payload:.4f}) — speak louder', YELLOW)
                    self._set_controls(True)
                    self._stop_anim()
                    self.verdict_lbl.config(text='LOW SIGNAL', fg=YELLOW)
                    self.verdict_sub.config(text='Microphone level too low — try again', fg=TEXT_DIM)
                elif msg == 'rec_done':
                    self._stop_anim()
                    self.rec_status.config(text='')
                    self._draw_wave(payload, ACCENT)
                    self._set_status('Analyzing…', YELLOW)
                elif msg == 'file_ok':
                    self._draw_wave(payload, ACCENT2)
                    self._set_status('Analyzing…', YELLOW)
                elif msg == 'infer':
                    threading.Thread(target=self._infer_worker,
                                     args=(payload,), daemon=True).start()
                elif msg == 'result':
                    self._show_result(payload)
                elif msg == 'error':
                    self._set_status(f'Error: {payload}', RED)
                    self.verdict_lbl.config(text='ERROR', fg=RED)
                    self.verdict_sub.config(text=str(payload)[:90], fg=RED_DIM)
                    self._set_controls(True)
                    self._stop_anim()
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)


if __name__ == '__main__':
    if not TORCH_OK:
        print('PyTorch required: pip install torch torchaudio')
        exit(1)
    root = tk.Tk()
    SpoofDetectorApp(root)
    root.mainloop()