
import os
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset

class ASVspoofDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.root_dir = config['paths']['data_root']
        
        protocol_dir = os.path.join(self.root_dir, 'raw', 'ASVspoof2019', 'LA', 'LA', 'ASVspoof2019_LA_cm_protocols')
        
        protocol_files = {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'
        }
        
        protocol_path = os.path.join(protocol_dir, protocol_files[split])
        
        # Lecture avec Pandas
        self.metadata = pd.read_csv(protocol_path, sep=' ', header=None, names=['speaker', 'filename', 'system', 'null', 'label'])
        
        # Mode Turbo (10%)
        if config['data'].get('debug_mode', False):
            fraction = config['data'].get('subset_fraction', 0.1)
            self.metadata = self.metadata.sample(frac=fraction, random_state=42).reset_index(drop=True)
            
        self.audio_dir = os.path.join(self.root_dir, 'raw', 'ASVspoof2019', 'LA', 'LA', f'ASVspoof2019_LA_{split}', 'flac')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['filename'] + '.flac')
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Cut/Pad simple
            max_len = 64000
            if waveform.shape[1] > max_len:
                waveform = waveform[:, :max_len]
            else:
                pad = max_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
        except Exception as e:
            # En cas d'erreur fichier, on renvoie du silence
            print(f"Erreur fichier: {audio_path}")
            waveform = torch.zeros(1, 64000)
            
        label = 0 if row['label'] == 'bonafide' else 1
        return waveform, torch.tensor(label)
