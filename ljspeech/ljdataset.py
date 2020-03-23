import os
import sys
#sys.path.append('..')
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import math
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
import librosa
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from shutil import copyfile

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from stts import audio, audio_util, util, textutil



class LJDataset(Dataset):
    def __init__(self, meta_path, _spec=True, _mel=True, stride=1, add_sos=False, add_eos=False, use_phone=False, use_stress=False):
        self._spec = _spec
        self._mel = _mel
        self.stride = stride
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.use_phone = use_phone
        self.use_stress = use_stress
        self.meta_dir = os.path.dirname(meta_path)
        
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        self.len = len(lines)
        self.meta = []
        for i, t in enumerate(lines):
            fname, ntext, n_frame, spec_path, mel_path = t.split('|')
            n_frame = int(n_frame)
            if mel_path.endswith('\n'):
                mel_path = mel_path[:-1]
            self.meta.append((fname, ntext, n_frame, os.path.join(self.meta_dir, spec_path), os.path.join(self.meta_dir, mel_path)))
            
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        meta = self.meta[idx]
        spec_path = meta[3]
        mel_path = meta[4]
        text = meta[1]
        n_frame = meta[2]
        text = textutil.text_normalize(text)
        if self.use_phone:
            if self.use_stress:
                pass
            else:
                pass
        else:
            text = np.array(textutil.char2idx(text), dtype=np.int64)
        sample = {'idx':idx, 'text':text, 'n_text':len(text), 'n_frame':math.ceil(n_frame/self.stride)}
        if self._spec:
            sample['spec'] = np.load(spec_path)[...,::self.stride]
        if self._mel:
            sample['mel'] = np.load(mel_path)[...,::self.stride]        
        return sample
            
    def collate(self, samples):
        text_lengths = []
        n_frames = []
        idxes = []
        texts = []
        specs = []
        mels = []
        for i, s in enumerate(samples):
            text_lengths.append(s['n_text'])
            n_frames.append(s['n_frame'])
            idxes.append(s['idx'])
        max_text_len = max(text_lengths)
        max_n_frame = max(n_frames)
        
        for i, s in enumerate(samples):
            texts.append(np.pad(s['text'], (0, max_text_len - text_lengths[i]), constant_values=0))
            if self._spec:
                specs.append(np.pad(s['spec'], ((0, 0), (0, max_n_frame - n_frames[i])), constant_values=0.0))
            if self._mel:
                mels.append(np.pad(s['mel'], ((0, 0), (0, max_n_frame - n_frames[i])), constant_values=0.0))    
                
        batch = {'idx':torch.tensor(idxes, dtype=torch.int64), 
                 'text':torch.tensor(texts, dtype=torch.int64), 
                 'n_text':torch.tensor(text_lengths, dtype=torch.int32),
                 'n_frame':torch.tensor(n_frames, dtype=torch.int32)}
        if self._spec:
            batch['spec'] = torch.tensor(specs, dtype=torch.float32)
        if self._mel:
            batch['mel'] = torch.tensor(mels, dtype=torch.float32)
        if self._guide:
            batch['guide'] = torch.tensor(attention_guide(text_lengths, n_frames, width=20), dtype=torch.float32)
        if self._mask:
            batch['mask'] = torch.tensor(attention_mask(text_lengths, n_frames), dtype=torch.float32)
        return batch
        
        
        
            


                








    
    
