import os
import sys
sys.path.append(os.path.dirname(__file__))
#sys.path.append('..')
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import math
import numpy as np
import time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from stts import audio, audio_util, util, textutil



class Mel2TextDataset(Dataset):
    def __init__(self, meta_path, _mel=True, _mp=True, _tp=False, _m2c=False, _mpwin=False, _m2cwin=False, 
                 stride=1, text_upsample=1, add_sos=False, add_eos=False):
        self._mel = _mel
        self._mp = _mp
        self._tp = _tp
        self._m2c = _m2c
        self._mpwin = _mpwin
        self._m2cwin = _m2cwin
        
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.pad = textutil._char_vocab[0]
        self.sos = textutil._char_vocab[1]
        self.eos = textutil._char_vocab[2]
        
        self.meta_path = meta_path
        self.stride = stride
        self.text_upsample = text_upsample
        self.meta_dir = os.path.dirname(meta_path)
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        self.len = len(lines)
        self.meta = []
        for i, t in enumerate(lines):
            fname, mel_path, mp_path, tp_path, m2c_path, mpwin_path, m2cwin_path, text= t.split('|')
            if text.endswith('\n'):
                text = text[:-1]
            self.meta.append((fname, text, mel_path, mp_path, tp_path, m2c_path, mpwin_path, m2cwin_path))
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        meta = self.meta[idx]
        fname, text, mel_path, mp_path, tp_path, m2c_path, mpwin_path, m2cwin_path = meta        
        mel_path = os.path.join(self.meta_dir, mel_path)
        mp_path = os.path.join(self.meta_dir, mp_path)
        tp_path = os.path.join(self.meta_dir, tp_path)          
        m2c_path = os.path.join(self.meta_dir, m2c_path)
        mpwin_path = os.path.join(self.meta_dir, mpwin_path)
        m2cwin_path = os.path.join(self.meta_dir, m2cwin_path)
        
        if self.add_sos:
            text = self.sos + text
        if self.add_eos:
            text = text + self.eos
        text = np.array(textutil.char2idx(text), dtype=np.int64)
        
        sample = {'idx':idx, 'text':text}
       
        if self._mel:
            mel = np.load(mel_path)[...,::self.stride]
            sample['mel'] = mel
        if self._mp:
            mp = np.load(mp_path)
            sample['mp'] = mp
            assert mp.shape[0] == len(text) * self.text_upsample
        if self._tp:
            tp = np.load(tp_path)
            sample['tp'] = tp
            assert tp.shape[0] == len(text) * self.text_upsample
        if self._m2c:
            m2c = np.load(m2c_path)
            sampe['m2c'] = m2c
        if self._mpwin:
            mp = np.load(mpwin_path)
            sample['mpwin'] = mp
        if self._m2cwin:
            m2c = np.load(m2cwin_path)
            sampe['m2cwin'] = m2cwin
        sample['n_text'] = len(text)
        sample['n_frame'] = mel.shape[-1]
        return sample
    
    def collate(self, samples):
        idxes = []
        mels = []
        mps = []
        tps = []
        m2cs = []
        mpwins = []
        m2cwins = []
        n_texts = []
        n_frames = []
        texts = []
        for i, s in enumerate(samples):
            n_texts.append(s['n_text'])
            n_frames.append(s['n_frame'])
            idxes.append(s['idx'])
        max_n_text = max(n_texts)
        max_n_frame = max(n_frames)
        
        for i, s in enumerate(samples):
            _tpad = max_n_text - n_texts[i]
            _mpad = max_n_frame - n_frames[i]
            texts.append(np.pad(s['text'], (0, _tpad), constant_values=0))
            if self._mel:
                mels.append(np.pad(s['mel'], ((0, 0), (0, _mpad)), constant_values=0.0)) 
            if self._mp:
                mps.append(np.pad(s['mp'], ((0, self.text_upsample * _tpad), (0, _mpad)), constant_values=0.0))
            if self._tp:
                tps.append(np.pad(s['tp'], ((0, self.text_upsample * _tpad), (0, _mpad)), constant_values=0.0))
            if self._m2c:
                m2cs.append(np.pad(s['m2c'], ((0, 0), (0, _mpad)), constant_values=0.0))
            if self._mpwin:
                mpwins.append(np.pad(s['mpwin'], ((0, self.text_upsample * _tpad), (0, _mpad)), constant_values=0.0))
            if self._tp:
                m2cwins.append(np.pad(s['m2cwin'], ((0, 0), (0, _mpad)), constant_values=0.0))
                
        batch = {'idx':torch.tensor(idxes, dtype=torch.int64), 
                 'text':torch.tensor(texts, dtype=torch.int64), 
                 'n_text':torch.tensor(n_texts, dtype=torch.int32),
                 'n_frame':torch.tensor(n_frames, dtype=torch.int32)}
        if self._mel:
            batch['mel'] = torch.tensor(mels, dtype=torch.float32)
        if self._mp:
            batch['mp'] = torch.tensor(mps, dtype=torch.float32)
        if self._tp:
            batch['tp'] = torch.tensor(tps, dtype=torch.float32)
        if self._m2c:
            batch['m2c'] = torch.tensor(m2cs, dtype=torch.float32)
        if self._mpwin:
            batch['mpwin'] = torch.tensor(mpwins, dtype=torch.float32)
        if self._m2cwin:
            batch['m2cwin'] = torch.tensor(m2cwins, dtype=torch.float32)
        return batch
            
            