import os
import sys
#sys.path.append(os.path.dirname(__file__))
#sys.path.append('..')
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import math
import numpy as np
import time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if __package__ == '':
    from stts import audio, audio_util, util, textutil
else:
    from .stts import audio, audio_util, util, textutil



class Mel2TextDataset(Dataset):
    def __init__(self, meta_path, use_mel=True, use_mp=True, use_tp=False, use_m2c=False, 
                 stride=1, text_upsample=1, add_sos=False, add_eos=False, in_memory=False):
        self.use_mel = use_mel
        self.use_mp = use_mp
        self.use_tp = use_tp
        self.use_m2c = use_m2c        
        self.stride = stride
        self.text_upsample = text_upsample
        
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.pad = textutil._char_vocab[0]
        self.sos = textutil._char_vocab[1]
        self.eos = textutil._char_vocab[2]
        
        self.in_memory = in_memory
        self.meta_dir = os.path.dirname(meta_path)
        self.meta_path = meta_path
                
        self._script = []
        self._text = []
        self._mel = []
        self._mp = []
        self._tp = []
        self._m2c = []
        self._mel_path = []
        self._mp_path = []
        self._tp_path = []
        self._m2c_path = []
        self._n_frame = []
        
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        self.len = len(lines)
        self.meta = []
        for i, t in enumerate(lines):
            fname, n_frame, mel_path, mp_path, tp_path, m2c_path, ntext= t.split('|')
            if ntext.endswith('\n'):
                ntext = ntext[:-1]
            self.meta.append((fname, ntext, n_frame, mel_path, mp_path, tp_path, m2c_path))
            _mel_path = os.path.join(self.meta_dir, mel_path)
            _mp_path = os.path.join(self.meta_dir, mp_path)
            _tp_path = os.path.join(self.meta_dir, tp_path)          
            _m2c_path = os.path.join(self.meta_dir, m2c_path)
            self._mel_path.append(_mel_path)
            self._mp_path.append(_mp_path)
            self._tp_path.append(_tp_path)
            self._m2c_path.append(_m2c_path)
            self._n_frame.append(n_frame)
            
            _script = ntext
            _text = textutil.char2idx(_script)
            if self.add_sos:
                _script = self.sos + _script
                _text = [1] + _text 
            if self.add_eos:
                _script = self.eos + _script
                _text = [2] + _text 
            self._script.append(_script)
            self._text.append(_text)
            if in_memory:
                if use_mel:
                    _mel = np.load(_mel_path)[...,::self.stride]
                    self._mel.append(_mel)
                if use_mp:
                    _mp = np.load(_mp_path)
                    self._mp.append(_mp)
                if use_tp:
                    _tp = np.load(_tp_path)
                    self._tp.append(_tp)
                if use_m2c:
                    _m2c = np.load(_m2c_path)
                    self._m2c.append(_m2c)
                                   
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):  
        _text = self._text[idx]
                
        _text = np.array(_text, dtype=np.int64)
        _n_frame = 0
        sample = {'idx':idx, 'text':_text, 'n_text':len(_text)}        
       
        if self.use_mel:
            _mel = self._mel[idx] if self.in_memory else np.load(self._mel_path[idx])[...,::self.stride]
            _n_frame = _mel.shape[-1]
            sample['mel'] = _mel
        if self.use_mp:
            _mp = self._mp[idx] if self.in_memory else np.load(self._mp_path[idx])
            _n_frame = _mp.shape[-1]
            sample['mp'] = _mp
            assert _mp.shape[0] == len(_text) * self.text_upsample
        if self.use_tp:
            _tp = self._tp[idx] if self.in_memory else np.load(self._tp_path[idx])
            _n_frame = _tp.shape[-1]
            sample['tp'] = _tp
            assert _tp.shape[0] == len(_text) * self.text_upsample
        if self.use_m2c:
            _m2c = self._m2c[idx] if self.in_memory else np.load(self._m2c_path[idx])
            _n_frame = _m2c.shape[-1]
            sample['m2c'] = _m2c
        sample['n_frame'] = _n_frame
        return sample
    
    def collate(self, samples):
        idxes = []
        mels = []
        mps = []
        tps = []
        m2cs = []
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
            if self.use_mel:
                mels.append(np.pad(s['mel'], ((0, 0), (0, _mpad)), constant_values=0.0)) 
            if self.use_mp:
                mps.append(np.pad(s['mp'], ((0, self.text_upsample * _tpad), (0, _mpad)), constant_values=0.0))
            if self.use_tp:
                tps.append(np.pad(s['tp'], ((0, self.text_upsample * _tpad), (0, _mpad)), constant_values=0.0))
            if self.use_m2c:
                m2cs.append(np.pad(s['m2c'], ((0, 0), (0, _mpad)), constant_values=0.0))
                
        batch = {'idx':torch.tensor(idxes, dtype=torch.int64), 
                 'text':torch.tensor(texts, dtype=torch.int64), 
                 'n_text':torch.tensor(n_texts, dtype=torch.int32),
                 'n_frame':torch.tensor(n_frames, dtype=torch.int32)}
        if self.use_mel:
            batch['mel'] = torch.tensor(mels, dtype=torch.float32)
        if self.use_mp:
            batch['mp'] = torch.tensor(mps, dtype=torch.float32)
        if self.use_tp:
            batch['tp'] = torch.tensor(tps, dtype=torch.float32)
        if self.use_m2c:
            batch['m2c'] = torch.tensor(m2cs, dtype=torch.float32)
        return batch
    
    def get_length_sampler(self, batch_size, noise=10.0, shuffle=True):
        sampler = LengthSampler(self._n_frame, batch_size, noise, shuffle)
        return sampler
        
        
class LengthSampler(Sampler):
    def __init__(self, lengths, batch_size, noise=10.0, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.noise = noise
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.lengths)
    
    def __iter__(self):
        _lengths = torch.tensor(self.lengths, dtype=torch.float32)
        _lengths = _lengths + 2 * self.noise * torch.rand_like(_lengths)
        _sorted, _idx = torch.sort(_lengths)
        
        if self.shuffle:
            _len = len(_lengths)
            _num_full_batches = _len // self.batch_size
            _full_batch_size = _num_full_batches * self.batch_size
            _full_batch_idx = _idx[:_full_batch_size]
            _remnant_idx = _idx[_full_batch_size:]
            _rand_batch_idx = torch.randperm(_num_full_batches)
            _rand_full_idx = _full_batch_idx.reshape(_num_full_batches, self.batch_size)[_rand_batch_idx].reshape(-1) 
            _rand_idx = torch.cat([_rand_full_idx, _remnant_idx], dim=0)
            return iter(_rand_idx)
        else:
            return iter(_idx)
            
            