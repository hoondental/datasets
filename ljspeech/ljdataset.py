import os
import sys
sys.path.append(os.path.dirname(__file__))
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
    def __init__(self, meta_path, use_spec=True, use_mel=True, use_phone=False, stride=1, add_sos=False, add_eos=False, in_memory=False):
        self.use_spec = use_spec
        self.use_mel = use_mel
        self.use_phone = use_phone
        self.stride = stride
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.in_memory = in_memory
        self.meta_dir = os.path.dirname(meta_path)
        
        self._script = []
        self._text = []
        self._phone1 = []
        self._phone2 = []
        self._mel = []
        self._spec = []
        self._n_text = []
        self._n_phone = []
        self._n_frame = []
        
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        self.len = len(lines)
        self.meta = []
        for i, t in enumerate(lines):
            fname, text, spec_path, mel_path = t.split('|')
            if mel_path.endswith('\n'):
                mel_path = mel_path[:-1]
            _spec_path = os.path.join(self.meta_dir, spec_path)
            _mel_path = os.path.join(self.meta_dir, mel_path)
            self.meta.append((fname, text, _spec_path, _mel_path))
            
            _script = list(textutil.text_normalize(text))
            _text = textutil.char2idx(_script)
            if use_phone:
                _phone = textutil.text2phone(_script)
                _phone1 = textutil.phone2idx(_phone)
                _phone2 = textutil.phone2idx2(_phone)
            if self.add_sos:
                _script = [textutil._char_vocab[1]] + _script
                _text = [1] + _text 
                if use_phone:
                    _phone = [textutil._char_vocab[1]] + _phone
                    _phone1 = [1] + _phone1
                    _phone2 = [(1, 0)] + _phone2
            if self.add_eos:
                _script = [textutil._char_vocab[2]] + _script
                _text = [2] + _text 
                if use_phone:
                    _phone = [textutil._char_vocab[2]] + _phone
                    _phone1 = [2] + _phone1
                    _phone2 = [(2, 0)] + _phone2
            self._script.append(_script)
            self._text.append(_text)
            self._n_text.append(len(_text))
            if use_phone:
                self._phone.append(_phone)
                self._phone1.append(_phone1)
                self._phone2.append(_phone2)
                self._n_phone.append(len(_phone))
            if in_memory:
                _n_frame = 0
                if use_spec:
                    _spec = np.load(_spec_path)[...,::self.stride]
                    self._spec.append(_spec) 
                    _n_frame = _spec.shape[-1]
                if use_mel:
                    _mel = np.load(_mel_path)[...,::self.stride]
                    self._mel.append(_mel) 
                    _n_frame = _mel.shape[-1]
                self._n_frame.append(_n_frame)           
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        meta = self.meta[idx]
        _spec_path = meta[2]
        _mel_path = meta[3]
        _text = self._text[idx]

        text = np.array(_text, dtype=np.int64)
        sample = {'idx':idx, 'text':text, 'n_text':self._n_text[idx], 'n_frame':self._n_frame[idx]}
        if self.use_spec:
            sample['spec'] = self._spec[idx] if self.in_memory else np.load(spec_path)[...,::self.stride]
        if self.use_mel:
            sample['mel'] = self._mel[idx] if self.in_memory else np.load(mel_path)[...,::self.stride]        
        if self.use_phone:
            sample['phone1'] = np.array(self._phone1[idx], dtype=np.int64)
            sample['phone2'] = np.array(self._phone2[idx], dtype=np.int64).T    
            sample['n_phone1'] = self._n_phone[idx]
            sample['n_phone2'] = self._n_phone[idx]
        return sample
            
    def collate(self, samples):
        text_lengths = []
        n_phones1 = []
        n_phones2 = []
        n_frames = []
        idxes = []
        texts = []
        specs = []
        mels = []
        phones1 = []
        phones2 = []
        for i, s in enumerate(samples):
            text_lengths.append(s['n_text'])
            n_frames.append(s['n_frame'])
            idxes.append(s['idx'])
            if self._phone:
                n_phones1.append(s['n_phone1'])
                n_phones2.append(s['n_phone2'])
        max_text_len = max(text_lengths)
        max_n_frame = max(n_frames)
        if self._phone:
            max_n_phone1 = max(n_phones1)
            max_n_phone2 = max(n_phones2)
        
        for i, s in enumerate(samples):
            texts.append(np.pad(s['text'], (0, max_text_len - text_lengths[i]), constant_values=0))
            if self._spec:
                specs.append(np.pad(s['spec'], ((0, 0), (0, max_n_frame - n_frames[i])), constant_values=0.0))
            if self._mel:
                mels.append(np.pad(s['mel'], ((0, 0), (0, max_n_frame - n_frames[i])), constant_values=0.0))    
            if self._phone:
                phones1.append(np.pad(s['phone1'], (0, max_n_phone1 - n_phones1[i]), constant_values=0))
                phones2.append(np.pad(s['phone2'],((0, 0), (0, max_n_phone2 - n_phones2[i])), constant_values=0))
                               
                
        batch = {'idx':torch.tensor(idxes, dtype=torch.int64), 
                 'text':torch.tensor(texts, dtype=torch.int64), 
                 'n_text':torch.tensor(text_lengths, dtype=torch.int32),
                 'n_frame':torch.tensor(n_frames, dtype=torch.int32)}
        if self._spec:
            batch['spec'] = torch.tensor(specs, dtype=torch.float32)
        if self._mel:
            batch['mel'] = torch.tensor(mels, dtype=torch.float32)
        if self._phone:
            batch['phone1'] = torch.tensor(phones1, dtype=torch.int64)
            batch['phone2'] = torch.tensor(phones2, dtype=torch.int64)
            batch['n_phone1'] = torch.tensor(n_phones1, dtype=torch.int32)
            batch['n_phone2'] = torch.tensor(n_phones2, dtype=torch.int32)
        return batch
        
        
        
            


                








    
    
