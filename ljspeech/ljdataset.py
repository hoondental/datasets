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
    def __init__(self, meta_path, _spec=True, _mel=True, _phone=False, stride=1, add_sos=False, add_eos=False):
        self._spec = _spec
        self._mel = _mel
        self._phone = _phone
        self.stride = stride
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.meta_dir = os.path.dirname(meta_path)
        
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        self.len = len(lines)
        self.meta = []
        for i, t in enumerate(lines):
            fname, text, n_frame, spec_path, mel_path = t.split('|')
            n_frame = int(n_frame)
            if mel_path.endswith('\n'):
                mel_path = mel_path[:-1]
            self.meta.append((fname, text, n_frame, os.path.join(self.meta_dir, spec_path), os.path.join(self.meta_dir, mel_path)))
            
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        meta = self.meta[idx]
        spec_path = meta[3]
        mel_path = meta[4]
        _text = meta[1]
        n_frame = meta[2]
        _text = textutil.text_normalize(_text)
        text = textutil.char2idx(_text)
        if self._phone:
            _phone = textutil.text2phone(_text)
            phone1 = textutil.phone2idx(_phone)
            phone2 = textutil.phone2idx2(_phone)
        if self.add_sos:
            text = [1] + text 
            if self._phone:
                phone1 = [1] + phone1
                phone2 = [(1, 0)] + phone2
        if self.add_eos:
            text = text + [2]
            if self._phone:
                phone1 = phone1 + [2]
                phone2 = phone2 + [(2, 0)]
        
        text = np.array(text, dtype=np.int64)
        sample = {'idx':idx, 'text':text, 'n_text':len(text), 'n_frame':math.ceil(n_frame/self.stride)}
        if self._spec:
            sample['spec'] = np.load(spec_path)[...,::self.stride]
        if self._mel:
            sample['mel'] = np.load(mel_path)[...,::self.stride]        
        if self._phone:
            sample['phone1'] = np.array(phone1, dtype=np.int64)
            sample['phone2'] = np.array(phone2, dtype=np.int64).T    
            sample['n_phone1'] = len(phone1)
            sample['n_phone2'] = len(phone2)
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
        
        
        
            


                








    
    
