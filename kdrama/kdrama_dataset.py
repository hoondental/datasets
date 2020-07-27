import os
import sys
#sys.path.append(os.path.dirname(__file__))
#sys.path.append('..')
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import math
import re
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
from torch.utils.data.sampler import Sampler
import random

import tensorflow as tf




if __package__ == '':
    from stts import audio, audio_util, util, textutil, kor_util
    from preprocess import read_meta, save_meta
else:
    from .stts import audio, audio_util, util, textutil, kor_util
    from .preprocess import read_meta, save_meta

def _process_text(i, text, add_sos=False, add_eos=False):
    _text = text
    text1 = text.replace('^', ' ')
    text2 = text1.replace('\t', ' ')
    text3 = text2.replace('2+1', '2 플러스 1')
    text3 = text3.replace('닭 키', '닥 키')
    text4 = re.sub("[ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅉㅃㅆㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅔㅒㅖㅢ]", " ", text3)
    text5 = re.sub("[木‘…@#$%&*♪`♡′♥♬♩_+]", " ", text4)
    text = text5
    try:
        ntext = kor_util.text_normalize(text)
        stext = kor_util.text2symbol(ntext, add_sos, add_eos)
        itext = kor_util.symbol2idx(stext)
    except Exception as ex:
        print('i:', i)
        print(ex)
        print(_text)
        print(text1)
        print(text2)
        print(text3)
        print(text4)
        print(text5)
        print(ntext)
        print(stext)
    return (i, text, ntext, stext, itext)

class KdramaDataset(Dataset):
    def __init__(self, meta_path, use_spec=True, use_mel=False, stride=1, add_sos=False, add_eos=False, 
                 include_numbers=True, tensor_type='torch'):
        self.use_spec = use_spec
        self.use_mel = use_mel
        self.stride = stride
        
        self.util = kor_util
        self.include_numbers = include_numbers
        self.symbols = kor_util._symbols
        
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.pad = self.symbols[0]
        self.sos = self.symbols[1]
        self.eos = self.symbols[2]
        
        self.tensor_type = tensor_type
        self.meta_dir = os.path.dirname(meta_path)
        self.meta_path = meta_path
        
        self.meta = read_meta(meta_path, spec_mel=True)
        self._text = []
        self._ntext = []
        self._stext = []
        self._itext = []
        self._mel_path = []
        self._spec_path = []
        self._n_frame = []
      
       
        executor = ProcessPoolExecutor(max_workers=10)
        jobs = []
        for i, _meta in enumerate(self.meta):
            _partial = partial(_process_text, i, _meta[-1], self.add_sos, self.add_eos)
            job = executor.submit(_partial)
            jobs.append(job)
        results = [job.result() for job in tqdm(jobs)]

        for i, text, ntext, stext, itext in results:
            fname, path, spec_path, mel_path, n_frame, script = self.meta[i]            
            self._spec_path.append(spec_path)
            self._mel_path.append(mel_path)
            self._n_frame.append(n_frame)
            self._text.append(text)
            self._ntext.append(ntext)
            self._stext.append(stext)
            self._itext.append(itext)

            
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        _spec_path = self._spec_path[idx]
        _mel_path = self._mel_path[idx]
        _text = self._itext[idx]
        _text = np.array(_text, dtype=np.int64)
        _n_frame = 0
        sample = {'idx':idx, 'text':_text, 'n_text':len(_text)}
        if self.use_spec:
            _spec = self._spec[idx] if self.in_memory else np.load(_spec_path)[...,::self.stride]
            _n_frame = _spec.shape[-1]
            sample['spec'] = _spec
        if self.use_mel:
            _mel = self._mel[idx] if self.in_memory else np.load(_mel_path)[...,::self.stride]        
            _n_frame = _mel.shape[-1]
            sample['mel'] = _mel
        sample['n_frame'] = _n_frame
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
            if self.use_spec:
                specs.append(np.pad(s['spec'], ((0, 0), (0, max_n_frame - n_frames[i])), constant_values=0.0))
            if self.use_mel:
                mels.append(np.pad(s['mel'], ((0, 0), (0, max_n_frame - n_frames[i])), constant_values=0.0))    
                    
        if self.tensor_type == 'torch':
            tensor = torch.tensor
            int32 = torch.int32
            int64 = torch.int64
            float32 = torch.float32
        elif self.tensor_type == 'numpy':
            tensor = np.array
            int32 = np.int32
            int64 = np.int64
            float32 = np.float32
        else:
            raise Exception('only torch or numpy is supported')
            
        batch = {'idx':tensor(idxes, dtype=int64), 
                 'text':tensor(texts, dtype=int64), 
                 'n_text':tensor(text_lengths, dtype=int32),
                 'n_frame':tensor(n_frames, dtype=int32)}
        if self.use_spec:
            batch['spec'] = tensor(specs, dtype=float32)
        if self.use_mel:
            batch['mel'] = tensor(mels, dtype=float32)
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
            return iter(list(_rand_idx.numpy()))
        else:
            return iter(list(_idx.numpy()))









    
    
