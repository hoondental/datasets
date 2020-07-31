import os
import sys
#sys.path.append(os.path.dirname(__file__))
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
import argparse
import re
from num2words import num2words
import pickle
from scipy.io import wavfile
import scipy.io



import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if __package__ == '':
    from stts import audio, audio_util, util
else:
    from .stts import audio, audio_util, util
    

        
    
# meta in memory use absolute path, metafile use relative path w.r.t the base dir of the metafile

def read_transcript(path_transcript, path_meta=None, ver='1.0'):
    dir_base = os.path.dirname(path_transcript)
    meta = []
    with open(path_transcript, 'r') as f:
        lines = f.readlines()
    if ver == '1.0':
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) == 4:
                print(parts)
                _path, script1, script2, duration = parts
                _fname = _path.split('/')[-1]
                meta.append((_fname, os.path.join(dir_base, _path), script2))
    elif ver == '1.1':
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) == 5:
                _path, script1, script2, script3, duration = parts
                _fname = _path.split('/')[-1][:-4]
                meta.append((_fname, os.path.join(dir_base, _path), script3))
    else:
        raise Exception(ver, ' is not supported')
        
    if path_meta is not None:
        save_meta(meta, path_meta)
    else:
        return meta
    

def read_meta(meta_path, spec_mel=False):  
    dir_base = os.path.dirname(meta_path)
    with open(meta_path, 'r') as f:
        lines = f.readlines()
    meta = []
    if spec_mel:
        for line in lines:
            wavpath, specpath, melpath, nframe, script = line.strip().split('|')
            _nframe = int(nframe)
            _fname = wavpath.split('/')[-1][:-4]
            _wavpath = os.path.join(dir_base, wavpath)        
            _specpath = os.path.join(dir_base, specpath)
            _melpath = os.path.join(dir_base, melpath)
            meta.append((_fname, _wavpath, _specpath, _melpath, _nframe, script))
    else:
        for line in lines:
            wavpath, script = line.strip().split('|')
            _fname = wavpath.split('/')[-1][:-4]
            _wavpath = os.path.join(dir_base, wavpath)        
            meta.append((_fname, _wavpath, script))
    return meta

def save_meta(meta, meta_path, spec_mel=False):
    dir_base = os.path.dirname(meta_path)
    with open(meta_path, 'w') as f:
        lines = ''
        if spec_mel:
            for _meta in meta:
                fname, wavpath, specpath, melpath, nframe, script = _meta
                _wavpath = os.path.relpath(wavpath, dir_base)
                _specpath = os.path.relpath(specpath, dir_base)
                _melpath = os.path.relpath(melpath, dir_base)
                _nframe = str(nframe)
                lines += '|'.join([_wavpath, _specpath, _melpath, _nframe, script]) + '\n'            
        else:
            for _meta in meta:
                fname, wavpath, script = _meta
                _wavpath = os.path.relpath(wavpath, dir_base)
                lines += '|'.join([_wavpath, script]) + '\n'
        f.write(lines[:-1])

def seperate_train_val(meta_path, val_ratio=0.2, shuffle=False, train_metafile='train_meta.txt', val_metafile='val_meta.txt'):
    dir_base = os.path.dirname(meta_path)
    meta = read_meta(meta_path)
    idx = [i for i in range(len(meta))]
    train = []
    val = []
    if shuffle:
        idx = np.random.permutation(idx)
    n_val = int(len(meta) * val_ratio)
    n_train = len(meta) - n_val
    val_idx = np.random.choice(idx, size=n_val, replace=False)
    for i in idx:
        if i in val_idx:
            val.append(meta[i])
        else:
            train.append(meta[i])
    save_meta(train, os.path.join(dir_base, train_metafile))
    save_meta(val, os.path.join(dir_base, val_metafile))
    
                  

def preprocess(meta_path, dir_data, out_subdir='train', sample_rate=22050, n_fft=1024, win_length=None, 
               hop_length=None, n_mels=80, mono=True, trim_db=None, decibel=True, normalize=True):
    meta = read_meta(meta_path)
    out_dir = os.path.join(dir_data, out_subdir)
    process_wavfiles(meta, out_dir, sample_rate, trim_db, mono, n_fft, win_length, hop_length, n_mels, decibel, normalize)
                     
    
    
def process_wavfiles(meta, out_dir, sample_rate=22050, trim_db=None, mono=True, n_fft=1024, win_length=None, hop_length=None, 
                     n_mels=80, decibel=True, normalize=True):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length / 4)
    _spec_dir = 'spec'
    _mel_dir = 'mel'
    spec_dir = os.path.join(out_dir, _spec_dir)
    mel_dir = os.path.join(out_dir, _mel_dir)
    meta_path = os.path.join(out_dir, 'meta.txt')
   
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(spec_dir):
        os.mkdir(spec_dir)
    if not os.path.exists(mel_dir):
        os.mkdir(mel_dir)
    wav_paths = []
    spec_paths = []
    mel_paths = []
    _meta = []
    for fname, wavpath, script in meta:
        wav_paths.append(wavpath)
        spec_file = fname + '.spec.npy'
        mel_file = fname + '.mel.npy'
        spec_path = os.path.join(spec_dir, spec_file)
        mel_path = os.path.join(mel_dir, mel_file)
        spec_paths.append(spec_path)
        mel_paths.append(mel_path)
        _meta.append([fname, wavpath, spec_path, mel_path, None, script])
                     
                     
    n_frames = audio_util.wav_to_spec_save_many(wav_paths, sample_rate, trim_db, mono, n_fft, win_length, hop_length, n_mels, 
                                           spec_paths, None, mel_paths, decibel, normalize)
    
    for i, _m in enumerate(_meta):
        _m[4] = n_frames[i]
                  
    save_meta(_meta, meta_path, spec_mel=True)
            
    with open(os.path.join(out_dir, 'settings.txt'), 'w') as f:
        f.write('sample_rate:' + str(sample_rate) + '\n')
        f.write('n_fft:' + str(n_fft) + '\n')
        f.write('win_length:' + str(win_length) + '\n')
        f.write('hop_length:' + str(hop_length) + '\n')
        f.write('n_mels:' + str(n_mels) + '\n')
        f.write('trim_db:' + str(trim_db) + '\n')
        f.write('mono:' + str(mono) + '\n')
        f.write('decibel:' + str(decibel) + '\n')
        f.write('normalize:' + str(normalize) + '\n')
        
