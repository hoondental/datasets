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
    


def process_text(text, as_sounds=True):
    text = text.replace('*', '')
    text = text.replace('+', '')
    text = text.replace('u/', '')
    text = text.replace('b/', '')
    text = text.replace('b/', '')
    text = text.replace('l/', '')
    text = text.replace('o/', '')
    text = text.replace('n/', '')
    text = text.replace('/', '')
    while(')(' in text):
        try:
            idx1 = text.index('(')
            idx2 = text.index(')')
            idx3 = text.index('(', idx2)
            idx4 = text.index(')', idx3)
            if idx3 != idx2 + 1:
                raise Exception('unexpected text with (): ')
        except Exception as ex:
            print(ex)
            print(text)
        else:
            if as_sounds:
                text = text[:idx1] + text[idx3+1:idx4] + text[idx4+1:]
            else:
                text = text[:idx1] + text[idx1+1:idx2] + text[idx4+1:]
    return text.strip()
        
    
# meta in memory use absolute path, metafile use relative path w.r.t the base dir of the metafile
    
def make_metafile(dir_data, metapath=None):
    if metapath is None:
        metapath = os.path.join(dir_data, 'meta.txt')
    dir_base = os.path.dirname(metapath)
    
    meta = []
    for a, b, c in os.walk(dir_data):
        if len(c) == 0:
            continue
        for fname in c:
            if 'txt' in fname:
                _fname = fname[:-4]
                txtpath = os.path.join(a, fname)
                with open(txtpath, 'r', encoding='cp949') as f:
                    script = ' '.join(f.readlines()).strip()
                script_ph = process_text(script, True)
                script_gr = process_text(script, False)
                pcmpath = os.path.join(a, _fname + '.pcm')
                if os.path.exists(pcmpath):
                    meta.append((_fname, pcmpath, script, script_gr, script_ph))
    save_meta(meta, metapath)
    

def read_meta(meta_path, spec_mel=False):  
    dir_base = os.path.dirname(meta_path)
    with open(meta_path, 'r') as f:
        lines = f.readlines()
    meta = []
    if spec_mel:
        for line in lines:
            pcmpath, specpath, melpath, nframe, script, script_gr, script_ph = line.strip().split('|')
            _nframe = int(nframe)
            _fname = pcmpath.split('/')[-1][:-4]
            _pcmpath = os.path.join(dir_base, pcmpath)        
            _specpath = os.path.join(dir_base, specpath)
            _melpath = os.path.join(dir_base, melpath)
            meta.append((_fname, _pcmpath, _specpath, _melpath, _nframe, script, script_gr, script_ph))
    else:
        for line in lines:
            pcmpath, script, script_gr, script_ph = line.strip().split('|')
            _fname = pcmpath.split('/')[-1][:-4]
            _pcmpath = os.path.join(dir_base, pcmpath)        
            meta.append((_fname, _pcmpath, script, script_gr, script_ph))
    return meta

def save_meta(meta, meta_path, spec_mel=False):
    dir_base = os.path.dirname(meta_path)
    with open(meta_path, 'w') as f:
        lines = ''
        if spec_mel:
            for _meta in meta:
                fname, pcmpath, specpath, melpath, nframe, script, script_gr, script_ph = _meta
                _pcmpath = os.path.relpath(pcmpath, dir_base)
                _specpath = os.path.relpath(specpath, dir_base)
                _melpath = os.path.relpath(melpath, dir_base)
                _nframe = str(nframe)
                lines += '|'.join([_pcmpath, _specpath, _melpath, _nframe, script, script_gr, script_ph]) + '\n'            
        else:
            for _meta in meta:
                fname, pcmpath, script, script_gr, script_ph = _meta
                _pcmpath = os.path.relpath(pcmpath, dir_base)
                lines += '|'.join([_pcmpath, script, script_gr, script_ph]) + '\n'
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
    for fname, pcmpath, script, script_gr, script_ph in meta:
        wav_paths.append(pcmpath)
        spec_file = fname + '.spec.npy'
        mel_file = fname + '.mel.npy'
        spec_path = os.path.join(spec_dir, spec_file)
        mel_path = os.path.join(mel_dir, mel_file)
        spec_paths.append(spec_path)
        mel_paths.append(mel_path)
        _meta.append([fname, pcmpath, spec_path, mel_path, None, script, script_gr, script_ph])
                     
                     
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
        
