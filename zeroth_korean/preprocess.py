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

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if __package__ == '':
    from stts import audio, audio_util, util
else:
    from .stts import audio, audio_util, util
    
    
    
def _make_metafile(dir_base, subdir='train_data_01', metafile='train_meta.txt'):
    meta = []
    dir_data = os.path.join(dir_base, subdir)
    for a, b, c in os.walk(dir_data):
        if len(c) == 0:
            continue
        for fname in c:
            if 'txt' in fname:
                fpath = os.path.join(a, fname)
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            line = line.strip()
                            idx_sep = line.index(' ')
                            _flacfile = line[:idx_sep] + '.flac'
                            _script = line[idx_sep+1:]
                        except:
                            continue
                        else:
                            _meta = (os.path.join(a[len(dir_base)+1:], _flacfile), _script)
                        meta.append(_meta)        
    with open(os.path.join(dir_base, metafile), 'w') as f:
        lines = ''
        for _meta in meta:
            lines += '|'.join(_meta) + '\n'
        f.write(lines[:-1])
    return meta

def make_metafiles(dir_base, subdir_train='train_data_01', subdir_test='test_data_01'):
    _make_metafile(dir_base, subdir=subdir_train, metafile='train_meta.txt')
    _make_metafile(dir_base, subdir=subdir_test, metafile='test_meta.txt')
    

def read_metadata(data_dir, metafile='meta.txt'):    
    metadata_path = os.path.join(data_dir, metafile)
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    meta = []
    for line in lines:
        wpath, ntext = line.strip().split('|')
        if ntext.endswith('\n'):
            ntext = ntext[:-1]
        fname = wpath.split('/')[-1][:-5]
        meta.append((fname, os.path.join(data_dir, wpath), ntext))
    return meta

def preprocess(data_dir, metafile, out_subdir='train', sample_rate=22050, n_fft=1024, win_length=None, 
               hop_length=None, n_mels=80, mono=True, trim_db=None, decibel=True, normalize=True):
    meta = read_metadata(data_dir, metafile)
    out_dir = os.path.join(data_dir, out_subdir)
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
    for fname, path, ntext in meta:
        wav_paths.append(path)
        spec_file = fname + '.spec.npy'
        mel_file = fname + '.mel.npy'
        spec_path = os.path.join(spec_dir, spec_file)
        mel_path = os.path.join(mel_dir, mel_file)
        spec_paths.append(spec_path)
        mel_paths.append(mel_path)
        _meta.append((fname, ntext, os.path.join(_spec_dir, spec_file), os.path.join(_mel_dir, mel_file)))
                     
    n_frames = audio_util.wav_to_spec_save_many(wav_paths, sample_rate, trim_db, mono, n_fft, win_length, hop_length, n_mels, 
                                           spec_paths, None, mel_paths, decibel, normalize)
    with open(meta_path, 'w') as f: 
        for i, m in enumerate(_meta):
            f.write(m[0] + '|' + m[1] + '|' + str(n_frames[i]) + '|' + m[2] + '|' + m[3] + '\n')
            
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