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

def main():
    parser = argparse.ArgumentParser(description='stft ClovaCall dataset to mel and linear spectrogram')
    parser.add_argument('--data_dir', '-d')
    parser.add_argument('--out_dir', '-o')
    parser.add_argument('--sample_rate', '-s', type=int, default=22050)
    parser.add_argument('--n_fft', '-f', type=int, default=1024)
    parser.add_argument('--win_length', '-w', type=int, default=1024)
    parser.add_argument('--hop_length', '-p', type=int, default=256)
    parser.add_argument('--n_mels', '-m', type=int, default=80)
    parser.add_argument('--mono', type=bool, default=True)
    parser.add_argument('--trim_db', type=int, default=None)
    parser.add_argument('--decibel', type=bool, default=True)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--num_vals', type=int, default=3100)
    args = parser.parse_args()
    
    preprocess(args.data_dir, args.out_dir, args.num_vals, args.sample_rate, args.n_fft, args.win_length, args.hop_length, 
               args.n_mels, args.mono, args.trim_db, args.decibel, args.normalize)

def make_metafile(dir_base, subdir_wav='wavs', metafile='meta.txt'):
    meta = []
    dir_wavs = os.path.join(dir_base, subdir_wav)
    for a, b, c in os.walk(dir_wavs):
        if len(c) == 0:
            continue
        for fname in c:
            if 'csv' in fname:
                fpath = os.path.join(a, fname)
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) != 2:
                            continue
                        _meta = (os.path.join(a[len(dir_base)+1:], parts[0].strip()), parts[1].strip())
                        meta.append(_meta)        
    with open(os.path.join(dir_base, metafile), 'w') as f:
        lines = ''
        for _meta in meta:
            lines += '|'.join(_meta) + '\n'
        f.write(lines[:-1])
    return meta


def seperate_train_val(metafile, dir_base, val_ratio=0.2, train_file='train_meta.txt', val_file='val_meta.txt'):
    meta = []
    meta_path = os.path.join(dir_base, metafile)
    with open(meta_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('|')
        meta.append((parts[0], parts[1]))
    _len = len(meta)
    n_val = int(_len * val_ratio)
    n_train = _len - n_val
    train_meta = []
    val_meta = []
    _train_index = np.random.choice(_len, n_train, replace=False)
    for i in range(_len):
        if i in _train_index:
            train_meta.append(meta[i])
        else:
            val_meta.append(meta[i])
    with open(os.path.join(dir_base, train_file), 'w') as f:
        lines = ''
        for _meta in train_meta:
            lines += '|'.join(_meta) + '\n'
        f.write(lines[:-1])
    with open(os.path.join(dir_base, val_file), 'w') as f:
        lines = ''
        for _meta in val_meta:
            lines += '|'.join(_meta) + '\n'
        f.write(lines[:-1])
        
    
def preprocess(data_dir, metafile, out_subdir='train', sample_rate=22050, n_fft=1024, win_length=None, 
               hop_length=None, n_mels=80, mono=True, trim_db=None, decibel=True, normalize=True):
    meta = read_metadata(data_dir, metafile)
    out_dir = os.path.join(data_dir, out_subdir)
    process_wavfiles(meta, out_dir, sample_rate, trim_db, mono, n_fft, win_length, hop_length, n_mels, decibel, normalize)
    
def read_metadata(data_dir, metafile='meta.txt'):    
    metadata_path = os.path.join(data_dir, metafile)
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    meta = []
    for line in lines:
        wpath, ntext = line.strip().split('|')
        if ntext.endswith('\n'):
            ntext = ntext[:-1]
        fname = wpath.split('/')[-1][:-4]
        meta.append((fname, os.path.join(data_dir, wpath), ntext))
    return meta


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
    
if __name__ == '__main__':
    main()
            
