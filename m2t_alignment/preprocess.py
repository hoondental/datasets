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
import argparse

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from stts import audio, audio_util, util, textutil

def main():
    parser = argparse.ArgumentParser(description='generate mprob and tprob with a trained Aligner')
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
    
    
def preprocess(data_dir, out_dir, num_vals=3100, sample_rate=22050, n_fft=1024, win_length=1024, 
               hop_length=256, n_mels=80, mono=True, trim_db=None, decibel=True, normalize=True):
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    meta = read_metadata(data_dir)
    train, val = seperate_train_val(meta, num_vals)
    process_wavfiles(train, train_dir, sample_rate, trim_db, mono, n_fft, win_length, hop_length, n_mels, decibel, normalize)
    process_wavfiles(val, val_dir, sample_rate, trim_db, mono, n_fft, win_length, hop_length, n_mels, decibel, normalize)
    

def load_aligner(trained_path, chars, embedding_dim=128, n_mels=80, text_upsample=3, kernel_size=3, padding='same', only_embedding=False, 
                 te_dilation=1, te_layers=3, me_dilation=1, me_layers=3, key_normalize=False, share_qv=True):
    



def read_metadata(data_dir, metafile='metadata.csv', wav_dir='./wavs'):    
    metadata_path = os.path.join(data_dir, metafile)
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    meta = []
    for line in lines:
        fname, text, ntext = line.strip().split('|')
        if ntext.endswith('\n'):
            ntext = ntext[:-1]
        meta.append((fname, os.path.join(data_dir, wav_dir, fname + '.wav'), ntext))
    return meta

def seperate_train_val(meta, num_val=3100, shuffle=True):
    idx = [i for i in range(len(meta))]
    if shuffle:
        idx = np.random.permutation(idx)
    train_idx = idx[num_val:]
    val_idx = idx[:num_val]
    train = []
    val = []
    for i in train_idx:
        train.append(meta[i])
    for i in val_idx:
        val.append(meta[i])
    return train, val

def process_wavfiles(meta, out_dir, sample_rate=22050, trim_db=None, mono=True, n_fft=1024, win_length=None, hop_length=None, 
                     n_mels=80, decibel=True, normalize=True):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length / 4)
    spec_dir = os.path.join(out_dir, 'spec')
    mel_dir = os.path.join(out_dir, 'mel')
    meta_path = os.path.join(out_dir, 'meta.txt')
   
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
        _meta.append((fname, ntext, os.path.join('spec', spec_file), os.path.join('mel', mel_file)))
                     
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
            
