# -*- coding: utf-8 -*-
import glob
import os
import re
import time
import datetime
import librosa.util as librosa_util
import librosa

if __package__ == '':
    from stts import audio, audio_util, util
else:
    from .stts import audio, audio_util, util

LANG = 'KRCC'
STRF = '%H:%M:%S,%f'
Ztime = datetime.datetime.strptime('00:00:00,000', STRF)

def parse(path, start=0, encoding=None):
    if encoding is None:
        try:
            with open(path, 'r', encoding='utf8') as f:
                lines = f.readlines()
        except:
            try: 
                with open(path, 'r', encoding='cp949') as f:
                    lines = f.readlines()
            except:
                raise Exception('both utf8 and cp949 failed to read subtitle file: ', path)
    else:
        try:
            with open(path, 'r', encoding=encoding) as f:
                lines = f.readlines()
        except:
            raise Exception(encoding + ' failed to read subtitle file: ', path) 
            
    if path.endswith('smi'):
        ext = 'smi'
        lines = '\n'.join(lines)
        return parse_smi(lines, start)
    elif path.endswith('srt'):
        ext = 'srt'
        return parse_srt(lines, start)
    else:
        raise Exception('only smi or srt is valid subtitle')
    

def _normalize_subtitle(s):
    s = s.replace('-', ' ').strip()  
    s = s.replace('|', '/')
    s = s.replace('<br>', ' ')
    while('<' in s):    
        idx0 = s.index('<')
        try:
            idx1 = s.index('>')
        except:
            s = s.replace('<', '')
        else:
            if idx0 < idx1:
                s = (s[:idx0] + s[idx1+1:]).strip()
            else:
                s = s[:idx1] + s[idx1+1:]
        
    while('(' in s):    
        idx0 = s.index('(')
        try:
            idx1 = s.index(')')
        except:
            s = s.replace('(', '')
        else:
            if idx0 < idx1:
                s = (s[:idx0] + s[idx1+1:]).strip()
            else:
                s = s[:idx1] + s[idx1+1:]
    if ':' in s:
        idx = s.index(':')
        s = s[idx+1:].strip()
    while('\n' in s):    
        idx0 = s.index('\n')
        try:
            idx1 = s.index(':')
        except:
            s = s.replace('\n', ' ')
        else:
            if idx0 < idx1:
                s = (s[:idx0] + s[idx1+1:]).strip()
            else:
                s = s[:idx1] + s[idx1+1:]  
    return s.strip()

def parse_srt(lines, start=0):
    subtitle = []
    _time = None
    _content = ''
    _seq_no = None
    for line in lines:
        line = line.strip()
        if _seq_no is None:
            try:
                _seq_no = int(line)
            except Exception as ex:
                print(ex, line)
        elif len(line) == 0:
            if _time is not None and len(_content) > 0:
                if _time[0] > start:
                    subtitle.append([_time, _content])
            _time = None
            _content = ''
            _seq_no = None
        elif '-->' in line:
            if _seq_no is None:
                continue
            _start, _end = line.split('-->')
            _start = (datetime.datetime.strptime(_start.strip(), STRF) - Ztime).total_seconds()
            _end = (datetime.datetime.strptime(_end.strip(), STRF) - Ztime).total_seconds()
            _time = [_start, _end]
        else:
            if _time is None:
                continue
            line = _normalize_subtitle(line)
            _content = (_content + ' ' + line).strip()
    return subtitle
        

def parse_smi(smi, start=0):
    smi = _parse_smi(smi)
    _subtitle = []
    for t, s in smi:
        t = float(t) / 1000
        s = s['KRCC']
        s = _normalize_subtitle(s)
        _subtitle.append([t, s])    
    
    _start = None
    _end = None
    _content = ''
    subtitle = []
    for t, s in _subtitle:
        if t < start:
            continue
        if _start is not None: # closing
            _end = t
            _content = _content.strip()
            if len(_content):
                subtitle.append([[_start, _end], _content])
            _start = None
            _end = None
            _content = ''

        if not 'nbsp' in s: # starting
            _start = t
            _content = s
    return subtitle
                
    

def _parse_smi(smi): # smi parsing algorithm written in PYSAMI by g6123 (https://github.com/g6123/PySAMI)
    search = lambda string, pattern: re.search(pattern, string, flags=re.I)
    tags = re.compile('<[^>]+>')

    def split_content(string, tag):
        threshold = '<'+tag

        if tag is 'p':
            standard = threshold + ' Class=' + LANG + '>'
            if standard.upper() not in string.upper():
                idx = string.find('>') + 1
                string = string[:idx] + standard + string[idx:]

        return list(map(
            lambda item: (threshold+item).strip(),
            re.split(threshold, string, flags=re.I)
        ))[1:]

    def remove_tag(matchobj):
        matchtag = matchobj.group().lower()
        keep_tags = ['font', 'b', 'i', 'u']
        for keep_tag in keep_tags:
            if keep_tag in matchtag:
                return matchtag
        return ''

    def parse_p(item):
        lang = search(item, '<p(.+)class=([a-z]+)').group(2)
        content = item[search(item, '<p[^>]+>').end():]
        content = content.replace('\r', '')
        content = content.replace('\n', '')
        content = re.sub('<br ?/?>', '\n', content, flags=re.I)
        content = re.sub('<[^>]+>', remove_tag, content)
        return [lang, content]

    data = []

    try:
        for item in split_content(smi, 'sync'):
            pattern = search(item, '<sync start=([0-9]+)')
            if pattern!=None:
                timecode = pattern.group(1)
                content = dict(map(parse_p, split_content(item, 'p')))
                data.append([timecode, content])
    except Exception as ex:
        print(ex)
        print('Conversion ERROR: maybe this file is not supported.')
    
    return data


def extract_wav(path_wav, path_cap, sample_rate=16000, trim_db=None, start=0, end=0, start_margin=0, end_margin=0):
    wav, sr = audio.load_wav(path_wav, sample_rate)
    subtitle = parse(path_cap)
    parts = []
    _max = wav.shape[-1]
    _len = _max / sample_rate
    
    for cap in subtitle:
        _script = cap[1]
        _start = cap[0][0] - start_margin
        _end = cap[0][1] + end_margin
        if _start < start or _len - _end < end:
            continue
        if '청각장애인' in _script or 'blog.naver.com' in _script:
            continue
        _sidx = int(max(0, _start * sr))
        _eidx = int(min(_max, _end * sr))
        _wav = wav[_sidx:_eidx]
        if trim_db is not None:
            _wav, _ = librosa.effects.trim(_wav, top_db=trim_db)
        if _wav.shape[-1] > 2048:
            parts.append([_script, _wav])
    return parts
        
def extract_wav_save(path_wav, path_cap, dir_out, metafile='meta.txt', prefix='', sample_rate=16000, trim_db=None, 
                     start=0, end=0, start_margin=0, end_margin=0):
    os.makedirs(dir_out, exist_ok=True)
    subtitles = extract_wav(path_wav, path_cap, sample_rate, trim_db, start, end, start_margin, end_margin)
    meta = []
    for i, cap in enumerate(subtitles):
        script, wav = cap
        fname = prefix + '_' + '%06d'%(i)
        path = os.path.join(dir_out, fname + '.wav')
        audio.save_wav(wav, path, sample_rate)
        meta.append((path, script)) 
    metapath = os.path.join(dir_out, metafile)
    with open(metapath, 'w') as f:
        lines = ''
        for p, s in meta:
            lines += '|'.join([os.path.relpath(p, dir_out), s]) + '\n'
        f.write(lines[:-1])