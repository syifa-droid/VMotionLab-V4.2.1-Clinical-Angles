from __future__ import annotations
from pathlib import Path
import json, re
import numpy as np, pandas as pd
from .constants import HALPE26

def _num(path):
    m=re.findall(r'(\d+)', path.stem); return int(m[-1]) if m else 0

def camera_json_dirs(pose_dir):
    result=[]
    for d in Path(pose_dir).rglob('*'):
        if d.is_dir() and any(d.glob('*.json')): result.append(d)
    if any(Path(pose_dir).glob('*.json')): result.append(Path(pose_dir))
    return sorted(set(result))

def match_camera_dir(pose_dir, video_stem):
    dirs=camera_json_dirs(pose_dir)
    key=re.sub(r'[^a-z0-9]','',video_stem.lower())
    for d in dirs:
        dk=re.sub(r'[^a-z0-9]','',d.name.lower())
        if key in dk or dk in key: return d
    if len(dirs)==1: return dirs[0]
    raise FileNotFoundError(f'Could not identify pose JSON directory for {video_stem}. Found: {[d.name for d in dirs]}')

def read_openpose_json_dir(directory, fps):
    files=sorted(Path(directory).glob('*.json'), key=_num)
    if not files: raise FileNotFoundError(f'No pose JSON files in {directory}')
    rows=[]
    for i,p in enumerate(files):
        payload=json.loads(p.read_text(encoding='utf-8'))
        people=payload.get('people',[])
        person=None
        if people:
            person=max(people, key=lambda q: np.nanmean(np.asarray(q.get('pose_keypoints_2d',[])[2::3],float)) if q.get('pose_keypoints_2d') else -1)
        arr=np.full((26,3),np.nan)
        if person:
            vals=np.asarray(person.get('pose_keypoints_2d',[]),float)
            n=min(26,len(vals)//3); arr[:n]=vals[:n*3].reshape(n,3)
        row={'frame':i,'source_frame':_num(p),'time':i/float(fps)}
        for name,idx in HALPE26.items():
            row[f'{name}_x']=arr[idx,0]; row[f'{name}_y']=arr[idx,1]; row[f'{name}_visibility']=arr[idx,2]; row[f'{name}_presence']=arr[idx,2]
        rows.append(row)
    return pd.DataFrame(rows)

def find_pose_video(pose_dir, stem):
    videos=list(Path(pose_dir).rglob('*.mp4'))+list(Path(pose_dir).rglob('*.avi'))
    key=re.sub(r'[^a-z0-9]','',stem.lower())
    for p in videos:
        pk=re.sub(r'[^a-z0-9]','',p.stem.lower())
        if key in pk: return p
    return videos[0] if len(videos)==1 else None
