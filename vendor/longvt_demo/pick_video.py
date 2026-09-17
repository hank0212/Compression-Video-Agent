import pandas as pd, os, cv2
pq="/local1/cfyang/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"
df=pd.read_parquet(pq)
print("shape:", df.shape, "| cols:", list(df.columns))
if 'duration' in df.columns:
    print("duration vals:", df['duration'].value_counts().to_dict())
D="/local1/cfyang/.cache/huggingface/videomme/videomme/data"
vidcol = 'videoID' if 'videoID' in df.columns else ('video_id' if 'video_id' in df.columns else df.columns[0])
print("using video col:", vidcol)
df['on_disk']=df[vidcol].apply(lambda v: os.path.exists(f"{D}/{v}.mp4"))
d=df[df['on_disk'] & (df.get('duration')=='medium')] if 'duration' in df.columns else df[df['on_disk']]
print("medium on disk:", len(d))
seen=set()
for _,r in d.iterrows():
    v=r[vidcol]
    if v in seen: continue
    seen.add(v)
    cap=cv2.VideoCapture(f"{D}/{v}.mp4"); dur=cap.get(7)/max(cap.get(5),1); cap.release()
    print(f"\n=== {v}  dur={dur:.0f}s  task={r.get('task_type')} ===")
    print("Q:", r['question'])
    print("opts:", list(r['options']))
    print("ans:", r['answer'])
    if len(seen)>=4: break
