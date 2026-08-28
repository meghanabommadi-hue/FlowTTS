#!/usr/bin/env python3
"""One-shot sanity check: align a genuinely long clip and report the timings."""
import sys, io, base64, time, statistics, requests
sys.path.insert(0, '/home/jovyan/omnivoice-train/omnivoice_training')
from hf_parquet import list_shards, HttpFile
import pyarrow.parquet as pq, soundfile as sf

BASE = '/home/jovyan/omnivoice-train'
tok = open(f'{BASE}/token.read').read().strip()
lang = sys.argv[1] if len(sys.argv) > 1 else 'hau'
wlang = {'hau': 'ha', 'ibo': 'ig', 'yor': 'yo', 'pcm': 'pcm'}[lang]

shards = list_shards('kapturecx/ohun', 'train', subdir=lang, token=tok)
path, size = shards[0]
pf = pq.ParquetFile(HttpFile(f'https://huggingface.co/datasets/kapturecx/ohun/resolve/main/{path}', size, tok))
cols = [c for c in ['audio_id', 'transcript', 'duration_seconds', 'audio', 'audio_path']
        if c in pf.schema_arrow.names]
tbl = pf.read_row_group(0, columns=cols).to_pylist()

def dur_of(row):
    d = row.get('duration_seconds')
    if d:
        return float(d)
    b = (row.get('audio') or row.get('audio_path'))
    return len(b['bytes']) / 96000.0 if isinstance(b, dict) and b.get('bytes') else 0.0

longs = [x for x in tbl if dur_of(x) > 60]
print(f'{lang}: {len(tbl)} rows in rg0, {len(longs)} longer than 60s', flush=True)
if not longs:
    sys.exit('no long clips in this row group')
r = max(longs, key=dur_of)
blob = (r.get('audio') or r.get('audio_path'))
wav, sr = sf.read(io.BytesIO(blob['bytes']), dtype='float32')
gt = r['transcript']
buf = io.BytesIO(); sf.write(buf, wav, sr, format='WAV', subtype='PCM_16')

print(f"clip {r['audio_id']} dur={dur_of(r):.0f}s gt_words={len(gt.split())}", flush=True)
t0 = time.time()
resp = requests.post('http://127.0.0.1:8899/align',
                     json={'audio_b64': base64.b64encode(buf.getvalue()).decode(),
                           'transcript': gt, 'language': wlang, 'sample_rate': sr},
                     timeout=1800)
print('HTTP', resp.status_code, f'in {time.time()-t0:.0f}s', flush=True)
if resp.status_code != 200:
    print(resp.text[:500]); sys.exit(1)
d = resp.json()
print(f"aligned={d['duration']:.1f}s asr_words={d['n_words']}", flush=True)
for w in d['words'][:5]:
    print('   ', w)
gaps = [d['words'][i+1]['start'] - d['words'][i]['end'] for i in range(len(d['words'])-1)]
if gaps:
    print(f'gaps median={statistics.median(gaps):.2f}s max={max(gaps):.2f}s '
          f'pauses>0.3s={sum(1 for g in gaps if g > 0.3)}')
print('GT :', gt[:120])
print('ASR:', d.get('asr_text', '')[:120])
