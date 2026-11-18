"""Fetch the author's smallest checkpoint in resumable ranges and verify LFS SHA256."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
SIZE = 2377399148
SHA = 'ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab'
URL = 'https://hf-mirror.com/Weibang/NeuroLM/resolve/main/checkpoints/NeuroLM-B.pt'
DEST = ROOT/'data/models/NeuroLM-B.pt'


def main():
    if DEST.exists():
        print('Checkpoint already exists; verify its hash before loading.')
        return
    chunks = DEST.with_suffix('.parts')
    chunks.mkdir(parents=True, exist_ok=True)
    width = 16 * 1024**2
    def get(i):
        start, end = i*width, min(SIZE, (i+1)*width)-1
        p = chunks/str(i)
        if not p.exists() or p.stat().st_size != end-start+1:
            subprocess.run(['curl','-4','-fLsS','--connect-timeout','20','--max-time','600',
                            '--retry','2','--range',f'{start}-{end}',
                            URL+f'?download=true&part={i}', '-o',str(p)],check=True)
        assert p.stat().st_size == end-start+1, (i,p.stat().st_size)
        print('downloaded range',i,flush=True)
        return p
    with ThreadPoolExecutor(max_workers=16) as pool:
        pieces = list(pool.map(get, range((SIZE+width-1)//width)))
    digest = hashlib.sha256()
    part = DEST.with_suffix('.part')
    with part.open('wb') as out:
        for p in pieces:
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024**2),b''):
                    out.write(block); digest.update(block)
    assert part.stat().st_size == SIZE and digest.hexdigest() == SHA
    part.replace(DEST)
    shutil.rmtree(chunks)
    print('Verified',DEST,SHA,flush=True)


if __name__ == '__main__':
    main()
