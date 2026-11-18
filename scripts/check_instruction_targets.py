"""Check answer supervision against the actual tokenized inference prefix."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT/'data/tokenizer-cache')
import pickle
import tiktoken
from downstream_dataset import WorkloadLoader, HMCLoader

enc = tiktoken.get_encoding('gpt2')
for task, cls in [('EEGWorkload',WorkloadLoader),('HMC',HMCLoader)]:
    folder = ROOT/'data/processed'/task/'train'
    seen = set()
    for path in sorted(folder.glob('*.pkl')):
        with path.open('rb') as f: y=int(pickle.load(f)['y'])
        if y in seen:continue
        seen.add(y)
        ds = cls(str(folder),[path.name],is_instruct=True,eeg_max_len=276,text_max_len=80)
        _,text,target,*_ = ds[0]
        prefix = ds.prompt
        assert text[:len(prefix)].equal(prefix), (task,'inference prefix differs from training')
        actual = target[target!=-1]
        expected = text[len(prefix):len(prefix)+len(actual)]
        print(task,y,'actual:',repr(enc.decode(actual.tolist())),
              'expected:',repr(enc.decode(expected.tolist())),flush=True)
        assert actual.equal(expected),(task,y,'answer supervision is shifted')
print('ALL ANSWER ALIGNMENT CHECKS PASSED')
