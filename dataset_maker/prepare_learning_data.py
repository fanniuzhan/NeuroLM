"""Reproducible learning subsets. Run on a Slurm compute node, never login."""
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed'
TRACE = []


def read(path, channels, notch):
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    before = {'file': str(path.relative_to(ROOT)), 'sfreq': raw.info['sfreq'],
              'shape': [len(raw.ch_names), int(raw.n_times)], 'channels': raw.ch_names}
    raw.pick(channels)
    raw.filter(0.1, 75., n_jobs=1, verbose='ERROR')
    raw.notch_filter(notch, n_jobs=1, verbose='ERROR')
    raw.resample(200, n_jobs=1, verbose='ERROR')
    x = raw.get_data(units='uV').astype(np.float32)
    assert np.isfinite(x).all()
    TRACE.append({**before, 'selected': channels, 'after_shape': list(x.shape),
                  'after_sfreq': 200, 'unit': 'microvolt', 'notch_hz': notch,
                  'min_uV': float(x.min()), 'max_uV': float(x.max())})
    raw.close()
    return x


def save(task, split, stem, x, names, label=None, source=None, start=None):
    assert x.ndim == 2 and x.shape[0] == len(names) and x.shape[1] % 200 == 0
    assert np.isfinite(x).all()
    path = OUT / task / split / (stem + '.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = {'X': x, 'ch_names': names, 'source': source, 'start_seconds': start}
    if label is not None:
        sample['y'] = int(label)
    with path.open('wb') as f:
        pickle.dump(sample, f)


def main():
    mne.set_log_level('ERROR')
    workload_channels = ['EEG ' + n for n in
        'Fp1 Fp2 F3 F4 F7 F8 T3 T4 C3 C4 T5 T6 P3 P4 O1 O2 Fz Cz Pz'.split()]
    for path in sorted((ROOT/'data/raw/eegmat-1.0.0').glob('*.edf')):
        subject = int(path.stem[7:9])
        split = 'train' if subject < 26 else 'eval' if subject < 31 else 'test'
        label = int(path.stem[-1]) - 1
        x = read(path, workload_channels, 50)
        offset = x.shape[1]/200 - 60
        x = x[:, -12000:]
        assert x.shape == (19, 12000)
        for i in range(29):
            save('EEGWorkload', split, f'{path.stem}-{i:03d}', x[:, i*400:i*400+800],
                 [n.split()[-1].upper() for n in workload_channels], label, path.name, offset+i*2)
    stages = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2,
              'Sleep stage N3': 3, 'Sleep stage R': 4}
    channels = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']
    for subject, split in [('SN001', 'train'), ('SN103', 'eval'), ('SN128', 'test')]:
        path = ROOT/'data/raw/hmc-1.1/recordings'/f'{subject}.edf'
        x = read(path, channels, 50)
        with path.with_name(subject+'_sleepscoring.txt').open() as f:
            for i, row in enumerate(csv.DictReader(f)):
                stage = row[' Annotation'].strip()
                if stage not in stages:
                    assert stage in ('Lights off', 'Lights on'), row
                    continue
                start = round(float(row[' Recording onset'])*200)
                duration = float(row[' Duration'])
                assert duration == 30 and start >= 0 and start+6000 <= x.shape[1], row
                save('HMC', split, f'{subject}-{i:04d}', x[:, start:start+6000],
                     ['F4', 'C4', 'O2', 'C3'], stages[stage], path.name, start/200)
    for subject, split in [('S001','train'), ('S002','val'), ('S003','test')]:
        for path in sorted((ROOT/'data/raw/eegmmidb-1.0.0'/subject).glob('*.edf')):
            raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
            channels = raw.ch_names
            raw.close()
            x = read(path, channels, 60)
            names = [n.rstrip('.').upper() for n in channels]
            assert len(names) == 64
            width = (1024//64)*200
            for i in range(x.shape[1]//width):
                save('MotorImagery', split, f'{path.stem}-{i:03d}', x[:, i*width:(i+1)*width],
                     names, source=path.name, start=i*16)
    counts = {}
    for task in ('EEGWorkload', 'HMC', 'MotorImagery'):
        counts[task] = {}
        for folder in sorted((OUT/task).iterdir()):
            labels = Counter()
            files = list(folder.glob('*.pkl'))
            for path in files:
                with path.open('rb') as f:
                    sample = pickle.load(f)
                labels[str(sample.get('y', 'unlabeled'))] += 1
            counts[task][folder.name] = {'samples':len(files), 'labels':dict(labels)}
    assert sum(v['samples'] for v in counts['EEGWorkload'].values()) == 2088
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'preprocessing_trace.json').write_text(json.dumps({'files': TRACE, 'counts':counts}, indent=2)+'\n')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
