import os
from pathlib import Path
import numpy as np
import mne
from yasa import SleepStaging, bandpower_from_psd
from scipy.signal import welch
from somnus_utils import STAGE_MAP, align_epochs_with_stages, standardize_raw

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

def pick_eeg_channel(raw):    
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False)
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels found.")
    ch_names = np.array(raw.ch_names)[eeg_picks]
    target = None
    for preferred in ["Fpz-Cz", "Fz-Cz", "EEG Fpz-Cz"]:
        if preferred in ch_names:
            target = preferred
            break
    if target is None:
        target = ch_names[0]
    return target

def load_psg_hyp(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)
    raw = standardize_raw(raw)

   
    events, event_ids = mne.events_from_annotations(raw, event_id={
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3': 4,
        'Sleep stage 4': 4, 
        'Sleep stage R': 5
    }, verbose=False)

    stage_seq = []
    for e in events:
        code = e[2]
        if code == 1: stage_seq.append(STAGE_MAP['W'])
        elif code == 2: stage_seq.append(STAGE_MAP['N1'])
        elif code == 3: stage_seq.append(STAGE_MAP['N2'])
        elif code == 4: stage_seq.append(STAGE_MAP['N3'])
        elif code == 5: stage_seq.append(STAGE_MAP['R'])

    
    epochs = mne.make_fixed_length_epochs(raw, duration=30., preload=True, overlap=0, verbose=False)
    
    n_epochs = len(epochs)
    n_labels = len(stage_seq)
    n = min(n_epochs, n_labels)
    if n_epochs != n_labels:
     print(f"⚠️  Epochs ({n_epochs}) and labels ({n_labels}) differ, trimming to {n}")
    epochs = epochs[:n]
    y = np.array(stage_seq[:n], dtype=int)


    ch = pick_eeg_channel(raw)
    data = epochs.copy().pick(ch).get_data()[:, 0, :]  
    sf = raw.info['sfreq']

    feats = []
    for ep in data:
        f, pxx = welch(ep, fs=sf, nperseg=min(4*int(sf), len(ep)))
       
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30)}
        row = []
        for (lo, hi) in bands.values():
            idx = (f >= lo) & (f <= hi)
            row.append(np.trapz(pxx[idx], f[idx]))
        feats.append(row)
    X = np.array(feats)  

    return X, y

if __name__ == "__main__":
    print("This module provides feature extraction. Run somnus_train.py to train the model.")
 
