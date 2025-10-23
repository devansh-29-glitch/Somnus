import os
from pathlib import Path
import mne
from mne.datasets.sleep_physionet import age

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    # Download 2 subjects (keeps it light). You can increase n_subjects later.
    print("Downloading Sleep-EDF subset (2 subjects)...")
    records = age.fetch_data(subjects=[0, 1], recording=[1])  # (PSG, Hypnogram) pairs

    # Copy file paths into data/ (MNE returns local paths; we just show the list)
    print("\nDownloaded files:")
    for psg_path, hyp_path in records:
        print("PSG:", psg_path)
        print("HYP:", hyp_path)
    print("\nDone. We'll load directly by paths in somnus_features.py")
