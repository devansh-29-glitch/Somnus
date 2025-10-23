import numpy as np
import mne

STAGE_MAP = {
    'W': 0,   
    'N1': 1,  
    'N2': 2,  
    'N3': 3,  
    'R': 4    
}

INV_STAGE_MAP = {v: k for k, v in STAGE_MAP.items()}

def standardize_raw(raw):
    """1–40 Hz bandpass + adaptive notch filter."""
    raw.load_data()
    sfreq = raw.info['sfreq']

    # Band-pass between 0.3 and 40 Hz
    raw.filter(0.3, 40., fir_design='firwin')

    # Apply notch only if Nyquist > 51 Hz
    if sfreq / 2 > 51:
        raw.notch_filter(50.)
    else:
        print(f"Skipping notch filter — Nyquist = {sfreq/2:.1f} Hz too low")

    return raw

def align_epochs_with_stages(epochs, stage_labels):
    """Trim stage labels to number of epochs (30s windows)."""
    n = len(epochs)
    y = np.array(stage_labels[:n], dtype=int)
    return y

def plot_and_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
