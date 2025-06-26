"""
sEMG Fatigue Analysis Pipeline with NDI, NES, MOS, and Set Comparison

- Loads two sEMG data files (Excel/CSV).
- Merges signals side by side (concatenates in time).
- Adds a black dashed line at the boundary between sets in all plots.
- Calculates MOS (Mechanical Output Score) for each set: MOS = (AUC × Peak) / Tactive.
- Calculates NDI, NES, and Tactive for the merged signal and for each set.
- Plots signals, features, fatigue index, and annotated metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PyEMD import EMD
from tkinter import Tk, filedialog

# ===== Parameters =====
FS = 500
WINDOW_SIZE = 500
STEP_SIZE = 125
LOWPASS_CUTOFF = 0.08
LOWPASS_ORDER = 2
DOWNSAMPLED_FS = FS // STEP_SIZE
THRESHOLD = 0.05

# ===== File Selection =====
def select_up_to_three_files():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select up to THREE Excel Files",
        filetypes=[("Excel files", "*.xlsx *.xls *.csv")]
    )
    return list(file_paths)[:3]

# ===== Signal Smoothing =====
def smooth_signal(signal, window_len):
    window = np.ones(window_len) / window_len
    return np.convolve(signal, window, mode='same')

# ===== Windowing =====
def window_signal(signal):
    n = len(signal)
    windows = []
    indices = []
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        windows.append(signal[start:end])
        indices.append(start)
    return np.array(windows), np.array(indices)

# ===== Feature Extraction =====
def calc_mnf_arv_ratio(window):
    freqs = np.fft.rfftfreq(len(window), d=1/FS)
    fft_vals = np.fft.rfft(window)
    psd = np.abs(fft_vals) ** 2
    mnf = np.sum(freqs * psd) / np.sum(psd)
    arv = np.mean(np.abs(window))
    return mnf / arv

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def calc_ima_diff(window):
    lfc = bandpass_filter(window, fs=FS, lowcut=10, highcut=40)
    hfc = bandpass_filter(window, fs=FS, lowcut=40, highcut=175)
    return np.sum(np.abs(hfc)) - np.sum(np.abs(lfc))

def calc_emd_mdf1_2(window):
    emd = EMD()
    imfs = emd(window)
    if len(imfs) < 2:
        return 0, 0
    def mdf(imf):
        freqs = np.fft.rfftfreq(len(imf), d=1/FS)
        psd = np.abs(np.fft.rfft(imf)) ** 2
        cumsum = np.cumsum(psd)
        idx = np.searchsorted(cumsum, cumsum[-1]/2)
        return freqs[idx] if idx < len(freqs) else 0
    return mdf(imfs[0]), mdf(imfs[1])

def calc_fluct_metrics(window):
    return np.ptp(window), np.mean(np.abs(np.diff(window))), np.var(window)

def extract_features(windows):
    features = []
    for w in windows:
        rms = np.mean(np.abs(w))
        mnf_arv = calc_mnf_arv_ratio(w)
        ima_diff = calc_ima_diff(w)
        mdf1, mdf2 = calc_emd_mdf1_2(w)
        rng, mean_diff, var = calc_fluct_metrics(w)
        features.append([rms, mnf_arv, ima_diff, mdf1, mdf2, rng, var, mean_diff])
    return np.array(features)

def invert_decreasing_features(features):
    features[:, [1, 3, 4]] *= -1
    return features

def apply_pca(features):
    scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=1).fit_transform(scaled).flatten()

def lowpass_filter(signal):
    nyq = 0.5 * DOWNSAMPLED_FS
    b, a = butter(LOWPASS_ORDER, LOWPASS_CUTOFF / nyq, btype='low')
    return filtfilt(b, a, signal)

# ===== NDI, NES, MOS Calculation =====
def compute_ndi_center(fi_filtered, indices, window_size, active_segments):
    centers = indices + window_size // 2
    mask = np.zeros(len(centers), dtype=bool)
    for i, c in enumerate(centers):
        for start, end in active_segments:
            if start <= c < end:
                mask[i] = True
                break
    pc1_active = fi_filtered[mask]
    if len(pc1_active) == 0:
        return np.nan
    p1 = pc1_active[0]
    ndi = np.mean(np.abs((pc1_active - p1) / (p1 + 1e-8)))
    return ndi

def compute_tactive(norm_rms, fs, active_segments):
    total_samples = 0
    for start, end in active_segments:
        total_samples += (end - start)
    tactive = total_samples / fs
    return tactive

def compute_nes(tactive, ndi):
    return tactive / (ndi + 1e-8)

def compute_mos(rectified, rms, tactive, fs, active_segments):
    # Concatenate all active samples
    active_rect = np.concatenate([rectified[start:end] for start, end in active_segments]) if active_segments else np.array([])
    active_rms = np.concatenate([rms[start:end] for start, end in active_segments]) if active_segments else np.array([])
    if len(active_rect) == 0 or len(active_rms) == 0 or tactive <= 0:
        return np.nan
    auc = np.trapezoid(active_rect, dx=1/fs)
    peak = np.max(active_rms)
    mos = (auc * peak) / tactive
    return mos

def find_active_segments(norm_rms, threshold=THRESHOLD, min_gap_sec=1.0, min_duration_sec=0.5):
    fs = FS
    above_thr = norm_rms > threshold
    starts = np.where(np.diff(above_thr.astype(int)) == 1)[0] + 1
    ends = np.where(np.diff(above_thr.astype(int)) == -1)[0] + 1
    if above_thr[0]:
        starts = np.insert(starts, 0, 0)
    if above_thr[-1]:
        ends = np.append(ends, len(norm_rms))
    merged = []
    min_gap = int(min_gap_sec * fs)
    min_duration = int(min_duration_sec * fs)
    i = 0
    while i < len(starts):
        seg_start = starts[i]
        seg_end = ends[i]
        while i + 1 < len(starts) and (starts[i + 1] - seg_end) < min_gap:
            seg_end = ends[i + 1]
            i += 1
        if (seg_end - seg_start) >= min_duration:
            merged.append((seg_start, seg_end))
        i += 1
    return merged

# ===== Integrated Pipeline =====
def fatigue_pipeline_with_ndi_nes_mos(df, mvc_raw, mvc_rectify, mvc_rms, fs=500):
    n_remove = int(2 * fs)
    if len(df) > n_remove:
        df = df.iloc[:-n_remove]

    raw = df['raw'].values
    rectify = df['rectify'].values
    rms = df['RMS'].values
    time = df['time'].values

    # Per-person normalization for features/metrics
    norm_raw = raw / mvc_raw
    norm_rectify = rectify / mvc_rectify
    norm_rms = rms / mvc_rms

    # Per-set normalization for active segment detection
    set_mvc_rms = np.max(rms)
    norm_rms_set = rms / set_mvc_rms

    windows, indices = window_signal(norm_raw)
    rect_windows, _ = window_signal(norm_rectify)
    rms_windows, _ = window_signal(norm_rms)

    features = []
    for w_raw, w_rect, w_rms in zip(windows, rect_windows, rms_windows):
        mnf_arv = calc_mnf_arv_ratio(w_raw)
        ima_diff = calc_ima_diff(w_raw)
        mdf1, mdf2 = calc_emd_mdf1_2(w_raw)
        rms_val = np.mean(w_rms)
        rng, mean_diff, var = calc_fluct_metrics(w_rect)
        features.append([rms_val, mnf_arv, ima_diff, mdf1, mdf2, rng, var, mean_diff])
    features = np.array(features)
    features = invert_decreasing_features(features)
    fatigue_index = apply_pca(features)
    fi_filtered = lowpass_filter(fatigue_index)
    time_fi = time[indices + WINDOW_SIZE // 2]

    # Use per-set normalized RMS for active segment detection
    smoothed_rms_set = smooth_signal(norm_rms_set, int(0.5 * FS))
    active_segments = find_active_segments(smoothed_rms_set, threshold=THRESHOLD, min_gap_sec=1.0, min_duration_sec=0.5)

    tactive = compute_tactive(norm_rms, fs, active_segments)
    ndi = compute_ndi_center(fi_filtered, indices, WINDOW_SIZE, active_segments)
    nes = compute_nes(tactive, ndi)
    mos = compute_mos(norm_rectify, norm_rms, tactive, fs, active_segments)

    return (time, norm_raw, norm_rectify, norm_rms,
            time_fi, features, fatigue_index, fi_filtered,
            ndi, tactive, nes, mos, active_segments)

# ===== Plotting =====
def plot_signals_compare(time, raw, rectified, rms, active_segments=None, boundary_idx=None, title='sEMG Signals'):
    fig = plt.figure(figsize=(14, 5))
    plt.plot(time, raw, label='Raw')
    plt.plot(time, rectified, label='Rectified')
    plt.plot(time, rms, label='RMS')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    if active_segments is not None:
        for start, end in active_segments:
            plt.axvline(time[start], color='red', linestyle='--', alpha=0.8)
            plt.axvline(time[end-1], color='green', linestyle='--', alpha=0.8)
    if boundary_idx is not None:
        plt.axvline(time[boundary_idx], color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.tight_layout()
    return fig

def plot_fatigue_index_compare(time_fi, fatigue_index, fi_filtered, boundary_idx=None, title='sEMG Fatigue Index (PCA)'):
    fig = plt.figure(figsize=(12, 5))
    plt.plot(time_fi, fatigue_index, label='Fatigue Index (PCA)')
    plt.plot(time_fi, fi_filtered, label='Fatigue Index (PCA, filtered)', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Fatigue Index')
    plt.title(title)
    plt.legend()
    if boundary_idx is not None:
        plt.axvline(time_fi[np.searchsorted(time_fi, boundary_idx)], color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.tight_layout()
    return fig

def print_metrics(label, ndi, tactive, nes, mos):
    print(f"{label}:")
    print(f"  NDI: {ndi:.3f}")
    print(f"  Tactive: {tactive:.2f} s")
    print(f"  NES: {nes:.3f}")
    print(f"  MOS: {mos:.3f}")

# ===== Main Execution =====
def main():
    file_paths = select_up_to_three_files()
    if len(file_paths) < 2:
        print("Please select at least two files (max three).")
        return

    # Load all sets
    dfs = [pd.read_excel(fp) for fp in file_paths]

    # Find global MVCs across all sets for per-person normalization
    all_raw = np.concatenate([df['raw'].values for df in dfs])
    all_rectify = np.concatenate([df['rectify'].values for df in dfs])
    all_rms = np.concatenate([df['RMS'].values for df in dfs])
    mvc_raw = np.max(all_raw)
    mvc_rectify = np.max(all_rectify)
    mvc_rms = np.max(all_rms)

    # Process each set with per-person normalization
    results = [fatigue_pipeline_with_ndi_nes_mos(df, mvc_raw, mvc_rectify, mvc_rms) for df in dfs]

    # Concatenate signals and time, adjusting time for each set
    time_merged = []
    norm_raw_merged = []
    norm_rect_merged = []
    norm_rms_merged = []
    active_segments_merged = []
    boundary_idxs = []

    time_offset = 0
    for i, res in enumerate(results):
        time, norm_raw, norm_rect, norm_rms, *_ = res
        if i == 0:
            time_adj = time
        else:
            dt = time[1] - time[0]
            time_adj = time + time_offset + dt
        # Store boundary index (for black dashed line)
        if i > 0:
            boundary_idxs.append(len(time_merged))
        # Merge signals
        time_merged.extend(time_adj)
        norm_raw_merged.extend(norm_raw)
        norm_rect_merged.extend(norm_rect)
        norm_rms_merged.extend(norm_rms)
        # Adjust active segment indices for merged signal
        segs = [(start + len(time_merged) - len(time_adj), end + len(time_merged) - len(time_adj)) for start, end in res[-1]]
        active_segments_merged.extend(segs)
        time_offset = time_adj[-1]
    
    time_merged = np.array(time_merged)
    norm_raw_merged = np.array(norm_raw_merged)
    norm_rect_merged = np.array(norm_rect_merged)
    norm_rms_merged = np.array(norm_rms_merged)

    # Plot signals with black dashed lines at each set boundary
    fig = plt.figure(figsize=(14, 5))
    plt.plot(time_merged, norm_raw_merged, label='Raw')
    plt.plot(time_merged, norm_rect_merged, label='Rectified')
    plt.plot(time_merged, norm_rms_merged, label='RMS')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Figure 1: sEMG Signals (Merged Sets)')
    plt.legend()
    for start, end in active_segments_merged:
        plt.axvline(time_merged[start], color='red', linestyle='--', alpha=0.8)
        plt.axvline(time_merged[end-1], color='green', linestyle='--', alpha=0.8)
    for idx in boundary_idxs:
        plt.axvline(time_merged[idx], color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.tight_layout()

    # Print metrics for each set
    for i, res in enumerate(results):
        print_metrics(f"Set {i+1}", res[8], res[9], res[10], res[11])

    # Calculate NES percent change
    nes_list = [res[10] for res in results]
    nes_percent_change = ["--- (baseline)"]
    for i in range(1, len(nes_list)):
        prev = nes_list[i-1]
        curr = nes_list[i]
        percent = ((curr - prev) / prev) * 100 if prev != 0 else float('nan')
        nes_percent_change.append(f"{percent:+.1f}%")
    
    # Print NES percent change
    print("\nNES Percent Change between Sets:")
    for i, change in enumerate(nes_percent_change):
        print(f"  Set {i}: {change}")

    plt.show()

if __name__ == "__main__":
    main()