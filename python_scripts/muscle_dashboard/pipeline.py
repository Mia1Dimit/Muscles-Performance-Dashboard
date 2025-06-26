import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PyEMD import EMD

FS = 500
WINDOW_SIZE = 500
STEP_SIZE = 125
LOWPASS_CUTOFF = 0.08
LOWPASS_ORDER = 2
DOWNSAMPLED_FS = FS // STEP_SIZE
THRESHOLD = 0.05

def smooth_signal(signal, window_len):
    print(f"[DEBUG] Smoothing signal with window length {window_len}")
    window = np.ones(window_len) / window_len
    return np.convolve(signal, window, mode='same')

def window_signal(signal):
    n = len(signal)
    windows = []
    indices = []
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        windows.append(signal[start:end])
        indices.append(start)
    return np.array(windows), np.array(indices)

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

def invert_decreasing_features(features):
    features[:, [1, 3, 4]] *= -1
    return features

def apply_pca(features):
    print("[DEBUG] Applying PCA to features...")
    scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=1).fit_transform(scaled).flatten()

def lowpass_filter(signal):
    nyq = 0.5 * DOWNSAMPLED_FS
    b, a = butter(LOWPASS_ORDER, LOWPASS_CUTOFF / nyq, btype='low')
    return filtfilt(b, a, signal)

def find_active_segments(norm_rms, threshold=THRESHOLD, min_gap_sec=1.0, min_duration_sec=0.5):
    print(f"[DEBUG] Finding active segments with threshold {threshold}")
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
    print(f"[DEBUG] Found {len(merged)} active segments.")
    return merged

def fatigue_pipeline(df, mvc_raw, mvc_rectify, mvc_rms, set_mvc_rms):
    print("[INFO] Running fatigue pipeline for a set...")
    raw = df['raw'].values
    rectify = df['rectify'].values
    rms = df['RMS'].values
    time = df['time'].values

    # Per-person MVC normalization for features
    print(f"[DEBUG] Windowing signal: window={WINDOW_SIZE}, step={STEP_SIZE}")
    norm_raw = raw / mvc_raw
    norm_rectify = rectify / mvc_rectify
    norm_rms = rms / mvc_rms

    # Per-set MVC normalization for active segment detection
    norm_rms_set = rms / set_mvc_rms

    # Feature extraction (per-person normalized)
    print(f"[DEBUG] Windowing signal: window={WINDOW_SIZE}, step={STEP_SIZE}")
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

    # Active segment detection (per-set normalized RMS)
    smoothed_rms = smooth_signal(norm_rms_set, int(0.5 * FS))
    active_segments = find_active_segments(smoothed_rms)

    print("[INFO] Fatigue pipeline complete.")
    return {
        "time": time,
        "raw": raw,
        "rectify": rectify,
        "rms": rms,
        "norm_raw": norm_raw,
        "norm_rectify": norm_rectify,
        "norm_rms": norm_rms,
        "time_fi": time_fi,
        "features": features,
        "fatigue_index": fatigue_index,
        "fi_filtered": fi_filtered,
        "active_segments": active_segments
    }