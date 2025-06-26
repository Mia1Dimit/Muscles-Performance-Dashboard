import numpy as np

def compute_ndi_center(fi_filtered, indices, window_size, active_segments):
    print("[DEBUG] Computing NDI...")
    centers = indices + window_size // 2
    mask = np.zeros(len(centers), dtype=bool)
    for i, c in enumerate(centers):
        for start, end in active_segments:
            if start <= c < end:
                mask[i] = True
                break
    pc1_active = fi_filtered[mask]
    if len(pc1_active) == 0:
        print("[WARNING] No active PCA points for NDI.")
        return np.nan
    p1 = pc1_active[0]
    ndi = np.mean(np.abs((pc1_active - p1) / (p1 + 1e-8)))
    print(f"[DEBUG] NDI: {ndi}")
    return ndi

def compute_tactive(norm_rms, fs, active_segments):
    print("[DEBUG] Computing tactive...")
    total_samples = 0
    for start, end in active_segments:
        total_samples += (end - start)
    tactive = total_samples / fs
    print(f"[DEBUG] tactive: {tactive}")
    return tactive

def compute_nes(tactive, ndi):
    print("[DEBUG] Computing NES...")
    nes = tactive / (ndi + 1e-8)
    print(f"[DEBUG] NES: {nes}")
    return nes

def compute_mos(rectified, rms, tactive, fs, active_segments):
    print("[DEBUG] Computing MOS...")
    active_rect = np.concatenate([rectified[start:end] for start, end in active_segments]) if active_segments else np.array([])
    active_rms = np.concatenate([rms[start:end] for start, end in active_segments]) if active_segments else np.array([])
    if len(active_rect) == 0 or len(active_rms) == 0 or tactive <= 0:
        print("[WARNING] Not enough data for MOS.")
        return np.nan
    auc = np.trapezoid(active_rect, dx=1/fs)
    peak = np.max(active_rms)
    mos = (auc * peak) / tactive
    print(f"[DEBUG] MOS: {mos}")
    return mos

def compute_auc(y, x):
    auc = np.trapezoid(y, x)
    print(f"[DEBUG] AUC: {auc}")
    return auc