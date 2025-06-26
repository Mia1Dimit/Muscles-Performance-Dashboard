import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from muscle_dashboard.io import select_up_to_three_files, load_and_trim_files, compute_mvc
from muscle_dashboard.pipeline import fatigue_pipeline, window_signal, FS, WINDOW_SIZE
from muscle_dashboard.metrics import compute_ndi_center, compute_tactive, compute_nes, compute_mos
import numpy as np

def main():
    print("[INFO] Starting pipeline...")
    file_paths = select_up_to_three_files()
    if not file_paths or len(file_paths) < 1:
        print("[ERROR] No files selected.")
        return

    # Load and trim files
    dfs, trimmed_raw, trimmed_rectify, trimmed_rms = load_and_trim_files(file_paths)
    if len(dfs) < 1:
        print("[ERROR] No valid files loaded.")
        return

    # Compute per-person MVCs (across all sets)
    mvc_raw, mvc_rectify, mvc_rms = compute_mvc(trimmed_raw, trimmed_rectify, trimmed_rms)
    if None in (mvc_raw, mvc_rectify, mvc_rms):
        print("[ERROR] MVC computation failed. Exiting.")
        return

    print("\n[RESULT] Metrics for each set:")
    nes_list = []
    for i, df in enumerate(dfs):
        print(f"\n[INFO] Processing set {i+1}...")
        raw = df['raw'].values
        rectify = df['rectify'].values
        rms = df['RMS'].values
        time = df['time'].values

        # 1. Normalize for features/metrics (global MVC)
        norm_raw = raw / mvc_raw
        norm_rectify = rectify / mvc_rectify
        norm_rms = rms / mvc_rms

        # 2. Normalize for active segment detection (per-set MVC)
        set_mvc_rms = np.max(rms)
        norm_rms_set = rms / set_mvc_rms
        from muscle_dashboard.pipeline import smooth_signal, find_active_segments, calc_mnf_arv_ratio, calc_ima_diff, calc_emd_mdf1_2, calc_fluct_metrics, invert_decreasing_features, apply_pca, lowpass_filter, window_signal, FS, WINDOW_SIZE
        smoothed_rms_set = smooth_signal(norm_rms_set, int(0.5 * FS))
        active_segments = find_active_segments(smoothed_rms_set)

        # 3. Windowing
        windows, indices = window_signal(norm_raw)
        rect_windows, _ = window_signal(norm_rectify)
        rms_windows, _ = window_signal(norm_rms)

        # 4. Feature extraction, PCA, filtering
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

        # 5. NDI calculation (use indices, not time)
        ndi = compute_ndi_center(fi_filtered, indices, WINDOW_SIZE, active_segments)
        tactive = compute_tactive(norm_rms, FS, active_segments)
        nes = compute_nes(tactive, ndi)
        mos = compute_mos(norm_rectify, norm_rms, tactive, FS, active_segments)
        print(f"Set {i+1}:")
        print(f"  NDI: {ndi:.3f}")
        print(f"  Tactive: {tactive:.2f} s")
        print(f"  NES: {nes:.3f}")
        print(f"  MOS: {mos:.3f}")
        nes_list.append(nes)

    # NES percent change between sets
    nes_percent_change = ["--- (baseline)"]
    for i in range(1, len(nes_list)):
        prev = nes_list[i-1]
        curr = nes_list[i]
        percent = ((curr - prev) / prev) * 100 if prev != 0 else float('nan')
        nes_percent_change.append(f"{percent:+.1f}%")
    print("\nNES Percent Change between Sets:")
    for i, change in enumerate(nes_percent_change):
        print(f"  Set {i}: {change}")

if __name__ == "__main__":
    main()