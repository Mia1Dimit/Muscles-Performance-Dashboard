import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from muscle_dashboard.io import select_files, load_and_trim_files, compute_mvc
from muscle_dashboard.pipeline import fatigue_pipeline, window_signal, FS, WINDOW_SIZE
from muscle_dashboard.metrics import compute_ndi_center, compute_tactive, compute_nes, compute_mos, compute_auc
from export_to_json import prompt_metadata, get_next_session_id, export_json
import numpy as np

def main():
    print("[INFO] Starting pipeline...")
    file_paths = select_files()
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
    auc_matrix = []
    # After your per-set loop, build the pipeline_results list:
    pipeline_results = []
    for i, df in enumerate(dfs):
        print(f"\n[INFO] Processing set {i+1}...")
        set_mvc_rms = np.max(df['RMS'].values)
        res = fatigue_pipeline(df, mvc_raw, mvc_rectify, mvc_rms, set_mvc_rms)

        ndi = compute_ndi_center(res["fi_filtered"], res["indices"], WINDOW_SIZE, res["active_segments"])
        tactive = compute_tactive(res["norm_rms"], FS, res["active_segments"])
        nes = compute_nes(tactive, ndi)
        mos = compute_mos(res["norm_rectify"], res["norm_rms"], tactive, FS, res["active_segments"])
        print(f"Set {i+1}:")
        print(f"  NDI: {ndi:.3f}")
        print(f"  Tactive: {tactive:.2f} s")
        print(f"  NES: {nes:.3f}")
        print(f"  MOS: {mos:.3f}")
        nes_list.append(nes)

        # Windowed RMS (normalized)
        rms_windows, _ = window_signal(res["norm_rms"])
        mean_rms_per_window = np.mean(rms_windows, axis=1)

        # Min-max normalize PCA fatigue index
        fi_min = res["fatigue_index"].min()
        fi_max = res["fatigue_index"].max()
        fatigue_index_norm = (res["fatigue_index"] - fi_min) / (fi_max - fi_min + 1e-8)

        # Mask for active segments
        mask = np.zeros_like(res["time_fi"], dtype=bool)
        for seg_start, seg_end in res["active_segments"]:
            seg_time_start = res["time"][seg_start]
            seg_time_end = res["time"][seg_end - 1]
            mask |= (res["time_fi"] >= seg_time_start) & (res["time_fi"] <= seg_time_end)

        if not np.any(mask):
            print(f"  No active segment detected for AUC calculation.")
            auc_pca = np.nan
            auc_rms = np.nan
            ratio = np.nan
        else:
            auc_pca = compute_auc(fatigue_index_norm[mask], res["time_fi"][mask])
            auc_rms = compute_auc(mean_rms_per_window[mask], res["time_fi"][mask])
            ratio = auc_pca / auc_rms if auc_rms != 0 else float('nan')

        print(f"  AUC(PCA): {auc_pca:.3f}")
        print(f"  AUC(RMS): {auc_rms:.3f}")
        print(f"  Ratio (PCA/RMS): {ratio:.3f}")
        auc_matrix.append([i+1, auc_pca, auc_rms, ratio])

        pipeline_results.append({
            "time": res["time"],
            "raw": res["raw"],
            "rectify": res["rectify"],
            "rms": res["rms"],
            "rms_val": res["rms_val"],
            "ima_diff": res["ima_diff"],
            "emd_mdf1": res["emd_mdf1"],
            "emd_mdf2": res["emd_mdf2"],
            "mnf_arv_ratio": res["mnf_arv_ratio"],
            "fluctuation_range": res["fluctuation_range"],
            "fluctuation_var": res["fluctuation_var"],
            "fluctuation_mean_diff": res["fluctuation_mean_diff"],
            "pca": res["fatigue_index"],
            "pca_smoothed": res["fi_filtered"],
            "ndi": ndi,
            "nes": nes,
            "mos": mos,
            "tactive": tactive,
            "auc_pca": auc_pca,
            "auc_rms": auc_rms,
            "auc_ratio": ratio
        })

    print("\n[RESULT] AUC Metrics Matrix (Set, AUC(PCA), AUC(RMS), Ratio):")
    for row in auc_matrix:
        print(row)

    # Prompt for metadata and export
    athlete_id, date = prompt_metadata()
    session_id = get_next_session_id(athlete_id)
    export_json(athlete_id, session_id, date, pipeline_results)

if __name__ == "__main__":
    main()