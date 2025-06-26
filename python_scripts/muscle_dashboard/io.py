import pandas as pd
from tkinter import Tk, filedialog

FS = 500  # Sampling frequency

def select_files():
    print("[INFO] Opening file dialog: Select Excel files...")
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select two or more Excel Files",
        filetypes=[("Excel files", "*.xlsx *.xls *.csv")]
    )
    selected = list(file_paths)
    if len(selected) < 1:
        print("[ERROR] Please select at least one file.")
        return []
    print(f"[INFO] Selected files: {selected}")
    return selected

def load_and_trim_files(file_paths):
    """Load Excel files, remove last 2 seconds from each, return list of DataFrames."""
    dfs = []
    trimmed_raw = []
    trimmed_rectify = []
    trimmed_rms = []
    for fp in file_paths:
        print(f"[INFO] Loading file: {fp}")
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            print(f"[ERROR] Failed to load {fp}: {e}")
            continue
        n_remove = int(2 * FS)
        if len(df) > n_remove:
            df = df.iloc[:-n_remove]
            print(f"[INFO] Trimmed last {n_remove} samples ({2} seconds) from {fp}")
        else:
            print(f"[WARNING] File {fp} is too short to trim 2 seconds.")
        dfs.append(df)
        try:
            trimmed_raw.append(df['raw'].values)
            trimmed_rectify.append(df['rectify'].values)
            trimmed_rms.append(df['RMS'].values)
        except KeyError as e:
            print(f"[ERROR] Missing expected column in {fp}: {e}")
            continue
    print(f"[INFO] Loaded and trimmed {len(dfs)} files.")
    return dfs, trimmed_raw, trimmed_rectify, trimmed_rms

def compute_mvc(trimmed_raw, trimmed_rectify, trimmed_rms):
    """Compute per-person MVCs (across all sets, after trimming)."""
    import numpy as np
    print("[INFO] Computing MVCs across all sets...")
    try:
        all_raw = np.concatenate(trimmed_raw)
        all_rectify = np.concatenate(trimmed_rectify)
        all_rms = np.concatenate(trimmed_rms)
        mvc_raw = np.max(all_raw)
        mvc_rectify = np.max(all_rectify)
        mvc_rms = np.max(all_rms)
        print(f"[INFO] MVCs computed: raw={mvc_raw}, rectify={mvc_rectify}, rms={mvc_rms}")
        return mvc_raw, mvc_rectify, mvc_rms
    except Exception as e:
        print(f"[ERROR] Failed to compute MVCs: {e}")
        return None, None, None