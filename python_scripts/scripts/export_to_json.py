import os
import json
from datetime import datetime

EXPORT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'json_exports')
os.makedirs(EXPORT_DIR, exist_ok=True)

def prompt_metadata(num_sets):
    athlete_id = input("Enter athlete ID (e.g. A001): ").strip()
    same_day = input("Are all sets on the same day? (y/n): ").strip().lower()
    dates = []
    if same_day == "y":
        date = input("Enter session date (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD.")
            exit(1)
        dates = [date] * num_sets
    else:
        for i in range(num_sets):
            date = input(f"Enter date for set {i+1} (YYYY-MM-DD): ").strip()
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                print("Invalid date format. Use YYYY-MM-DD.")
                exit(1)
            dates.append(date)
    return athlete_id, dates

def get_next_session_id(athlete_id):
    files = [f for f in os.listdir(EXPORT_DIR) if f.startswith(f"athlete_{athlete_id}_session_")]
    session_numbers = []
    for f in files:
        try:
            session_str = f.split("_session_")[1].split("_")[0]
            session_numbers.append(int(session_str[1:]))
        except Exception:
            continue
    next_num = max(session_numbers, default=0) + 1
    return f"S{next_num:03d}"

def export_json(athlete_id, session_id, dates, pipeline_results):
    all_sets = []
    for i, res in enumerate(pipeline_results):
        set_json = {
            "athlete_id": athlete_id,
            "session_id": session_id,
            "set_number": i+1,
            "date": dates[i],
            "time": res["time"].tolist(),
            "raw": res["raw"].tolist(),
            "rectify": res["rectify"].tolist(),
            "rms": res["rms"].tolist(),
            "rms_val": res["rms_val"].tolist(),
            "mnf_arv_ratio": res["mnf_arv_ratio"].tolist(),
            "ima_diff": res["ima_diff"].tolist(),
            "emd_mdf1": res["emd_mdf1"].tolist(),
            "emd_mdf2": res["emd_mdf2"].tolist(),
            "fluctuation_range": res["fluctuation_range"].tolist(),
            "fluctuation_var": res["fluctuation_var"].tolist(),
            "fluctuation_mean_diff": res["fluctuation_mean_diff"].tolist(),
            "pca": res["pca"].tolist(),
            "pca_smoothed": res["pca_smoothed"].tolist(),
            "ndi": res["ndi"],
            "nes": res["nes"],
            "mos": res["mos"],
            "tactive": res["tactive"],
            "auc_pca": res["auc_pca"],
            "auc_rms": res["auc_rms"],
            "auc_ratio": res["auc_ratio"]
        }
        all_sets.append(set_json)
    out_path = os.path.join(EXPORT_DIR, f"{athlete_id}_{session_id}_{dates[0]}.json")
    with open(out_path, "w") as f:
        json.dump(all_sets, f, indent=2)
    print(f"[INFO] Exported JSON to {out_path}")

if __name__ == "__main__":
    # TODO: Replace this with actual pipeline results
    pipeline_results = []  # Fill with your per-set dicts from run_pipeline
    num_sets = len(pipeline_results) if pipeline_results else int(input("How many sets? "))
    athlete_id, dates = prompt_metadata(num_sets)
    session_id = get_next_session_id(athlete_id)
    export_json(athlete_id, session_id, dates, pipeline_results)