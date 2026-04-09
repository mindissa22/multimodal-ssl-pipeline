"""
preprocess_har.py
=================
Preprocesses PAMAP2 and WISDM into a single harmonised HAR dataset.

Outputs
-------
data/processed/har/
    har_pretrain.npy         -- [N, 6, 200] float32, no labels (10s windows, no overlap)
    har_supervised.npy       -- [N, 6, 100] float32, with labels (5s windows, 50% overlap)
    har_pretrain_metadata.csv
    har_supervised_metadata.csv
submission_sample/har/
    100 windows from each output
"""

import os
import glob
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocess_har.log")
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM   = PROJECT_ROOT / "data" / "interim" / "har"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "har"
SAMPLE_DIR     = PROJECT_ROOT / "submission_sample" / "har"

for d in [DATA_INTERIM, DATA_PROCESSED, SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_HZ         = 20
CHANNELS          = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
PRETRAIN_WINDOW   = 10 * TARGET_HZ        # 200 samples
SUPERVISED_WINDOW = 5  * TARGET_HZ        # 100 samples
SUPERVISED_STRIDE = int(SUPERVISED_WINDOW * 0.5)  # 50 samples

SHARED_LABELS = {
    "walking", "jogging", "stairs", "sitting",
    "standing", "cycling", "rope_jumping"
}

PAMAP2_LABEL_MAP = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "jogging", 6: "cycling", 7: "nordic_walking",
    12: "stairs", 13: "stairs", 24: "rope_jumping",
    9: "watching_tv", 10: "computer_work", 11: "car_driving",
    16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry",
    19: "house_cleaning", 20: "playing_soccer",
}

WISDM_LABEL_MAP = {
    "A": "walking", "B": "jogging", "C": "stairs", "D": "cycling",
    "E": "stairs", "F": "typing", "G": "teeth_brushing",
    "H": "soup_eating", "I": "chips_eating", "J": "rope_jumping",
    "K": "kicking", "L": "sitting", "M": "standing",
    "O": "clapping", "P": "folding_clothes", "Q": "hair_brushing",
    "R": "walking_dog", "S": "kicking",
}

# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def butter_lowpass(cutoff_hz, fs, order=4):
    nyq = fs / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sos

def downsample_signal(data, orig_hz, target_hz):
    factor = orig_hz // target_hz
    if factor == 1:
        return data
    sos = butter_lowpass(cutoff_hz=target_hz / 2.0, fs=orig_hz, order=4)
    filtered = sosfilt(sos, data, axis=0)
    return filtered[::factor]

def zscore_normalise(data):
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    std[std < 1e-8] = 1.0
    return (data - mean) / std

def forward_fill_nans(data, max_gap_samples=10):
    df = pd.DataFrame(data)
    filled = df.fillna(method="ffill", limit=max_gap_samples)
    return filled.values

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(signal, labels, window_size, stride,
                 subject_id, dataset, source_file, include_labels=True):
    windows  = []
    metadata = []
    n_samples = len(signal)
    label_list   = sorted(SHARED_LABELS)
    label_to_int = {l: i for i, l in enumerate(label_list)}

    for start in range(0, n_samples - window_size + 1, stride):
        end   = start + window_size
        chunk = signal[start:end]

        if np.isnan(chunk).any():
            continue

        chunk_t   = chunk.T.astype(np.float32)
        label_val = "unlabelled"

        if include_labels:
            chunk_labels = labels[start:end]
            valid = chunk_labels[chunk_labels >= 0]
            if len(valid) == 0:
                continue
            counts = pd.Series(valid).value_counts()
            if len(counts) >= 2 and counts.iloc[0] == counts.iloc[1]:
                continue
            label_val = counts.index[0]
            if label_val not in SHARED_LABELS:
                continue

        windows.append(chunk_t)
        metadata.append({
            "sample_id":             f"{dataset}_{subject_id}_{start}_{end}",
            "dataset_name":          dataset,
            "modality":              "HAR",
            "subject_or_patient_id": subject_id,
            "source_file_or_record": source_file,
            "split":                 "pretrain" if not include_labels else "supervised",
            "label_or_event":        label_val,
            "sampling_rate_hz":      TARGET_HZ,
            "n_channels":            6,
            "n_samples":             window_size,
            "channel_schema":        ",".join(CHANNELS),
            "qc_flags":              "",
        })

    return windows, metadata

# ---------------------------------------------------------------------------
# PAMAP2
# ---------------------------------------------------------------------------

def load_pamap2(raw_dir):
    dat_files = sorted(glob.glob(str(raw_dir / "**" / "Protocol" / "*.dat"), recursive=True))
    if not dat_files:
        dat_files = sorted(glob.glob(str(raw_dir / "**" / "*.dat"), recursive=True))
    if not dat_files:
        raise FileNotFoundError(f"No PAMAP2 .dat files found under {raw_dir}")

    log.info(f"PAMAP2: found {len(dat_files)} .dat files")
    all_dfs = []

    for fpath in dat_files:
        subject_id = Path(fpath).stem
        log.info(f"  Loading {subject_id}...")
        df = pd.read_csv(fpath, sep=" ", header=None, low_memory=False)
        extracted = df[[1, 4, 5, 6, 10, 11, 12]].copy()
        extracted.columns = ["activity_id", "acc_x", "acc_y", "acc_z",
                              "gyr_x", "gyr_y", "gyr_z"]
        extracted["subject_id"]  = subject_id
        extracted["source_file"] = Path(fpath).name

        # Drop class 0 (transient/other)
        n_before = len(extracted)
        extracted = extracted[extracted["activity_id"] != 0].copy()
        log.info(f"    Dropped {n_before - len(extracted)} class-0 rows")

        extracted["unified_label"] = extracted["activity_id"].map(PAMAP2_LABEL_MAP)
        all_dfs.append(extracted)

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(f"PAMAP2: {len(combined)} rows total")
    return combined


def preprocess_pamap2(df):
    pretrain_windows, pretrain_meta     = [], []
    supervised_windows, supervised_meta = [], []
    label_list   = sorted(SHARED_LABELS)
    label_to_int = {l: i for i, l in enumerate(label_list)}

    for subject_id, group in df.groupby("subject_id"):
        log.info(f"  PAMAP2 subject {subject_id}...")
        signal_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        signal = group[signal_cols].values.astype(float)
        signal = forward_fill_nans(signal, max_gap_samples=10)
        signal = downsample_signal(signal, orig_hz=100, target_hz=20)
        signal = zscore_normalise(signal)

        labels_raw  = group["unified_label"].values
        labels_20hz = labels_raw[::5]
        labels_int  = np.array([
            label_to_int.get(l, -1) if pd.notna(l) else -1
            for l in labels_20hz
        ])

        source_file = group["source_file"].iloc[0]

        pw, pm = make_windows(signal, labels_int, PRETRAIN_WINDOW,
                               PRETRAIN_WINDOW, str(subject_id),
                               "pamap2", source_file, include_labels=False)
        pretrain_windows.extend(pw); pretrain_meta.extend(pm)

        sw, sm = make_windows(signal, labels_int, SUPERVISED_WINDOW,
                               SUPERVISED_STRIDE, str(subject_id),
                               "pamap2", source_file, include_labels=True)
        supervised_windows.extend(sw); supervised_meta.extend(sm)

    log.info(f"PAMAP2: {len(pretrain_windows)} pretrain, {len(supervised_windows)} supervised windows")
    return (pretrain_windows, pretrain_meta), (supervised_windows, supervised_meta)

# ---------------------------------------------------------------------------
# WISDM
# ---------------------------------------------------------------------------

def load_wisdm(raw_dir):
    accel_files = sorted(raw_dir.rglob("*accel_watch*"))
    gyro_files  = sorted(raw_dir.rglob("*gyro_watch*"))

    if not accel_files:
        raise FileNotFoundError(f"No WISDM watch accelerometer files found under {raw_dir}")

    log.info(f"WISDM: found {len(accel_files)} accel files")

    def parse_wisdm_file(fpath, sensor):
        rows = []
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip().rstrip(";")
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                try:
                    rows.append({
                        "subject_id":    parts[0].strip(),
                        "activity_code": parts[1].strip(),
                        "timestamp":     int(parts[2].strip()),
                        f"{sensor}_x":   float(parts[3].strip()),
                        f"{sensor}_y":   float(parts[4].strip()),
                        f"{sensor}_z":   float(parts[5].strip()),
                    })
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(rows)

    all_subjects = []
    for accel_file in accel_files:
        subject_id = accel_file.stem.split("_")[1]
        gyro_candidates = [g for g in gyro_files if f"_{subject_id}_" in g.name]
        if not gyro_candidates:
            log.warning(f"  No gyro file for subject {subject_id}, skipping.")
            continue

        accel_df = parse_wisdm_file(accel_file, "acc")
        gyro_df  = parse_wisdm_file(gyro_candidates[0], "gyr")

        if accel_df.empty or gyro_df.empty:
            continue

        accel_df = accel_df.sort_values("timestamp")
        gyro_df  = gyro_df.sort_values("timestamp")

        merged = pd.merge_asof(
            accel_df,
            gyro_df[["timestamp", "gyr_x", "gyr_y", "gyr_z"]],
            on="timestamp", tolerance=25_000_000, direction="nearest"
        )
        merged["source_file"]   = accel_file.name
        merged["unified_label"] = merged["activity_code"].map(WISDM_LABEL_MAP)
        all_subjects.append(merged)

    combined = pd.concat(all_subjects, ignore_index=True)
    log.info(f"WISDM: {len(combined)} rows total")
    return combined


def preprocess_wisdm(df):
    pretrain_windows, pretrain_meta     = [], []
    supervised_windows, supervised_meta = [], []
    label_list   = sorted(SHARED_LABELS)
    label_to_int = {l: i for i, l in enumerate(label_list)}

    for subject_id, group in df.groupby("subject_id"):
        log.info(f"  WISDM subject {subject_id}...")
        signal_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        group = group.dropna(subset=signal_cols)
        if len(group) < PRETRAIN_WINDOW:
            log.warning(f"    Subject {subject_id} too few rows, skipping.")
            continue

        signal     = group[signal_cols].values.astype(float)
        signal     = zscore_normalise(signal)
        labels_int = np.array([
            label_to_int.get(l, -1) if pd.notna(l) else -1
            for l in group["unified_label"].values
        ])
        source_file = group["source_file"].iloc[0]

        pw, pm = make_windows(signal, labels_int, PRETRAIN_WINDOW,
                               PRETRAIN_WINDOW, str(subject_id),
                               "wisdm", source_file, include_labels=False)
        pretrain_windows.extend(pw); pretrain_meta.extend(pm)

        sw, sm = make_windows(signal, labels_int, SUPERVISED_WINDOW,
                               SUPERVISED_STRIDE, str(subject_id),
                               "wisdm", source_file, include_labels=True)
        supervised_windows.extend(sw); supervised_meta.extend(sm)

    log.info(f"WISDM: {len(pretrain_windows)} pretrain, {len(supervised_windows)} supervised windows")
    return (pretrain_windows, pretrain_meta), (supervised_windows, supervised_meta)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(pretrain_windows, pretrain_meta,
                 supervised_windows, supervised_meta):
    log.info("Saving outputs...")

    X_pretrain   = np.stack(pretrain_windows,   axis=0).astype(np.float32)
    X_supervised = np.stack(supervised_windows, axis=0).astype(np.float32)
    meta_pre     = pd.DataFrame(pretrain_meta)
    meta_sup     = pd.DataFrame(supervised_meta)

    np.save(DATA_PROCESSED / "har_pretrain.npy",   X_pretrain)
    np.save(DATA_PROCESSED / "har_supervised.npy", X_supervised)
    meta_pre.to_csv(DATA_PROCESSED / "har_pretrain_metadata.csv",   index=False)
    meta_sup.to_csv(DATA_PROCESSED / "har_supervised_metadata.csv", index=False)

    np.save(SAMPLE_DIR / "har_pretrain_sample.npy",   X_pretrain[:100])
    np.save(SAMPLE_DIR / "har_supervised_sample.npy", X_supervised[:100])
    meta_pre.head(100).to_csv(SAMPLE_DIR / "har_pretrain_metadata_sample.csv",   index=False)
    meta_sup.head(100).to_csv(SAMPLE_DIR / "har_supervised_metadata_sample.csv", index=False)

    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "har_pretrain": {
            "file": "data/processed/har/har_pretrain.npy",
            "shape": list(X_pretrain.shape),
            "dtype": "float32",
            "n_windows": len(X_pretrain),
            "window_size_s": 10,
            "overlap": 0.0,
        },
        "har_supervised": {
            "file": "data/processed/har/har_supervised.npy",
            "shape": list(X_supervised.shape),
            "dtype": "float32",
            "n_windows": len(X_supervised),
            "window_size_s": 5,
            "overlap": 0.5,
            "label_counts": meta_sup["label_or_event"].value_counts().to_dict(),
        },
    }

    with open(DATA_PROCESSED / "har_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Pretrain shape:   {X_pretrain.shape}")
    log.info(f"Supervised shape: {X_supervised.shape}")
    log.info("Done.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("HAR Preprocessing: PAMAP2 + WISDM")
    log.info("=" * 60)

    log.info("\n--- PAMAP2 ---")
    pamap2_df = load_pamap2(DATA_RAW / "pamap2")
    (p2_pre_w, p2_pre_m), (p2_sup_w, p2_sup_m) = preprocess_pamap2(pamap2_df)

    log.info("\n--- WISDM ---")
    wisdm_df = load_wisdm(DATA_RAW / "wisdm")
    (w_pre_w, w_pre_m), (w_sup_w, w_sup_m) = preprocess_wisdm(wisdm_df)

    log.info("\n--- Combining ---")
    all_pre_w = p2_pre_w + w_pre_w
    all_pre_m = p2_pre_m + w_pre_m
    all_sup_w = p2_sup_w + w_sup_w
    all_sup_m = p2_sup_m + w_sup_m

    rng     = np.random.default_rng(seed=42)
    pre_idx = rng.permutation(len(all_pre_w))
    sup_idx = rng.permutation(len(all_sup_w))

    all_pre_w = [all_pre_w[i] for i in pre_idx]
    all_pre_m = [all_pre_m[i] for i in pre_idx]
    all_sup_w = [all_sup_w[i] for i in sup_idx]
    all_sup_m = [all_sup_m[i] for i in sup_idx]

    save_outputs(all_pre_w, all_pre_m, all_sup_w, all_sup_m)
    log.info("HAR preprocessing complete.")

if __name__ == "__main__":
    main()