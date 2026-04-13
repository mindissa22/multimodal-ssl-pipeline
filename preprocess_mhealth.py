"""
preprocess_mhealth.py

Preprocesses mHealth harmonised to the
same 20 Hz, 6-channel schema as PAMAP2 and WISDM.

Channel selection: right wrist IMU (columns 15-20)
    acc_x, acc_y, acc_z (cols 15-17)
    gyr_x, gyr_y, gyr_z (cols 18-20)

Sampling rate: 50 Hz → 20 Hz via resample_poly(2, 5)

Label 0 (null class): dropped as null class, consistent with PAMAP2 class 0 handling.

Label mapping to unified 7-class schema:
    L1 (standing still) → standing
    L2 (sitting)        → sitting
    L4 (walking)        → walking
    L5 (stairs)         → stairs
    L9 (cycling)        → cycling
    L10 (jogging)       → jogging
    All other labels excluded — not present in shared schema.

Outputs are appended to existing HAR outputs:
    data/processed/har/har_pretrain.npy
    data/processed/har/har_supervised.npy
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocess_mhealth.log")
    ]
)
log = logging.getLogger(__name__)

PROJECT_ROOT   = Path(__file__).parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw" / "mhealth" / "MHEALTHDATASET"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "har"
SAMPLE_DIR     = PROJECT_ROOT / "submission_sample" / "har"

TARGET_HZ         = 20
ORIG_HZ           = 50
CHANNELS          = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
PRETRAIN_WINDOW   = 10 * TARGET_HZ   # 200 samples
SUPERVISED_WINDOW = 5  * TARGET_HZ   # 100 samples
SUPERVISED_STRIDE = SUPERVISED_WINDOW // 2  # 50% overlap

SHARED_LABELS = {
    "walking", "jogging", "stairs", "sitting",
    "standing", "cycling", "rope_jumping"
}

MHEALTH_LABEL_MAP = {
    1:  "standing",
    2:  "sitting",
    4:  "walking",
    5:  "stairs",
    9:  "cycling",
    10: "jogging",
}

def zscore_normalise(data):
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    std[std < 1e-8] = 1.0
    return (data - mean) / std

def make_windows(signal, labels, window_size, stride,
                 subject_id, source_file, include_labels=True):
    windows  = []
    metadata = []
    label_list   = sorted(SHARED_LABELS)
    label_to_int = {l: i for i, l in enumerate(label_list)}

    for start in range(0, len(signal) - window_size + 1, stride):
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
            label_int = counts.index[0]
            label_val = label_list[label_int] if label_int >= 0 else "unlabelled"
            if label_val not in SHARED_LABELS:
                continue

        windows.append(chunk_t)
        metadata.append({
            "sample_id":             f"mhealth_{subject_id}_{start}_{end}",
            "dataset_name":          "mhealth",
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

def main():
    log.info("=" * 60)
    log.info("mHealth Preprocessing (Bonus HAR Dataset)")
    log.info("=" * 60)

    log_files = sorted(DATA_RAW.glob("mHealth_subject*.log"))
    if not log_files:
        raise FileNotFoundError(f"No mHealth log files found under {DATA_RAW}")

    log.info(f"Found {len(log_files)} subject files.")

    all_pre_w, all_pre_m = [], []
    all_sup_w, all_sup_m = [], []
    label_list   = sorted(SHARED_LABELS)
    label_to_int = {l: i for i, l in enumerate(label_list)}

    for fpath in log_files:
        subject_id = fpath.stem.replace("mHealth_subject", "mhealth_")
        log.info(f"  Processing {subject_id}...")

        df = pd.read_csv(fpath, header=None, sep=r'\s+')

        # Extract right wrist IMU (columns 14-19, 0-indexed = cols 15-20)
        signal = df.iloc[:, 14:20].values.astype(float)

        # Extract labels (column 23, 0-indexed = col 24)
        labels_raw = df.iloc[:, 23].values

        # Drop label 0 (null class) — rows with no valid activity
        valid_mask = labels_raw != 0
        signal     = signal[valid_mask]
        labels_raw = labels_raw[valid_mask]
        log.info(f"    Dropped {(~valid_mask).sum()} label-0 rows")

        # Resample from 50 Hz to 20 Hz using resample_poly(2, 5)
        signal = resample_poly(signal, 2, 5, axis=0)

        # Resample labels by taking every 2.5th sample (nearest neighbour)
        orig_len   = len(labels_raw)
        new_len    = len(signal)
        idx        = np.round(np.linspace(0, orig_len - 1, new_len)).astype(int)
        labels_raw = labels_raw[idx]

        # Map to unified labels
        labels_unified = np.array([
            label_to_int.get(MHEALTH_LABEL_MAP.get(int(l), ""), -1)
            for l in labels_raw
        ])

        # Per-subject z-score normalisation
        signal = zscore_normalise(signal)

        # Make windows
        pw, pm = make_windows(signal, labels_unified, PRETRAIN_WINDOW,
                               PRETRAIN_WINDOW, subject_id, fpath.name,
                               include_labels=False)
        sw, sm = make_windows(signal, labels_unified, SUPERVISED_WINDOW,
                               SUPERVISED_STRIDE, subject_id, fpath.name,
                               include_labels=True)

        all_pre_w.extend(pw); all_pre_m.extend(pm)
        all_sup_w.extend(sw); all_sup_m.extend(sm)
        log.info(f"    {len(pw)} pretrain, {len(sw)} supervised windows")

    log.info(f"mHealth total: {len(all_pre_w)} pretrain, {len(all_sup_w)} supervised windows")

    # Load existing HAR outputs
    log.info("Loading existing HAR outputs...")
    X_pre_existing  = np.load(DATA_PROCESSED / "har_pretrain.npy")
    X_sup_existing  = np.load(DATA_PROCESSED / "har_supervised.npy")
    meta_pre_existing = pd.read_csv(DATA_PROCESSED / "har_pretrain_metadata.csv")
    meta_sup_existing = pd.read_csv(DATA_PROCESSED / "har_supervised_metadata.csv")

    # Stack mHealth windows
    X_pre_new  = np.stack(all_pre_w, axis=0).astype(np.float32)
    X_sup_new  = np.stack(all_sup_w, axis=0).astype(np.float32)
    meta_pre_new = pd.DataFrame(all_pre_m)
    meta_sup_new = pd.DataFrame(all_sup_m)

    # Combine
    X_pre  = np.concatenate([X_pre_existing, X_pre_new], axis=0)
    X_sup  = np.concatenate([X_sup_existing, X_sup_new], axis=0)
    meta_pre = pd.concat([meta_pre_existing, meta_pre_new], ignore_index=True)
    meta_sup = pd.concat([meta_sup_existing, meta_sup_new], ignore_index=True)

    log.info(f"Combined pretrain shape:   {X_pre.shape}")
    log.info(f"Combined supervised shape: {X_sup.shape}")

    # Save updated outputs
    np.save(DATA_PROCESSED / "har_pretrain.npy",   X_pre)
    np.save(DATA_PROCESSED / "har_supervised.npy", X_sup)
    meta_pre.to_csv(DATA_PROCESSED / "har_pretrain_metadata.csv",   index=False)
    meta_sup.to_csv(DATA_PROCESSED / "har_supervised_metadata.csv", index=False)

    # Update sample packs
    np.save(SAMPLE_DIR / "har_pretrain_sample.npy",   X_pre[:100])
    np.save(SAMPLE_DIR / "har_supervised_sample.npy", X_sup[:100])
    meta_pre.head(100).to_csv(SAMPLE_DIR / "har_pretrain_metadata_sample.csv",   index=False)
    meta_sup.head(100).to_csv(SAMPLE_DIR / "har_supervised_metadata_sample.csv", index=False)

    # Update manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "har_pretrain": {
            "file":          "data/processed/har/har_pretrain.npy",
            "shape":         list(X_pre.shape),
            "dtype":         "float32",
            "n_windows":     len(X_pre),
            "window_size_s": 10,
            "overlap":       0.0,
            "datasets":      ["pamap2", "wisdm", "mhealth"],
        },
        "har_supervised": {
            "file":          "data/processed/har/har_supervised.npy",
            "shape":         list(X_sup.shape),
            "dtype":         "float32",
            "n_windows":     len(X_sup),
            "window_size_s": 5,
            "overlap":       0.5,
            "datasets":      ["pamap2", "wisdm", "mhealth"],
            "label_counts":  meta_sup["label_or_event"].value_counts().to_dict(),
        },
    }

    with open(DATA_PROCESSED / "har_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("mHealth preprocessing complete!")
    log.info(f"HAR now includes PAMAP2 + WISDM + mHealth")

if __name__ == "__main__":
    main()