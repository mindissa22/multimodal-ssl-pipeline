"""
preprocess_eeg.py

Preprocesses EEGMMIDB into event aligned windows for motor imagery classification.

Outputs
-------
data/processed/eeg/
    eeg_windows.npy          -- [N, 64, 640] float32 (4s windows at 160 Hz)
    eeg_metadata.csv         -- one row per window
submission_sample/eeg/
    eeg_windows_sample.npy   -- 100 windows
    eeg_metadata_sample.csv  -- 100 rows

Design:
- Only runs 4, 8, 12 used (left vs right fist imagery)
- T1 = left fist, T2 = right fist, T0 (rest) excluded
- 4-second windows starting at T1/T2 onset = 640 samples
- Band-pass filter 1-40 Hz, notch at 60 Hz
- Common average reference applied
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocess_eeg.log")
    ]
)
log = logging.getLogger(__name__)

PROJECT_ROOT   = Path(__file__).parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw" / "eegmmidb"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "eeg"
SAMPLE_DIR     = PROJECT_ROOT / "submission_sample" / "eeg"

for d in [DATA_PROCESSED, SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SFREQ        = 160       # native sampling rate — retained
WINDOW_SEC   = 4         # 4 second windows
WINDOW_SAMP  = SFREQ * WINDOW_SEC   # 640 samples
MOTOR_RUNS   = [4, 8, 12]           # left vs right fist imagery runs
LABEL_MAP    = {"T1": "left_fist", "T2": "right_fist"}


def preprocess_subject(subject_dir: Path):
    """
    Load runs 4, 8, 12 for one subject.
    Extract T1 and T2 event-aligned 4-second windows.
    Apply band-pass, notch filter, and common average reference.
    """
    subject_id = subject_dir.name  # e.g. S001
    windows    = []
    metadata   = []

    for run_num in MOTOR_RUNS:
        edf_pattern = f"{subject_id}R{run_num:02d}.edf"
        edf_files   = list(subject_dir.glob(edf_pattern))

        if not edf_files:
            log.warning(f"  {subject_id} run {run_num}: file not found, skipping.")
            continue

        edf_path = edf_files[0]

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            log.warning(f"  {subject_id} run {run_num}: failed to load — {e}")
            continue

        # Band-pass filter 1-40 Hz (removes slow drift and high freq noise)
        raw.filter(l_freq=1.0, h_freq=40.0, method="iir", verbose=False)

        # Notch filter at 60 Hz 
        raw.notch_filter(freqs=60.0, verbose=False)

        # Common average reference
        raw.set_eeg_reference("average", projection=False, verbose=False)

        # Get annotations
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Filter for T1 and T2 only
        target_events = {k: v for k, v in event_id.items() if k in ["T1", "T2"]}
        if not target_events:
            log.warning(f"  {subject_id} run {run_num}: no T1/T2 events found.")
            continue

        data = raw.get_data()  # shape: (64, n_timepoints)
        n_channels, n_times = data.shape

        for event in events:
            onset_sample = event[0]
            event_code   = event[2]

            # Find event label
            label_raw = None
            for name, code in event_id.items():
                if code == event_code and name in ["T1", "T2"]:
                    label_raw = name
                    break

            if label_raw is None:
                continue

            # Extract 4-second window
            end_sample = onset_sample + WINDOW_SAMP
            if end_sample > n_times:
                log.warning(f"  {subject_id} run {run_num}: window exceeds recording, skipping.")
                continue

            window = data[:, onset_sample:end_sample].astype(np.float32)

            # QC checks
            qc_flags = []
            if np.isnan(window).any():
                qc_flags.append("has_nan")
            if np.isinf(window).any():
                qc_flags.append("has_inf")
            channel_stds = window.std(axis=1)
            if (channel_stds < 1e-10).any():
                qc_flags.append("flat_channel")

            unified_label = LABEL_MAP.get(label_raw, label_raw)
            sample_id     = f"eeg_{subject_id}_run{run_num}_{onset_sample}"

            windows.append(window)
            metadata.append({
                "sample_id":             sample_id,
                "dataset_name":          "eegmmidb",
                "modality":              "EEG",
                "subject_or_patient_id": subject_id,
                "source_file_or_record": edf_path.name,
                "run_id":                run_num,
                "split":                 "supervised",
                "label_or_event":        unified_label,
                "event_code":            label_raw,
                "onset_sample":          onset_sample,
                "sampling_rate_hz":      SFREQ,
                "n_channels":            n_channels,
                "n_samples":             WINDOW_SAMP,
                "channel_schema":        "64ch_10-10_system",
                "qc_flags":              ";".join(qc_flags) if qc_flags else "",
            })

    return windows, metadata


def main():
    log.info("=" * 60)
    log.info("EEG Preprocessing: EEGMMIDB")
    log.info("=" * 60)

    # Find all subject directories
    eeg_root = DATA_RAW / "physionet.org" / "files" / "eegmmidb" / "1.0.0"
    if not eeg_root.exists():
        eeg_root = DATA_RAW
    
    subject_dirs = sorted([
        d for d in eeg_root.iterdir()
        if d.is_dir() and d.name.startswith("S")
    ])

    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories found under {eeg_root}")

    log.info(f"Found {len(subject_dirs)} subjects.")

    all_windows  = []
    all_metadata = []

    for subject_dir in subject_dirs:
        log.info(f"Processing {subject_dir.name}...")
        windows, metadata = preprocess_subject(subject_dir)
        all_windows.extend(windows)
        all_metadata.extend(metadata)

    if not all_windows:
        raise RuntimeError("No EEG windows were extracted. Check data paths.")

    log.info(f"Total windows extracted: {len(all_windows)}")

    # Stack into array
    X = np.stack(all_windows, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(all_metadata)

    log.info(f"EEG array shape: {X.shape}")

    # Save full outputs
    np.save(DATA_PROCESSED / "eeg_windows.npy", X)
    meta_df.to_csv(DATA_PROCESSED / "eeg_metadata.csv", index=False)

    # Save 100-window sample pack
    np.save(SAMPLE_DIR / "eeg_windows_sample.npy", X[:100])
    meta_df.head(100).to_csv(SAMPLE_DIR / "eeg_metadata_sample.csv", index=False)

    log.info("Sample pack saved (100 windows).")

    # Write manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "eeg_windows": {
            "file":            "data/processed/eeg/eeg_windows.npy",
            "file_size_bytes": (DATA_PROCESSED / "eeg_windows.npy").stat().st_size,
            "shape":           list(X.shape),
            "dtype":           "float32",
            "n_windows":       len(X),
            "window_size_s":   WINDOW_SEC,
            "sampling_rate_hz": SFREQ,
            "runs_used":       MOTOR_RUNS,
            "labels":          meta_df["label_or_event"].value_counts().to_dict(),
        }
    }

    with open(DATA_PROCESSED / "eeg_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("EEG manifest written.")
    log.info("EEG preprocessing complete.")


if __name__ == "__main__":
    main()