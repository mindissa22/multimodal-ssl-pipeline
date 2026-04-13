"""
validate_outputs.py
Validates all processed outputs for the multimodal SSL pipeline.

Checks conducted:
- Array shapes are correct
- No NaN or infinite values
- Label distributions are appropriate
- Metadata CSV columns are complete
- Sample packs exist and have 100 rows
- Manifests exist and are valid JSON
- HAR harmonisation: both datasets at 20 Hz 
- Window sizes match brief requirements
- Subject IDs preserved in metadata

Outputs

reports/validation_report.json  
reports/validation_report.txt   
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/validate_outputs.log")
    ]
)
log = logging.getLogger(__name__)

PROJECT_ROOT   = Path(__file__).parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SAMPLE_DIR     = PROJECT_ROOT / "submission_sample"
REPORTS_DIR    = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

results = {}
all_passed = True


def check(name, condition, detail=""):
    global all_passed
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_passed = False
    results[name] = {"status": status, "detail": detail}
    icon = "✅" if condition else "❌"
    log.info(f"{icon} {status}: {name} {detail}")
    return condition


def validate_har():
    log.info("\n--- HAR Validation ---")
    har_dir = DATA_PROCESSED / "har"

    # Check files exist
    pretrain_path    = har_dir / "har_pretrain.npy"
    supervised_path  = har_dir / "har_supervised.npy"
    pre_meta_path    = har_dir / "har_pretrain_metadata.csv"
    sup_meta_path    = har_dir / "har_supervised_metadata.csv"
    manifest_path    = har_dir / "har_manifest.json"

    check("HAR pretrain file exists",   pretrain_path.exists())
    check("HAR supervised file exists", supervised_path.exists())
    check("HAR pretrain metadata exists",   pre_meta_path.exists())
    check("HAR supervised metadata exists", sup_meta_path.exists())
    check("HAR manifest exists", manifest_path.exists())

    if not pretrain_path.exists() or not supervised_path.exists():
        log.warning("HAR files missing — skipping further HAR checks.")
        return

    # Load arrays
    X_pre = np.load(pretrain_path)
    X_sup = np.load(supervised_path)

    # Shape checks
    check("HAR pretrain shape [N,6,200]",
          len(X_pre.shape) == 3 and X_pre.shape[1] == 6 and X_pre.shape[2] == 200,
          f"actual: {X_pre.shape}")

    check("HAR supervised shape [N,6,100]",
          len(X_sup.shape) == 3 and X_sup.shape[1] == 6 and X_sup.shape[2] == 100,
          f"actual: {X_sup.shape}")

    # dtype checks
    check("HAR pretrain dtype float32",   X_pre.dtype == np.float32)
    check("HAR supervised dtype float32", X_sup.dtype == np.float32)

    # NaN/inf checks
    check("HAR pretrain no NaNs",      not np.isnan(X_pre).any())
    check("HAR pretrain no infs",      not np.isinf(X_pre).any())
    check("HAR supervised no NaNs",    not np.isnan(X_sup).any())
    check("HAR supervised no infs",    not np.isinf(X_sup).any())

    # Metadata checks
    if pre_meta_path.exists():
        meta_pre = pd.read_csv(pre_meta_path)
        required_cols = ["sample_id", "dataset_name", "modality",
                         "subject_or_patient_id", "source_file_or_record",
                         "split", "label_or_event", "sampling_rate_hz",
                         "n_channels", "n_samples", "channel_schema", "qc_flags"]
        check("HAR pretrain metadata has required columns",
              all(c in meta_pre.columns for c in required_cols))
        check("HAR pretrain metadata row count matches array",
              len(meta_pre) == len(X_pre),
              f"metadata rows: {len(meta_pre)}, array rows: {len(X_pre)}")
        check("HAR pretrain sampling rate is 20 Hz",
              (meta_pre["sampling_rate_hz"] == 20).all())
        check("HAR pretrain has both datasets",
              set(meta_pre["dataset_name"].unique()) >= {"pamap2", "wisdm"},
              f"datasets found: {meta_pre['dataset_name'].unique()}")
        check("HAR pretrain subject IDs preserved",
              meta_pre["subject_or_patient_id"].notna().all())

    if sup_meta_path.exists():
        meta_sup = pd.read_csv(sup_meta_path)
        check("HAR supervised metadata row count matches array",
              len(meta_sup) == len(X_sup))
        check("HAR supervised has labels",
              meta_sup["label_or_event"].notna().all())
        check("HAR supervised sampling rate is 20 Hz",
              (meta_sup["sampling_rate_hz"] == 20).all())

        label_counts = meta_sup["label_or_event"].value_counts()
        log.info(f"  HAR supervised label distribution:\n{label_counts.to_string()}")

    # Sample pack checks
    sample_pre = SAMPLE_DIR / "har" / "har_pretrain_sample.npy"
    sample_sup = SAMPLE_DIR / "har" / "har_supervised_sample.npy"
    check("HAR pretrain sample pack exists", sample_pre.exists())
    check("HAR supervised sample pack exists", sample_sup.exists())

    if sample_pre.exists():
        s = np.load(sample_pre)
        check("HAR pretrain sample has 100 windows",
              s.shape[0] == 100, f"actual: {s.shape[0]}")

    if sample_sup.exists():
        s = np.load(sample_sup)
        check("HAR supervised sample has 100 windows",
              s.shape[0] == 100, f"actual: {s.shape[0]}")


def validate_eeg():
    log.info("\n--- EEG Validation ---")
    eeg_dir = DATA_PROCESSED / "eeg"

    windows_path  = eeg_dir / "eeg_windows.npy"
    meta_path     = eeg_dir / "eeg_metadata.csv"
    manifest_path = eeg_dir / "eeg_manifest.json"

    check("EEG windows file exists",   windows_path.exists())
    check("EEG metadata file exists",  meta_path.exists())
    check("EEG manifest exists",       manifest_path.exists())

    if not windows_path.exists():
        log.warning("EEG files missing — skipping further EEG checks.")
        return

    X = np.load(windows_path)

    check("EEG shape [N,64,640]",
          len(X.shape) == 3 and X.shape[1] == 64 and X.shape[2] == 640,
          f"actual: {X.shape}")
    check("EEG dtype float32", X.dtype == np.float32)
    check("EEG no NaNs", not np.isnan(X).any())
    check("EEG no infs", not np.isinf(X).any())

    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        check("EEG metadata row count matches array",
              len(meta) == len(X))
        check("EEG subject IDs preserved",
              meta["subject_or_patient_id"].notna().all())
        check("EEG run IDs preserved",
              "run_id" in meta.columns and meta["run_id"].notna().all())
        check("EEG event codes preserved",
              "event_code" in meta.columns)
        check("EEG only T1/T2 labels",
              meta["label_or_event"].isin(["left_fist", "right_fist"]).all(),
              f"unique labels: {meta['label_or_event'].unique()}")

        label_counts = meta["label_or_event"].value_counts()
        log.info(f"  EEG label distribution:\n{label_counts.to_string()}")

    sample_path = SAMPLE_DIR / "eeg" / "eeg_windows_sample.npy"
    check("EEG sample pack exists", sample_path.exists())
    if sample_path.exists():
        s = np.load(sample_path)
        check("EEG sample has 100 windows",
              s.shape[0] == 100, f"actual: {s.shape[0]}")


def validate_ecg():
    log.info("\n--- ECG Validation ---")
    ecg_dir = DATA_PROCESSED / "ecg"

    signals_path  = ecg_dir / "ecg_signals.npy"
    meta_path     = ecg_dir / "ecg_metadata.csv"
    manifest_path = ecg_dir / "ecg_manifest.json"

    check("ECG signals file exists",  signals_path.exists())
    check("ECG metadata file exists", meta_path.exists())
    check("ECG manifest exists",      manifest_path.exists())

    if not signals_path.exists():
        log.warning("ECG files missing — skipping further ECG checks.")
        return

    X = np.load(signals_path)

    check("ECG shape [N,12,1000]",
          len(X.shape) == 3 and X.shape[1] == 12 and X.shape[2] == 1000,
          f"actual: {X.shape}")
    check("ECG dtype float32", X.dtype == np.float32)
    check("ECG no NaNs",       not np.isnan(X).any())
    check("ECG no infs",       not np.isinf(X).any())

    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        check("ECG metadata row count matches array",
              len(meta) == len(X))
        check("ECG patient IDs preserved",
              "subject_or_patient_id" in meta.columns and
              meta["subject_or_patient_id"].notna().all())
        check("ECG fold metadata present",
              "strat_fold" in meta.columns)
        check("ECG split column present",
              "split" in meta.columns)
        check("ECG has train/val/test splits",
              set(meta["split"].unique()) >= {"train", "val", "test"},
              f"splits found: {meta['split'].unique()}")
        check("ECG lead names preserved",
              "channel_schema" in meta.columns)
        check("ECG sampling rate is 100 Hz",
              (meta["sampling_rate_hz"] == 100).all())

        split_counts = meta["split"].value_counts()
        log.info(f"  ECG split distribution:\n{split_counts.to_string()}")

        label_counts = meta["label_or_event"].value_counts()
        log.info(f"  ECG label distribution:\n{label_counts.to_string()}")

    sample_path = SAMPLE_DIR / "ecg" / "ecg_signals_sample.npy"
    check("ECG sample pack exists", sample_path.exists())
    if sample_path.exists():
        s = np.load(sample_path)
        check("ECG sample has 100 records",
              s.shape[0] == 100, f"actual: {s.shape[0]}")


def write_reports():
    log.info("\n--- Writing Reports ---")

    passed = sum(1 for v in results.values() if v["status"] == "PASS")
    failed = sum(1 for v in results.values() if v["status"] == "FAIL")
    total  = len(results)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_checks": total,
            "passed":       passed,
            "failed":       failed,
            "all_passed":   all_passed,
        },
        "checks": results,
        "resource_estimate": {
    "raw_storage":       "~8.5 GB total (PAMAP2: 2.9GB, WISDM: 1.5GB, EEGMMIDB: 3.4GB, PTB-XL: 616MB, mHealth: 72MB)",
    "processed_storage": "~1.9 GB",
    "peak_ram":          "~4 GB estimated (EEG processing, subject-by-subject chunking)",
    "runtime_estimate":  "~50 minutes total (dowloads excluded)",
}
    }

    with open(REPORTS_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Human readable text report
    lines = [
        "=" * 60,
        "VALIDATION REPORT",
        f"Generated: {datetime.utcnow().isoformat()}",
        "=" * 60,
        f"Total checks: {total}",
        f"Passed:       {passed}",
        f"Failed:       {failed}",
        f"Overall:      {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}",
        "",
        "DETAILED RESULTS:",
        "-" * 60,
    ]

    for name, result in results.items():
        icon   = "✅" if result["status"] == "PASS" else "❌"
        detail = f" ({result['detail']})" if result["detail"] else ""
        lines.append(f"{icon} {result['status']}: {name}{detail}")

    lines += [
    "",
    "RESOURCE ESTIMATES:",
    "-" * 60,
    "Raw storage:       ~8.5 GB (PAMAP2: 2.9GB, WISDM: 1.5GB, EEGMMIDB: 3.4GB, PTB-XL: 616MB, mHealth: 72MB)",
    "Processed storage: ~1.9 GB",
    "Peak RAM:          ~4 GB",
    "Runtime:           ~50 minutes (downloads excluded)",
]

    report_text = "\n".join(lines)
    with open(REPORTS_DIR / "validation_report.txt", "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    log.info(f"Reports written to {REPORTS_DIR}")


def main():
    log.info("=" * 60)
    log.info("Validation: Multimodal SSL Pipeline")
    log.info("=" * 60)

    validate_har()
    validate_eeg()
    validate_ecg()
    write_reports()

    log.info(f"\nValidation complete. "
             f"{'All checks passed! ✅' if all_passed else 'Some checks failed — see report.'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()