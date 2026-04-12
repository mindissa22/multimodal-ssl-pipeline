"""
tests/test_smoke.py
Smoke tests for format checks on processed outputs.
Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
SAMPLE = ROOT / "submission_sample"

# --- HAR ---

def test_har_pretrain_loads():
    X = np.load(PROC / "har/har_pretrain.npy")
    assert X is not None

def test_har_pretrain_dtype():
    X = np.load(PROC / "har/har_pretrain.npy")
    assert X.dtype == np.float32

def test_har_pretrain_shape():
    X = np.load(PROC / "har/har_pretrain.npy")
    assert X.ndim == 3
    assert X.shape[1] == 6
    assert X.shape[2] == 200

def test_har_pretrain_no_nans():
    X = np.load(PROC / "har/har_pretrain.npy")
    assert not np.isnan(X).any()

def test_har_supervised_shape():
    X = np.load(PROC / "har/har_supervised.npy")
    assert X.shape[1] == 6
    assert X.shape[2] == 100

def test_har_supervised_dtype():
    X = np.load(PROC / "har/har_supervised.npy")
    assert X.dtype == np.float32

def test_har_supervised_no_nans():
    X = np.load(PROC / "har/har_supervised.npy")
    assert not np.isnan(X).any()

def test_har_pretrain_sample_100():
    X = np.load(SAMPLE / "har/har_pretrain_sample.npy")
    assert X.shape[0] == 100

def test_har_supervised_sample_100():
    X = np.load(SAMPLE / "har/har_supervised_sample.npy")
    assert X.shape[0] == 100

def test_har_metadata_columns():
    df = pd.read_csv(PROC / "har/har_pretrain_metadata.csv")
    required = ["sample_id", "dataset_name", "modality",
                "subject_or_patient_id", "sampling_rate_hz",
                "n_channels", "n_samples", "channel_schema", "qc_flags"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

def test_har_sampling_rate():
    df = pd.read_csv(PROC / "har/har_pretrain_metadata.csv")
    assert (df["sampling_rate_hz"] == 20).all()

def test_har_both_datasets_present():
    df = pd.read_csv(PROC / "har/har_pretrain_metadata.csv")
    assert "pamap2" in df["dataset_name"].values
    assert "wisdm" in df["dataset_name"].values

# --- EEG ---

def test_eeg_loads():
    X = np.load(PROC / "eeg/eeg_windows.npy")
    assert X is not None

def test_eeg_dtype():
    X = np.load(PROC / "eeg/eeg_windows.npy")
    assert X.dtype == np.float32

def test_eeg_shape():
    X = np.load(PROC / "eeg/eeg_windows.npy")
    assert X.ndim == 3
    assert X.shape[1] == 64
    assert X.shape[2] == 640

def test_eeg_no_nans():
    X = np.load(PROC / "eeg/eeg_windows.npy")
    assert not np.isnan(X).any()

def test_eeg_sample_100():
    X = np.load(SAMPLE / "eeg/eeg_windows_sample.npy")
    assert X.shape[0] == 100

def test_eeg_labels():
    df = pd.read_csv(PROC / "eeg/eeg_metadata.csv")
    assert set(df["label_or_event"].unique()) == {"left_fist", "right_fist"}

def test_eeg_sampling_rate():
    df = pd.read_csv(PROC / "eeg/eeg_metadata.csv")
    assert (df["sampling_rate_hz"] == 160).all()

# --- ECG ---

def test_ecg_loads():
    X = np.load(PROC / "ecg/ecg_signals.npy")
    assert X is not None

def test_ecg_dtype():
    X = np.load(PROC / "ecg/ecg_signals.npy")
    assert X.dtype == np.float32

def test_ecg_shape():
    X = np.load(PROC / "ecg/ecg_signals.npy")
    assert X.ndim == 3
    assert X.shape[1] == 12
    assert X.shape[2] == 1000

def test_ecg_no_nans():
    X = np.load(PROC / "ecg/ecg_signals.npy")
    assert not np.isnan(X).any()

def test_ecg_sample_100():
    X = np.load(SAMPLE / "ecg/ecg_signals_sample.npy")
    assert X.shape[0] == 100

def test_ecg_splits():
    df = pd.read_csv(PROC / "ecg/ecg_metadata.csv")
    assert {"train", "val", "test"} <= set(df["split"].unique())

def test_ecg_sampling_rate():
    df = pd.read_csv(PROC / "ecg/ecg_metadata.csv")
    assert (df["sampling_rate_hz"] == 100).all()

def test_ecg_fold_metadata():
    df = pd.read_csv(PROC / "ecg/ecg_metadata.csv")
    assert "strat_fold" in df.columns