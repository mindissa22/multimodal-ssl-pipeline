"""
tests/test_manifest.py
======================
Smoke tests for manifest validation.
Run with: pytest tests/
"""

import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"

def load_manifest(path):
    with open(path) as f:
        return json.load(f)

# HAR Manifest

def test_har_manifest_exists():
    assert (PROC / "har/har_manifest.json").exists()

def test_har_manifest_valid_json():
    man = load_manifest(PROC / "har/har_manifest.json")
    assert isinstance(man, dict)

def test_har_manifest_has_pretrain_key():
    man = load_manifest(PROC / "har/har_manifest.json")
    assert "har_pretrain" in man

def test_har_manifest_has_supervised_key():
    man = load_manifest(PROC / "har/har_manifest.json")
    assert "har_supervised" in man

def test_har_manifest_pretrain_shape():
    man = load_manifest(PROC / "har/har_manifest.json")
    shape = man["har_pretrain"]["shape"]
    assert shape[1] == 6
    assert shape[2] == 200

def test_har_manifest_supervised_shape():
    man = load_manifest(PROC / "har/har_manifest.json")
    shape = man["har_supervised"]["shape"]
    assert shape[1] == 6
    assert shape[2] == 100

def test_har_manifest_pretrain_window_size():
    man = load_manifest(PROC / "har/har_manifest.json")
    assert man["har_pretrain"]["window_size_s"] == 10

def test_har_manifest_supervised_window_size():
    man = load_manifest(PROC / "har/har_manifest.json")
    assert man["har_supervised"]["window_size_s"] == 5

def test_har_manifest_supervised_overlap():
    man = load_manifest(PROC / "har/har_manifest.json")
    assert man["har_supervised"]["overlap"] == 0.5

def test_har_manifest_pretrain_dtype():
    man = load_manifest(PROC / "har/har_manifest.json")
    assert man["har_pretrain"]["dtype"] == "float32"

def test_har_manifest_has_label_counts():
    man = load_manifest(PROC / "har/har_manifest.json")
    assert "label_counts" in man["har_supervised"]

# EEG Manifest 

def test_eeg_manifest_exists():
    assert (PROC / "eeg/eeg_manifest.json").exists()

def test_eeg_manifest_valid_json():
    man = load_manifest(PROC / "eeg/eeg_manifest.json")
    assert isinstance(man, dict)

def test_eeg_manifest_has_windows_key():
    man = load_manifest(PROC / "eeg/eeg_manifest.json")
    assert "eeg_windows" in man

def test_eeg_manifest_shape():
    man = load_manifest(PROC / "eeg/eeg_manifest.json")
    shape = man["eeg_windows"]["shape"]
    assert shape[1] == 64
    assert shape[2] == 640

def test_eeg_manifest_sampling_rate():
    man = load_manifest(PROC / "eeg/eeg_manifest.json")
    assert man["eeg_windows"]["sampling_rate_hz"] == 160

def test_eeg_manifest_window_size():
    man = load_manifest(PROC / "eeg/eeg_manifest.json")
    assert man["eeg_windows"]["window_size_s"] == 4

def test_eeg_manifest_runs_used():
    man = load_manifest(PROC / "eeg/eeg_manifest.json")
    assert man["eeg_windows"]["runs_used"] == [4, 8, 12]

def test_eeg_manifest_dtype():
    man = load_manifest(PROC / "eeg/eeg_manifest.json")
    assert man["eeg_windows"]["dtype"] == "float32"

def test_eeg_manifest_has_labels():
    man = load_manifest(PROC / "eeg/eeg_manifest.json")
    assert "labels" in man["eeg_windows"]

# ECG Manifest 

def test_ecg_manifest_exists():
    assert (PROC / "ecg/ecg_manifest.json").exists()

def test_ecg_manifest_valid_json():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    assert isinstance(man, dict)

def test_ecg_manifest_has_signals_key():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    assert "ecg_signals" in man

def test_ecg_manifest_shape():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    shape = man["ecg_signals"]["shape"]
    assert shape[1] == 12
    assert shape[2] == 1000

def test_ecg_manifest_sampling_rate():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    assert man["ecg_signals"]["sampling_rate_hz"] == 100

def test_ecg_manifest_dtype():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    assert man["ecg_signals"]["dtype"] == "float32"

def test_ecg_manifest_has_split_counts():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    assert "split_counts" in man["ecg_signals"]

def test_ecg_manifest_split_counts_keys():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    splits = man["ecg_signals"]["split_counts"]
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

def test_ecg_manifest_has_label_counts():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    assert "label_counts" in man["ecg_signals"]

def test_ecg_manifest_n_leads():
    man = load_manifest(PROC / "ecg/ecg_manifest.json")
    assert man["ecg_signals"]["n_leads"] == 12