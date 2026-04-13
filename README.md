# Multimodal SSL Pipeline

A reproducible pipeline for downloading, harmonising, preprocessing and validating open multimodal time series datasets for downstream self supervised learning (SSL) workflows.

## Datasets
| Dataset | Modality | Sampling Rate |
|---------|----------|---------------|
| PAMAP2 | HAR | 100 Hz → 20 Hz |
| WISDM | HAR | 20 Hz |
| EEGMMIDB | EEG | 160 Hz |
| PTB-XL | ECG | 100 Hz |
| mHealth | HAR/ECG | 50 Hz → 20 Hz |

## Project Structure
multimodal-ssl-pipeline/
setup_data.sh          # Downloads all datasets and creates folder structure
preprocess_har.py      # HAR preprocessing: PAMAP2 + WISDM
preprocess_eeg.py      # EEG preprocessing: EEGMMIDB
preprocess_ecg.py      # ECG preprocessing: PTB-XL
validate_outputs.py    # Validates all processed outputs
requirements.txt       # Python dependencies
configs/               # Configuration files
data/
raw/               # Raw downloaded datasets
interim/           # Intermediate processing outputs
processed/         # Final processed outputs
reports/               # Validation report and resource estimate
submission_sample/     # 100-window sample packs per dataset
logs/                  # Download and processing logs

## Requirements

- Python 3.9+
- bash (Mac/Linux) or equivalent
- wget
- unzip

## Installation

Clone the repository:

```bash
git clone https://github.com/mindissa22/multimodal-ssl-pipeline.git
cd multimodal-ssl-pipeline
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## How to Reproduce the Pipeline

Run these commands in order:

### Step 1 — Download all datasets

```bash
bash setup_data.sh
```

This will:
- Create the full folder structure
- Download PAMAP2, WISDM, EEGMMIDB, and PTB-XL automatically
- Verify each download completed successfully
- Write a download manifest to `data/download_manifest.json`

Expected time: 1-2 hours depending on internet speed.
Expected disk usage: ~4.5 GB

### Step 2 — Preprocess HAR data

```bash
python preprocess_har.py
```

This will:
- Load PAMAP2 wrist IMU and WISDM watch sensor data
- Harmonise both datasets to 20 Hz, 6 channel schema
- Drop PAMAP2 class 0 (transient) labels
- Create pretraining windows (10s, no overlap, no labels)
- Create supervised windows (5s, 50% overlap, majority vote labels)
- Save outputs to `data/processed/har/`
- Save 100-window sample pack to `submission_sample/har/`

### Step 3 — Preprocess EEG data

```bash
python preprocess_eeg.py
```

This will:
- Load EEGMMIDB runs 4, 8, and 12 for all 109 subjects
- Parse T1 (left fist) and T2 (right fist) annotations
- Extract 4-second event-aligned windows at 160 Hz
- Apply band-pass (1-40 Hz) and notch (60 Hz) filters
- Save outputs to `data/processed/eeg/`
- Save 100-window sample pack to `submission_sample/eeg/`

### Step 4 — Preprocess ECG data

```bash
python preprocess_ecg.py
```

This will:
- Load PTB-XL 12-lead ECG recordings at 100 Hz
- Apply train/val/test split using provided strat_fold column
- Apply high-pass filter (0.5 Hz) and notch filter (50 Hz)
- Save outputs to `data/processed/ecg/`
- Save 100-record sample pack to `submission_sample/ecg/`

### Step 5 — Validate outputs

```bash
python validate_outputs.py
```

This will:
- Check all array shapes, dtypes, and integrity
- Verify no NaN or infinite values
- Check label distributions
- Verify sample packs have 100 windows each
- Write validation report to `reports/validation_report.txt`
- Write machine-readable report to `reports/validation_report.json`

### Step 6 — Run smoke tests

```bash
pytest tests/ -v
```

This will:
- Check all array shapes and dtypes
- Verify sample packs have 100 windows each
- Validate all manifest JSON files

## Processed Output Format

All outputs are stored as float32 NumPy arrays with shape [N, C, T]:
- N = number of windows or records
- C = number of channels
- T = number of timepoints

| Dataset | Shape | Description |
|---------|-------|-------------|
| HAR pretrain | [N, 6, 200] | 10s windows, no labels |
| HAR supervised | [N, 6, 100] | 5s windows, with labels |
| EEG | [N, 64, 640] | 4s event-aligned windows |
| ECG | [N, 12, 1000] | 10s recordings |

Each output is accompanied by a metadata CSV with columns:
`sample_id, dataset_name, modality, subject_or_patient_id, source_file_or_record, split, label_or_event, sampling_rate_hz, n_channels, n_samples, channel_schema, qc_flags`

## Design Decisions

### HAR
- 6 channel schema: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z from wrist/watch
- PAMAP2 downsampled from 100 Hz to 20 Hz with anti-aliasing filter
- PAMAP2 class 0 (transient) rows dropped
- Majority vote used for window label assignment
- Tied votes discarded

### EEG
- Native 160 Hz retained (motor imagery signals up to ~40 Hz)
- Only runs 4, 8, 12 used (left vs right fist imagery)
- T0 (rest) windows excluded from primary output

### ECG
- 100 Hz selected over 500 Hz (sufficient for clinical features, lower storage)
- PTB-XL provided strat_fold used directly: folds 1-8 train, fold 9 val, fold 10 test
- Patient-level assignment guaranteed by dataset authors — no leakage

## How Outputs Feed a Self-Supervised Learning Pipeline

The processed outputs are designed to slot directly into a self supervised learning workflow:

1. **Pretraining** — The unlabelled HAR windows are used to train a model to recognise patterns in movement signals without any labels. The model learns by comparing different versions of the same window.

2. **Fine-tuning** — The labelled HAR, EEG and ECG outputs are used to teach the pretrained model specific tasks like activity recognition or heart condition classification. Fewer labels are needed because the model already understands the signals.

3. **Evaluation** — Subjects and patients held out from training entirely 
are used to test the final model. Subject and patient IDs are preserved 
in all metadata to prevent data leakage across splits.

## Resource Estimates

| | Storage |
|--|--|
| Raw data | ~4.5 GB |
| Processed data | ~1.15 GB |
| Peak RAM | ~4 GB (EEG processing) |
| Runtime | ~50 minutes (excluding downloads) |

## Validation Report

After running `validate_outputs.py`, the full validation report is available at:
- `reports/validation_report.txt` — human readable
- `reports/validation_report.json` — machine readable