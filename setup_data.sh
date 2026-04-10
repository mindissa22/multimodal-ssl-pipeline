#!/usr/bin/env bash

# setup_data.sh
# Orchestration wrapper: creates folder structure and downloads all datasets.
# No preprocessing — handled by Python scripts.


set -euo pipefail


# Configuration

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
LOG_DIR="$PROJECT_ROOT/logs"
MANIFEST="$PROJECT_ROOT/data/download_manifest.json"
DOWNLOAD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

PAMAP2_URL="https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
WISDM_URL="https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
EEGMMIDB_URL="https://physionet.org/files/eegmmidb/1.0.0/"
PTBXL_URL="https://physionet.org/files/ptb-xl/1.0.3/"


# Logging

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

fail() {
    echo "[ERROR] $*" | tee -a "$LOG_FILE" >&2
    exit 1
}


# Step 1: Creating the folder structure

log "Creating directory structure..."

mkdir -p \
    "$DATA_DIR/raw/pamap2" \
    "$DATA_DIR/raw/wisdm" \
    "$DATA_DIR/raw/eegmmidb" \
    "$DATA_DIR/raw/ptbxl" \
    "$DATA_DIR/interim/har" \
    "$DATA_DIR/interim/eeg" \
    "$DATA_DIR/interim/ecg" \
    "$DATA_DIR/processed/har" \
    "$DATA_DIR/processed/eeg" \
    "$DATA_DIR/processed/ecg" \
    "$PROJECT_ROOT/reports" \
    "$PROJECT_ROOT/submission_sample/har" \
    "$PROJECT_ROOT/submission_sample/eeg" \
    "$PROJECT_ROOT/submission_sample/ecg" \
    "$PROJECT_ROOT/configs"

log "Folder structure created."


# Verifying if a download completed

verify_download() {
    local path="$1"
    local label="$2"
    if [ ! -e "$path" ]; then
        fail "$label download failed — path does not exist: $path"
    fi
    if [ -f "$path" ] && [ ! -s "$path" ]; then
        fail "$label download failed — file is empty: $path"
    fi
    log "$label download verified."
}


# Step 2: Download of PAMAP2

PAMAP2_ZIP="$DATA_DIR/raw/pamap2/pamap2.zip"

if [ -f "$PAMAP2_ZIP" ]; then
    log "PAMAP2 zip already exists, skipping download."
else
    log "Downloading PAMAP2..."
    wget --quiet --show-progress \
        -O "$PAMAP2_ZIP" \
        "$PAMAP2_URL" 2>&1 | tee -a "$LOG_FILE" || fail "PAMAP2 download failed."
fi

verify_download "$PAMAP2_ZIP" "PAMAP2"
log "Extracting PAMAP2..."
unzip -q -o "$PAMAP2_ZIP" -d "$DATA_DIR/raw/pamap2/" 2>&1 | tee -a "$LOG_FILE"
log "PAMAP2 extracted."


# Step 3: Download of WISDM

WISDM_ZIP="$DATA_DIR/raw/wisdm/wisdm.zip"

if [ -f "$WISDM_ZIP" ]; then
    log "WISDM zip already exists, skipping download."
else
    log "Downloading WISDM..."
    wget --quiet --show-progress \
        -O "$WISDM_ZIP" \
        "$WISDM_URL" 2>&1 | tee -a "$LOG_FILE" || fail "WISDM download failed."
fi

verify_download "$WISDM_ZIP" "WISDM"
log "Extracting WISDM..."
unzip -q -o "$WISDM_ZIP" -d "$DATA_DIR/raw/wisdm/" 2>&1 | tee -a "$LOG_FILE"
log "WISDM extracted."


# Step 4: Download of EEGMMIDB (4, 8 and 12 runs only)

EEGMMIDB_DIR="$DATA_DIR/raw/eegmmidb"
mkdir -p "$EEGMMIDB_DIR"

EDF_COUNT=$(find "$EEGMMIDB_DIR" -name "*.edf" | wc -l)
if [ "$EDF_COUNT" -gt 300 ]; then
    log "EEGMMIDB already downloaded, skipping."
else
    log "Downloading EEGMMIDB runs 4, 8, 12 only"
    for subject in $(seq -f "%03g" 1 109); do
        for run in 04 08 12; do
            wget --quiet -N \
                -P "$EEGMMIDB_DIR/S${subject}/" \
                "https://physionet.org/files/eegmmidb/1.0.0/S${subject}/S${subject}R${run}.edf" \
                2>> "$LOG_FILE" || log "Warning: S${subject}R${run} download failed"
        done
    done
fi

EDF_COUNT=$(find "$EEGMMIDB_DIR" -name "*.edf" | wc -l)
if [ "$EDF_COUNT" -lt 300 ]; then
    fail "EEGMMIDB appears incomplete — only $EDF_COUNT EDF files found."
fi
log "EEGMMIDB verified: $EDF_COUNT EDF files found."


# Step 5: Download of PTB-XL

PTBXL_DIR="$DATA_DIR/raw/ptbxl"

if [ -f "$PTBXL_DIR/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv" ]; then
    log "PTB-XL already downloaded, skipping."
else
    log "Downloading PTB-XL (this is ~1.7 GB and may take a while)..."
    wget --quiet --show-progress \
        -r -N -c -np \
        -P "$PTBXL_DIR" \
        "$PTBXL_URL" 2>&1 | tee -a "$LOG_FILE" || fail "PTB-XL download failed."
fi

verify_download "$PTBXL_DIR/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv" "PTB-XL"
log "PTB-XL verified."


# Step 6: Writing download manifest

log "Writing download manifest..."

cat > "$MANIFEST" <<EOF
{
  "download_date": "$DOWNLOAD_DATE",
  "datasets": {
    "pamap2": {
      "url": "$PAMAP2_URL",
      "version": "UCI ML Repository ID 231",
      "local_path": "data/raw/pamap2/",
      "status": "downloaded"
    },
    "wisdm": {
      "url": "$WISDM_URL",
      "version": "UCI ML Repository ID 507",
      "local_path": "data/raw/wisdm/",
      "status": "downloaded"
    },
    "eegmmidb": {
      "url": "$EEGMMIDB_URL",
      "version": "1.0.0",
      "local_path": "data/raw/eegmmidb/",
      "status": "downloaded"
    },
    "ptbxl": {
      "url": "$PTBXL_URL",
      "version": "1.0.3",
      "local_path": "data/raw/ptbxl/",
      "status": "downloaded"
    }
  }
}
EOF

log "Download manifest written to $MANIFEST"
log "Setup complete. Run: preprocess_har.py to begin preprocessing."