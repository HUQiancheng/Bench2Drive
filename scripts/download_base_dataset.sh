#!/bin/bash
# Download Bench2Drive BASE dataset (400GB, 1000 clips)
# Run with: bash scripts/download_base_dataset.sh

set -e

DATA_DIR="/workspace/data"
DATASET_NAME="Bench2Drive-Base"

echo "=== Bench2Drive BASE Dataset Downloader ==="
echo "Target directory: ${DATA_DIR}/${DATASET_NAME}"
echo "Size: ~400GB (1000 clips)"
echo ""

# Create data directory
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# Download using huggingface_hub
python << 'EOF'
from huggingface_hub import snapshot_download
import os

print("Starting download from HuggingFace...")
print("This will take a while for 400GB. Download is resumable if interrupted.")
print("")

snapshot_download(
    repo_id="rethinklab/Bench2Drive",
    repo_type="dataset",
    local_dir="Bench2Drive-Base",
    resume_download=True
)

print("")
print("Download complete!")
print(f"Dataset location: {os.path.abspath('Bench2Drive-Base')}")
EOF

echo ""
echo "=== Download finished ==="
echo "Dataset is at: ${DATA_DIR}/${DATASET_NAME}"
