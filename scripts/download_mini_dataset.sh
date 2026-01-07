#!/bin/bash
# ============================================================================
# Bench2Drive MINI Dataset Downloader
# ============================================================================
# Downloads the 10 representative clips (~3.2GB) from HuggingFace
#
# USAGE: HF_TOKEN=your_token bash download_mini_dataset.sh
# ============================================================================

set -u

DATA_DIR="/workspace/data/Bench2Drive-Mini"

echo "=== Bench2Drive MINI Dataset Downloader ==="
echo "Target directory: $DATA_DIR"
echo "Files: 10 clips (~3.2GB)"
echo ""

mkdir -p "$DATA_DIR"

python << EOF
import os
from huggingface_hub import hf_hub_download, login

# Login with token from environment
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

DATA_DIR = "/workspace/data/Bench2Drive-Mini"

MINI_FILES = [
    "AccidentTwoWays_Town12_Route1444_Weather0.tar.gz",
    "Accident_Town03_Route156_Weather0.tar.gz",
    "ConstructionObstacle_Town05_Route68_Weather8.tar.gz",
    "ControlLoss_Town11_Route401_Weather11.tar.gz",
    "DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz",
    "HardBreakRoute_Town01_Route30_Weather3.tar.gz",
    "OppositeVehicleTakingPriority_Town13_Route600_Weather2.tar.gz",
    "ParkedObstacle_Town10HD_Route371_Weather7.tar.gz",
    "VehicleTurningRoute_Town15_Route443_Weather1.tar.gz",
    "YieldToEmergencyVehicle_Town04_Route165_Weather7.tar.gz",
]

print("Downloading from HuggingFace...")
print("(Uses cached token from previous login)")
print("")

ok = 0
fail = 0
total = len(MINI_FILES)

for i, filename in enumerate(MINI_FILES):
    print(f"[{i+1}/{total}] {filename}")

    target_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(target_path):
        print("  Already exists, skipping")
        ok += 1
        continue

    try:
        hf_hub_download(
            repo_id="rethinklab/Bench2Drive",
            repo_type="dataset",
            filename=filename,
            local_dir=DATA_DIR,
        )
        print("  OK")
        ok += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        fail += 1

print("")
print("=== Download complete ===")
print(f"Success: {ok}, Failed: {fail}")
print(f"Location: {DATA_DIR}")
EOF

echo ""
echo "Done!"
