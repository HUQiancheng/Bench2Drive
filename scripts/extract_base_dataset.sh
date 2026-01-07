#!/bin/bash
# ============================================================================
# Bench2Drive BASE Dataset Extraction Script
# ============================================================================
# Extracts tar.gz files one-by-one, deletes each after verified success.
# Resumable, logs everything, never deletes on failure.
#
# USAGE: bash extract_base_dataset.sh
# ============================================================================

set -u

DATA_DIR="/workspace/data/Bench2Drive-Base"
LOG_FILE="${DATA_DIR}/extraction_log.txt"
LOCK_FILE="${DATA_DIR}/.extraction.lock"
MIN_FREE_GB=3

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# ----------------------------------------------------------------------------
# Get free space in GB
# ----------------------------------------------------------------------------
get_free_gb() {
    df -k "$DATA_DIR" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

# ----------------------------------------------------------------------------
# Cleanup lock on exit
# ----------------------------------------------------------------------------
cleanup() {
    rm -f "$LOCK_FILE"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main() {
    # Check data directory exists
    if [[ ! -d "$DATA_DIR" ]]; then
        echo "ERROR: Directory not found: $DATA_DIR"
        exit 1
    fi

    cd "$DATA_DIR" || exit 1

    # Prevent concurrent runs
    if [[ -f "$LOCK_FILE" ]]; then
        echo "ERROR: Another extraction is running (lock file exists: $LOCK_FILE)"
        echo "If this is wrong, delete the lock file manually and retry."
        exit 1
    fi

    # Create lock file and ensure cleanup on exit
    echo "$$" > "$LOCK_FILE"
    trap cleanup EXIT

    # Start logging
    log "========================================"
    log "EXTRACTION STARTED"
    log "Directory: $DATA_DIR"
    log "========================================"

    # Get list of tar.gz files
    shopt -s nullglob
    local tar_files=(*.tar.gz)
    shopt -u nullglob

    local total=${#tar_files[@]}

    if [[ $total -eq 0 ]]; then
        log "No tar.gz files found. Nothing to do."
        exit 0
    fi

    log "Found $total tar.gz files"
    log ""

    local ok=0
    local fail=0

    for i in "${!tar_files[@]}"; do
        local tarfile="${tar_files[$i]}"
        local num=$((i + 1))
        local expected_dir="${tarfile%.tar.gz}"

        log "----------------------------------------"
        log "[$num/$total] $tarfile"

        # Check disk space
        local free
        free=$(get_free_gb)
        if [[ $free -lt $MIN_FREE_GB ]]; then
            log "STOP: Only ${free}GB free (need ${MIN_FREE_GB}GB)"
            log "Re-run after freeing space."
            exit 1
        fi
        log "Free space: ${free}GB"

        # Extract
        log "Extracting..."
        local tar_err
        if ! tar_err=$(tar --overwrite -xzf "$tarfile" 2>&1); then
            log "FAILED: tar error: $tar_err"
            log "Keeping: $tarfile"
            ((fail++))
            continue
        fi

        # Verify: directory exists?
        if [[ ! -d "$expected_dir" ]]; then
            log "FAILED: Expected directory '$expected_dir' not found"
            log "Keeping: $tarfile"
            ((fail++))
            continue
        fi

        # Verify: directory has files?
        local fcount
        fcount=$(find "$expected_dir" -type f 2>/dev/null | head -1 | wc -l)
        if [[ $fcount -eq 0 ]]; then
            log "FAILED: Directory '$expected_dir' is empty"
            log "Keeping: $tarfile"
            ((fail++))
            continue
        fi

        # Delete tar.gz
        if ! rm "$tarfile"; then
            log "FAILED: Could not delete $tarfile"
            ((fail++))
            continue
        fi

        # Final verify: tar.gz is gone?
        if [[ -e "$tarfile" ]]; then
            log "FAILED: $tarfile still exists after rm"
            ((fail++))
            continue
        fi

        log "OK: Extracted and deleted"
        ((ok++))
    done

    log ""
    log "========================================"
    log "COMPLETE: $ok succeeded, $fail failed"
    log "Free space: $(get_free_gb)GB"
    log "========================================"

    if [[ $fail -gt 0 ]]; then
        log "WARNING: $fail files failed - check log above"
        exit 1
    fi

    log "All done!"
    exit 0
}

main "$@"
