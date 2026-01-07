#!/bin/bash
# ============================================================================
# Bench2Drive BASE Dataset Extraction Script
# ============================================================================
#
# PURPOSE:
#   Safely extract all tar.gz files one-by-one and delete each original
#   immediately after VERIFIED successful extraction to save disk space.
#
# SAFETY FEATURES:
#   1. Logs every action with timestamps to extraction_log.txt
#   2. Checks disk space before each extraction (stops if < 3GB free)
#   3. Verifies extraction success before deleting any tar.gz
#   4. Verifies the tar.gz is actually deleted after rm command
#   5. Resumable: safe to re-run if interrupted
#   6. Never deletes tar.gz unless extraction is 100% verified
#   7. Keeps failed tar.gz files for manual inspection
#
# USAGE:
#   bash extract_base_dataset.sh
#
# RESUME:
#   If interrupted, simply run the script again. It will:
#   - Skip tar.gz files that no longer exist (already processed)
#   - Re-extract tar.gz files whose directories exist but may be partial
#
# ============================================================================

# Exit on undefined variables (but not on command errors - we handle those)
set -u

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR="/workspace/data/Bench2Drive-Base"
LOG_FILE="${DATA_DIR}/extraction_log.txt"
MIN_FREE_SPACE_GB=3

# ============================================================================
# LOGGING FUNCTION
# ============================================================================
log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    local message="[$timestamp] $1"
    echo "$message"
    echo "$message" >> "$LOG_FILE"
}

log_separator() {
    log "============================================================"
}

# ============================================================================
# DISK SPACE CHECK
# ============================================================================
get_free_space_gb() {
    # Returns available space in GB as integer
    local free_kb
    free_kb=$(df -k "$DATA_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
    echo $((free_kb / 1024 / 1024))
}

# ============================================================================
# VERIFY DIRECTORY HAS FILES
# ============================================================================
directory_has_files() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        return 1
    fi
    # Check if directory contains at least one file
    local file_count
    file_count=$(find "$dir" -type f 2>/dev/null | head -5 | wc -l)
    [[ "$file_count" -gt 0 ]]
}

# ============================================================================
# COUNT FILES IN DIRECTORY
# ============================================================================
count_files() {
    local dir="$1"
    find "$dir" -type f 2>/dev/null | wc -l
}

# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================
main() {
    # ------------------------------------------------------------------
    # STEP 1: Validate data directory exists
    # ------------------------------------------------------------------
    if [[ ! -d "$DATA_DIR" ]]; then
        echo "FATAL ERROR: Data directory does not exist: $DATA_DIR"
        echo "Please ensure the download has started before running this script."
        exit 1
    fi

    cd "$DATA_DIR" || {
        echo "FATAL ERROR: Cannot change to directory: $DATA_DIR"
        exit 1
    }

    # ------------------------------------------------------------------
    # STEP 2: Initialize logging
    # ------------------------------------------------------------------
    log_separator
    log "BENCH2DRIVE BASE DATASET EXTRACTION"
    log_separator
    log "Data directory: $DATA_DIR"
    log "Log file: $LOG_FILE"
    log "Minimum free space required: ${MIN_FREE_SPACE_GB}GB"
    log ""

    # ------------------------------------------------------------------
    # STEP 3: Find all tar.gz files
    # ------------------------------------------------------------------
    # Use nullglob to handle case where no tar.gz files exist
    shopt -s nullglob
    local tar_files=(*.tar.gz)
    shopt -u nullglob

    local total_files=${#tar_files[@]}

    if [[ $total_files -eq 0 ]]; then
        log "No tar.gz files found in $DATA_DIR"
        log "Possible reasons:"
        log "  - Download has not started yet"
        log "  - Download is in progress (wait for completion)"
        log "  - All files have already been extracted"
        log ""
        log "Listing current directory contents:"
        ls -la "$DATA_DIR" >> "$LOG_FILE" 2>&1
        exit 0
    fi

    log "Found $total_files tar.gz files to process"
    log ""

    # ------------------------------------------------------------------
    # STEP 4: Process each tar.gz file
    # ------------------------------------------------------------------
    local processed=0
    local failed=0
    local current=0

    for tarfile in "${tar_files[@]}"; do
        current=$((current + 1))

        # Expected directory name (tar filename without .tar.gz extension)
        local expected_dir="${tarfile%.tar.gz}"

        log "------------------------------------------------------------"
        log "[$current/$total_files] FILE: $tarfile"
        log "Expected output directory: $expected_dir"

        # --------------------------------------------------------------
        # SAFETY CHECK: Disk space
        # --------------------------------------------------------------
        local free_space
        free_space=$(get_free_space_gb)

        if [[ $free_space -lt $MIN_FREE_SPACE_GB ]]; then
            log ""
            log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            log "FATAL: INSUFFICIENT DISK SPACE"
            log "Available: ${free_space}GB, Required: ${MIN_FREE_SPACE_GB}GB"
            log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            log ""
            log "Script stopped to prevent disk full errors."
            log "To resume:"
            log "  1. Free up disk space"
            log "  2. Run this script again"
            log ""
            log "PROGRESS: $processed extracted, $failed failed, $((total_files - current + 1)) remaining"
            exit 1
        fi

        log "Disk space available: ${free_space}GB (OK)"

        # --------------------------------------------------------------
        # EXTRACTION
        # --------------------------------------------------------------
        log "Extracting..."

        # Use tar with:
        #   -x: extract
        #   -z: gunzip
        #   -f: file
        #   --overwrite: overwrite existing files (handles partial extractions)
        # Capture both stdout and stderr
        local tar_output
        local tar_exit_code

        tar_output=$(tar --overwrite -xzf "$tarfile" 2>&1)
        tar_exit_code=$?

        if [[ $tar_exit_code -ne 0 ]]; then
            log "EXTRACTION FAILED!"
            log "tar exit code: $tar_exit_code"
            log "tar output: $tar_output"
            log "KEEPING original tar.gz file for manual inspection"
            failed=$((failed + 1))
            log ""
            continue
        fi

        log "tar command completed successfully (exit code 0)"

        # --------------------------------------------------------------
        # VERIFICATION: Check extracted directory exists and has files
        # --------------------------------------------------------------
        if [[ ! -d "$expected_dir" ]]; then
            log "VERIFICATION FAILED!"
            log "Expected directory '$expected_dir' does not exist after extraction"
            log "The tar.gz may contain a different directory structure"
            log "KEEPING original tar.gz file for manual inspection"
            failed=$((failed + 1))
            log ""
            continue
        fi

        local file_count
        file_count=$(count_files "$expected_dir")

        if [[ $file_count -eq 0 ]]; then
            log "VERIFICATION FAILED!"
            log "Directory '$expected_dir' exists but contains no files"
            log "KEEPING original tar.gz file for manual inspection"
            failed=$((failed + 1))
            log ""
            continue
        fi

        log "Verification passed: $expected_dir contains $file_count files"

        # --------------------------------------------------------------
        # DELETION: Remove tar.gz only after verified extraction
        # --------------------------------------------------------------
        log "Deleting tar.gz file..."

        rm -f "$tarfile"
        local rm_exit_code=$?

        # Double-check the file is actually gone
        if [[ -e "$tarfile" ]]; then
            log "WARNING: rm command executed but file still exists!"
            log "rm exit code was: $rm_exit_code"
            log "This is unexpected. Marking as failed."
            failed=$((failed + 1))
            log ""
            continue
        fi

        log "SUCCESS: $tarfile extracted and deleted"
        processed=$((processed + 1))
        log ""
    done

    # ------------------------------------------------------------------
    # STEP 5: Final summary
    # ------------------------------------------------------------------
    log_separator
    log "EXTRACTION COMPLETE"
    log_separator
    log ""
    log "SUMMARY:"
    log "  Total tar.gz files found: $total_files"
    log "  Successfully processed:   $processed"
    log "  Failed (kept original):   $failed"
    log ""

    # Final disk space
    local final_free
    final_free=$(get_free_space_gb)
    log "Final free disk space: ${final_free}GB"
    log ""

    if [[ $failed -gt 0 ]]; then
        log "WARNING: $failed file(s) failed to extract properly."
        log "The original tar.gz files have been preserved."
        log "Please check the log above for details on each failure."
        log ""
        log "Remaining tar.gz files:"
        ls -lh *.tar.gz 2>/dev/null >> "$LOG_FILE" || log "  (none)"
        exit 1
    fi

    log "All $processed files extracted successfully!"
    log ""
    log "Dataset is ready at: $DATA_DIR"
    exit 0
}

# ============================================================================
# RUN MAIN FUNCTION
# ============================================================================
main "$@"
