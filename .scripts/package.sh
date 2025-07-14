#!/bin/bash
set -e


# --- Setup ---
# Get the directory of this script, and the project root (one level up)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")


# --- Validation ---
if [ -z "$1" ]; then
  echo "Usage: $0 <output_filename_without_extension>"
  exit 1
fi


# --- Change to Project Root ---
# All subsequent commands will run from the project's root directory
cd "$PROJECT_ROOT"


# --- Configuration ---
OUTPUT_DIR="out/docker"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILENAME="$OUTPUT_DIR/$1.zip"
TEMP_FILE_LIST="files_to_archive.txt"
LINE_SH_PATH=".scripts/line.sh"


# --- Run line.sh ---
echo "Running line.sh..."
if [ -f "$LINE_SH_PATH" ]; then
  bash "$LINE_SH_PATH"
  echo "Line ending check complete."
else
  echo "Warning: $LINE_SH_PATH not found, skipping line ending fixes..."
fi


# --- Find Files ---
# Get the list of all files not ignored by Git
echo "Finding files to archive..."
git ls-files --cached --others --exclude-standard \
  --exclude='data/*' \
  --exclude='.devcontainer/*' \
  --exclude='.scripts/*' \
  --exclude='.vscode/*' \
  --exclude='.docker-cache/*' \
  --exclude='out/*' >"$TEMP_FILE_LIST"


# --- Create Archive ---
# Check if any files were found
if [ ! -s "$TEMP_FILE_LIST" ]; then
  echo "No files to archive."
  rm -f "$TEMP_FILE_LIST"
  exit 1
fi

# Create the zip archive
echo "Creating archive: $OUTPUT_FILENAME"
zip -r "$OUTPUT_FILENAME" -@ < "$TEMP_FILE_LIST"

# Clean up temporary file
rm -f "$TEMP_FILE_LIST"

# Success message
echo "Archive created successfully: $OUTPUT_FILENAME"
echo "$(zip -sf "$OUTPUT_FILENAME" | wc -l) files archived."
