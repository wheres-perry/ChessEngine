#!/bin/bash

# Check if an output filename is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <output_filename_without_extension>"
  exit 1
fi

# Define the output directory and filename, and ensure the directory exists
OUTPUT_DIR="/out/docker"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILENAME="$OUTPUT_DIR/$1.zip"
TEMP_FILE_LIST="files_to_archive.txt"

# Run line.sh before packaging
echo "Running line.sh..."
if [ -f "line.sh" ]; then
  bash line.sh
  if [ $? -eq 0 ]; then
    echo "Line endings fixed successfully."
  else
    echo "Warning: line.sh failed, continuing with packaging..."
  fi
else
  echo "Warning: line.sh not found, skipping line ending fixes..."
fi

# Get the list of all files not ignored by Git, excluding specified directories
# --cached: All files tracked in the index
# --others: All untracked files
# --exclude-standard: Respect .gitignore, .git/info/exclude, and global gitignore
echo "Finding files to archive..."
git ls-files --cached --others --exclude-standard \
    --exclude='data/*' \
    --exclude='.devcontainer/*' \
    --exclude='.scripts/*' \
    --exclude='.vscode/*' \
    --exclude='out/*' > "$TEMP_FILE_LIST"

# Remove the temp file list from the archive list (in case it was tracked)
grep -v "^$TEMP_FILE_LIST$" "$TEMP_FILE_LIST" > "${TEMP_FILE_LIST}.tmp" && mv "${TEMP_FILE_LIST}.tmp" "$TEMP_FILE_LIST"

# Check if any files were found
if [ ! -s "$TEMP_FILE_LIST" ]; then
  echo "No files to archive (or error listing files)."
  rm -f "$TEMP_FILE_LIST"
  exit 1
fi

# Create the zip archive from the list of files
echo "Creating archive: $OUTPUT_FILENAME"
# The @ symbol tells zip to read the list of files from TEMP_FILE_LIST
zip "$OUTPUT_FILENAME" -@ < "$TEMP_FILE_LIST"

# Clean up the temporary file list
rm -f "$TEMP_FILE_LIST"

if [ -f "$OUTPUT_FILENAME" ]; then
  echo "Archive created: $OUTPUT_FILENAME"
else
  echo "Error creating archive."
  exit 1
fi