#!/bin/bash

# Convert line endings from DOS (CRLF) to Unix (LF) for specified file types
# Recursively processes all files starting from current working directory

echo "Converting line endings from DOS to Unix (recursive from $(pwd))..."

# Find and convert files recursively
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.toml" -o -name "*.sh" \) -exec dos2unix {} \; 2>/dev/null

echo "Line ending conversion completed!"

# Make all .sh files executable recursively
find . -type f -name "*.sh" -exec chmod +x {} \;

echo "Made all .sh files executable!"
echo "Processed files in $(pwd) and all subdirectories."