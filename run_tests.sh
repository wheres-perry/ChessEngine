#!/bin/sh
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Force immediate output with explicit flushing
python -u -m pytest unit_tests.py -v -s --tb=native "${@}" 2>&1 | while IFS= read -r line; do
    echo "$line"
done

