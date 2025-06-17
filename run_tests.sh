#!/bin/sh
# poetry install --with=dev --no-cache --no-root
# poetry config virtualenvs.create true 
# poetry config virtualenvs.in-project true 
poetry run pytest unit_tests.py ${@}