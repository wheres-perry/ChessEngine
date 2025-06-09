# ChessEngine

A Python-based chess engine.

## Project Overview

This project aims to create a chess engine capable of evaluating board positions and determining optimal moves. It utilizes the `python-chess` library for board representation and move generation, `torch` for potential neural network integration, and `pytest` for testing.

## Tech Stack

*   Python (>=3.12)
*   Poetry for dependency management
*   Miniconda for environment management
*   pytest for testing

##

## Getting Started

### Running Tests

This project uses Poetry for dependency management. To run tests:

```bash
poetry install
poetry run pytest
```

**Note:** Some performance tests may fail in their current state due to optimization work in progress. These failures do not indicate functional issues with the chess engine.

### Current Status

This project is still in development and not fully complete. The `main.py` file serves as a showcase of the current capabilities and features implemented so far.