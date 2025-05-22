# ChessEngine

A Python-based chess engine.

## Project Overview

This project aims to create a chess engine capable of evaluating board positions and determining optimal moves. It utilizes the `python-chess` library for board representation and move generation, `torch` for potential neural network integration, and `pytest` for testing.

## Tech Stack

*   Python (>=3.12)
*   Poetry for dependency management
*   Miniconda for environment management
*   pytest for testing

## Setup and Installation

Follow these steps to set up the project locally:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd ChessEngine
    ```

2.  **Install Miniconda:**
    If you don't have Miniconda installed, download and install it from [here](https://docs.conda.io/en/latest/miniconda.html).

3.  **Create and activate a Conda environment:**
    It's recommended to create a dedicated environment for this project.
    ```bash
    conda create -n chessenv python=3.12 -y
    conda activate chessenv
    ```

4.  **Install Poetry:**
    If you don't have Poetry installed, follow the instructions [here](https://python-poetry.org/docs/#installation).

5.  **Install project dependencies:**
    Poetry will read the `pyproject.toml` file and install all necessary dependencies.
    ```bash
    poetry install
    ```
6.  **Install Stockfish:**
    Stockfish is required for some tests.
    *   **Download Stockfish:** Go to the [Stockfish download page](https://stockfishchess.org/download/) and download the appropriate version for your operating system (Linux).
    *   **Extract the executable:** Extract the downloaded archive.
    *   **Make it accessible:** Add the directory containing the Stockfish executable to your system's PATH environment variable. For example, if you extracted Stockfish to `~/stockfish`, you would add `export PATH="$HOME/stockfish:$PATH"` to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`) and then source it or open a new terminal. Alternatively, you can place the Stockfish executable in a directory already in your PATH (e.g., `/usr/local/bin`).

## Running Tests

Tests are written using `pytest` and can be found in the `tests/` directory. To run the tests:

1.  Make sure your conda environment is activated and you are in the project's root directory.
2.  Run pytest:
    ```bash
    poetry run pytest
    ```

    Or, if you have activated the poetry shell (`poetry shell`):
    ```bash
    pytest
    ```

## Usage

The main entry point for the engine can be found in `src/engine/main.py`. You can run it using:

```bash
poetry run python src/engine/main.py
```
Or, if you have activated the poetry shell:
```bash
python src/engine/main.py
```

You can increase the verbosity of the output:
-   `poetry run python src/engine/main.py -v` for INFO level logging.
-   `poetry run python src/engine/main.py -vv` for DEBUG level logging.
