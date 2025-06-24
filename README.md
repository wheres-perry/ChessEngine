# Docker Chess Engine

## Building and Running the Chess Engine

This project is containerized using Docker with a multi-stage build for optimal size. The chess engine includes model weights and is ready to run without additional data.

⚠️ **Note**: Jupyter notebooks and IPython kernels are not included in the production Docker image. If you need to run notebooks for data exploration or model training, you'll need to install the development dependencies locally using `poetry add <package>`.

## Project Structure

- **main.py**: Currently not used for anything in the project. This file exists as a placeholder but has no functionality.
- **node_engine_profile.py**: Performance profiling tool that tests different engine configurations and measures node counts during search. Compares 4 key optimization strategies (no optimizations, alpha-beta only, all optimizations without TT aging, and full optimizations) to analyze search efficiency.

## Running the Chess Engine

Build the Docker image:

```sh
./build_docker.sh chess-engine
```

## Performance Profiling

Run the engine profiler to test different optimization configurations:

```sh
python node_engine_profile.py
```

This will benchmark the chess engine with different optimization settings and show node count comparisons to help identify the most efficient configurations.

## Running Tests

Run the test suite within the Docker container:

```sh
./build_docker.sh chess-engine
docker run -t chess-engine ./run_tests.sh
```

## Running a Specific Test

Run a specific test class or method:

```sh
./build_docker.sh chess-engine
docker run -t chess-engine ./run_tests.sh SomeTest
```

## Interactive Development

For development work requiring Jupyter notebooks or IPython, install dependencies locally:

```sh
poetry install  # Installs all dependencies including dev tools
poetry shell    # Activate the virtual environment
jupyter notebook  # Start Jupyter for data exploration
```

The Docker image is optimized for production use and contains only the essential dependencies needed to run the chess engine.