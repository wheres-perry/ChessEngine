# Docker Chess Engine

## Building and Running the Chess Engine

This project is containerized using Docker with a multi-stage build for optimal size. The chess engine includes model weights and is ready to run without additional data.

⚠️ **Note**: Jupyter notebooks and IPython kernels are not included in the production Docker image. If you need to run notebooks for data exploration or model training, you'll need to install the development dependencies locally using `poetry add <package>`.

## Project Structure

- **main.py**: Currently not used for anything in the project. This file exists as a placeholder but has no functionality.
- **search_profiler.py**: Simplified chess engine profiling tool that tests different engine configurations by measuring node counts during search. Compares base (no optimizations) vs all optimizations (and any added configs) to analyze search efficiency.

## Running the Chess Engine

Build the Docker image:

```sh
./build_docker.sh chess-engine
```

## Performance Profiling

- Run the search profiler to test different engine configurations:

```sh
python search_profiler.py
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
