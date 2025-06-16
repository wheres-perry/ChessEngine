# Docker Chess Engine

## Building and Running the Chess Engine

This project is containerized using Docker with a multi-stage build for optimal size. The chess engine includes model weights and is ready to run without additional data.

⚠️ **Note**: Jupyter notebooks and IPython kernels are not included in the production Docker image. If you need to run notebooks for data exploration or model training, you'll need to install the development dependencies locally using `poetry add <package>`.

## Running the Chess Engine

Build the Docker image and run the chess engine:

```sh
./build_docker.sh chess_engine
docker run -t chess_engine
```

## Running Tests

Run the test suite within the Docker container:

```sh
./build_docker.sh chess_engine
docker run -t chess_engine ./run_tests.sh
```

## Running a Specific Test

Run a specific test class or method:

```sh
./build_docker.sh chess_engine
docker run -t chess_engine ./run_tests.sh TestChessEngine.test_move_generation
```

## Interactive Development

For development work requiring Jupyter notebooks or IPython, install dependencies locally:

```sh
poetry install  # Installs all dependencies including dev tools
poetry shell    # Activate the virtual environment
jupyter notebook  # Start Jupyter for data exploration
```

The Docker image is optimized for production use and contains only the essential dependencies needed to run the chess engine.