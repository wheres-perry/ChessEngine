# Chess Engine

A Python-based chess engine implementation.

## Building

To build the Docker container:

```bash
./build_docker.sh
```

## Running Tests

To run the pytest tests in the Docker container:

```bash
docker run --rm <imagename> ./run_tests.sh
```

## Dependencies

All Python dependencies are automatically installed by the Dockerfile during the build process.