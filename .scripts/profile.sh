./build_docker.sh chess-engine linux/arm64 
docker run --rm chess-engine sh -c "poetry run python search_profiler.py"