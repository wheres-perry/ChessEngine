#!/bin/sh

DOCKER_TAG=${1:-chess-engine}
DOCKER_DEFAULT_PLATFORM=${2:-linux/amd64}

docker build --platform $DOCKER_DEFAULT_PLATFORM -t $DOCKER_TAG .
 