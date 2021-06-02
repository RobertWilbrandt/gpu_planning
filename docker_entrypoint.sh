#!/bin/bash
set -e

/entrypoint.sh

# Build documentation
cmake --build . --target doc

nvprof /build/gpu_planning -v |& tee /baseline.txt

exec "$@"
