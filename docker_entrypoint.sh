#!/bin/bash
set -e

/entrypoint.sh

nvprof /build/gpu_planning -v |& tee /baseline.txt

exec "$@"
