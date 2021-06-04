#!/bin/bash
set -e

/entrypoint.sh

# Build documentation
cmake --build . --target doc

# Create baseline profile with LTTngtrace information
lttng create baseline
lttng enable-event --userspace gpu_planning:*
lttng start
nvprof /build/gpu_planning -v |& tee /baseline.txt
lttng destroy

exec "$@"
