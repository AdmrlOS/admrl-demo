#!/usr/bin/env bash
# Local Docker equivalent of Admiral's OCI workload and hardware passthrough.
set -euo pipefail

image=${ANPR_IMAGE:-admrl-anpr:rk3588}
devices=()
for device in /dev/dri /dev/dma_heap /dev/rknpu /dev/galcore; do
    if [[ -e "$device" ]]; then
        devices+=(--device "$device")
    fi
done
if [[ ${#devices[@]} -eq 0 ]]; then
    echo "No Rockchip NPU device found. Run this helper on the RK3588 board with its RKNPU driver." >&2
    exit 1
fi
if [[ -e /dev/video0 ]]; then
    devices+=(--device /dev/video0)
fi
if [[ ! -f /proc/device-tree/compatible ]]; then
    echo "RKNNLite requires the host's /proc/device-tree/compatible. Use the RK3588 BSP host." >&2
    exit 1
fi

# Mount the current directory at /data for --image /data/car.jpg and results.
exec docker run --rm --init \
    "${devices[@]}" \
    --mount type=bind,src=/proc/device-tree/compatible,dst=/proc/device-tree/compatible,readonly \
    -p "${ANPR_PORT:-8000}:8000" \
    --mount "type=bind,src=$PWD,dst=/data" \
    "$image" "$@"
