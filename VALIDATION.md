# ANPR validation — 6 October 2026

Changes are based on the `rk3588-npu` branch at
`fce8b16d96d47ff5eb4e2396e9ba9ae942baabba`, in the `anpr-rk3588` working branch.
The original object detection demo is unchanged.

## Build results

Both of these commands completed successfully on Apple Silicon Docker Desktop
with a native `linux/arm64` Linux builder:

```sh
docker buildx build --platform linux/arm64 -f Dockerfile.anpr \
  -t admrl-anpr:rk3588 --load .

docker buildx build --platform linux/arm64 -f Dockerfile.anpr \
  --target cpu-test -t admrl-anpr:cpu-test --load .
```

The model stage actually exported the plate-trained PyTorch detector to static
ONNX, checked both models with ONNX Runtime, and compiled both to RK3588 FP16
with **RKNN Toolkit2 2.3.2**. It checked source/SDK SHA-256 hashes and Python
dependency consistency. The runtime stage imported RKNNLite and loaded the
native `librknnrt.so` successfully. The final runtime contains both `.rknn`
models, the ONNX originals, dictionary, sample image, provenance/checksum
metadata and licence notices.

A standalone OCI archive was also exported and verified. Its index/manifest
and every referenced config/layer hash match, its platform is `linux/arm64`,
and its embedded runtime source matches the final working tree. The image
manifest digest is
`sha256:e2d639d3d2b2b3d0bba3208ec3ffc4b03401649b618d2932eec5f1e2c526abbb`.
The source patch was checked for whitespace errors and verified to apply to a
clean copy of the original branch revision.

Build issues found and fixed:

- The pinned detector exporter imports additional YOLO utilities; its required
  Python packages are explicitly pinned in `anpr/requirements-build.txt`.
- Rockchip's OCR ONNX file omits output shape metadata. The setup script checks
  its actual output numerically and repairs an in-memory copy for the ONNX
  checker, preserving the downloaded model bytes and hash for conversion.
- RKNNLite's container check needs the host's device-tree compatibility file as
  well as the NPU DRM device. The board run helper and README include that bind.

## Tests and real model inference

The updated CPU container build passed **35 tests** covering CTC blank/repeat decoding,
letterbox inversion, NMS, corner rectification, shuffled corners, double-line
reordering, RGB/BGR preprocessing, malformed output rejection, confidence
reporting, capture/model cleanup, empty video failure, JSON output, HTTP Unicode
results and isolation of native buffered diagnostics from JSON stdout. The
webcam/dashboard additions verify automatic webcam/sample selection, one-shot
image compatibility, HTML/assets/MIME types, allowlisted routes, snapshot and
MJPEG encoding, startup/error states, measured FPS, and newest-frame capture
with exception-safe cleanup.

The rebuilt CPU container's default dashboard was checked in a real browser:
the annotated feed loaded, the two actual OCR candidates and scores appeared,
the page labelled the source as a sample, and preview reconnect worked. No
browser console warnings/errors were reported. Its existing narrow viewport
had no horizontal overflow. HTTP checks against the running container verified
the HTML, JavaScript, CSS, JSON, snapshot and multipart image stream endpoints.
Three frames from an actual OpenCV-decoded test video also passed through
detection/cropping/OCR and emitted frame records 0, 1 and 2.

Actual ONNX detection, perspective cropping and OCR also completed successfully
inside the ARM64 CPU container:

```sh
docker run --rm admrl-anpr:cpu-test \
  --backend onnx --image /opt/models/sample.jpg
```

The fixed bundled scene produces two candidates:

| OCR output | Detector score | OCR score | Combined score |
| --- | ---: | ---: | ---: |
| `B2V9L7` | 0.897043 | 0.906774 | 0.813415 |
| `EDU4356` | 0.817567 | 0.930309 | 0.760590 |

These are reproducible regression outputs, **not correct ground-truth plate
labels**: the general recognizer omits the first plate's Chinese province
prefix and reads the second prefix as `E`. `accepted` means the configured
score threshold passed. It does not validate the plate text. Camera/domain
validation or plate-specific training is needed before relying on recognition.

A separate real inference check against the pinned detector source's
`imgs/double_yellow.jpg` classified a double-line plate, rearranged its rows and
returned `京EA5331` with OCR score `0.9552` and detector score `0.8881`; the
annotated image was visually checked. This exercises double-line handling with
a real model as well as the synthetic unit test.

To rerun the unit suite explicitly after building the CPU image:

```sh
docker run --rm --entrypoint /venv/bin/python admrl-anpr:cpu-test \
  -m unittest discover -s /opt/anpr/tests -v
```

The ANPR CI workflow repeats both builds, the unit suite, and the bundled scene
regression with checks for strings, accepted flags, score ranges and box bounds.
It does not publish an image.

## Exact hardware blocker

This environment is an Apple Silicon desktop, with no RK3588 NPU or RKNPU
driver. Running the default RKNN image against the bundled image returned exit
code **1** with **zero bytes on stdout**. RKNNLite failed in its container check
because `/dev/dri/renderD129` and `/proc/device-tree/compatible` are absent,
then the application printed the required host mount/BSP guidance to stderr.
No CPU fallback or fabricated RKNN recognition result was emitted.

NPU inference correctness, BSP/runtime compatibility, live camera capture and
real-time throughput are therefore **unverified**. The CPU image checks are
not NPU benchmarks. On the Admiral board, preserve the OCI entrypoint, pass
through the NPU devices and compatibility file described in README, then run:

```sh
./scripts/run-anpr.sh --image /opt/models/sample.jpg
./scripts/run-anpr.sh --image /data/car.jpg --output-dir /data/results
./scripts/run-anpr.sh --source 0 --serve
```

Compare the first command with the CPU result, inspect your own plate images,
and use per-frame `elapsed_ms` to assess complete pipeline latency on the board.
The host must supply the working Rockchip BSP driver; the image does not
install a kernel driver.
