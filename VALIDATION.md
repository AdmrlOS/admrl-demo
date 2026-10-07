# ANPR validation — 7 October 2026

The `anpr-rk3588` branch is based on the original `rk3588-npu` demo at
`fce8b16d96d47ff5eb4e2396e9ba9ae942baabba`. The original YOLO11 demo is unchanged.

## Build and regression checks

Both native `linux/arm64` builds completed successfully on Apple Silicon Docker
Desktop with the current dashboard, capture settings and capacity benchmark:

```sh
docker buildx build --platform linux/arm64 -f Dockerfile.anpr \
  -t admrl-anpr:rk3588 --load .
docker buildx build --platform linux/arm64 -f Dockerfile.anpr \
  --target cpu-test -t admrl-anpr:cpu-test --load .
```

The model stage exports the pinned plate-trained detector to static ONNX and
compiles both models with RKNN Toolkit2 **2.3.2**, targeting RK3588 FP16 without
INT8 quantization. Source/SDK hashes are checked. The runtime imports RKNNLite
and loads its matching native `librknnrt.so`. Both generated RKNN models,
ONNX sources, character dictionary, public sample, provenance and licence
notices are embedded. Models and preprocessing contracts are unchanged by the
performance work.

**All 76 tests passed** on the host and in the ARM64 CPU test image. They cover
CTC decoding, NMS, boxes/corners, rectification, double-line crops, RGB/BGR
preprocessing, model/capture cleanup and JSON confidence outputs, plus:

- 720p aspect-preserving camera bounds and driver-request/report metadata.
- Timed stages, completion-based FPS, latest-frame capture and CPU policy.
- HTTP assets, official logo MIME/allowlist, Unicode JSON, snapshots, MJPEG
  boundaries and delivery of new frames without a fixed 10 FPS delay.
- Camera/video streaming versus static-image preview selection in the browser.
- Spawned benchmark validation, deadline counting, per-stream target checks,
  CPU selection, NPU load parsing and cleanup escalation.

Actual ONNX detection/cropping/OCR in the rebuilt CPU container returned the
same public-sample outputs, scores and geometry as before:

| OCR output | Detector score | OCR score | Combined score |
| --- | ---: | ---: | ---: |
| `B2V9L7` | 0.897043 | 0.906774 | 0.813415 |
| `EDU4356` | 0.817567 | 0.930309 | 0.760590 |

These are regression outputs, not correct ground-truth labels. Score acceptance
does not establish recognition accuracy. The stock detector's accuracy for
Australian or other local plates has not been established. Earlier actual
inference on the pinned source's double-line sample returned `京EA5331`, exercising
row rearrangement with a real model as well as a unit test.

The runtime `--help` and `--capacity --help` both passed. A real spawned CPU
capacity smoke completed 35 two-plate frames in its two-second measurement with
no errors or forced cleanup; it correctly makes no NPU capacity claim.
Both workflows passed actionlint. The ANPR workflow builds both targets and
repeats the unit suite and public-sample regression.

The branded dashboard was inspected in a real browser. The official logo and
annotated public sample loaded, recognised strings and scores appeared, and the
page had no horizontal overflow or console warnings/errors. A static image
uses the snapshot endpoint so its preview also displays without subsequent
MJPEG frames. Logo provenance is in [web/BRAND-ASSETS.md](web/BRAND-ASSETS.md).

## Actual Admiral RK3588 measurements

The device is running AdmiralOS **1.0.6**, Linux **6.1.115** on ARM64, RKNPU
kernel driver **0.9.8**, and RKNN runtime **2.3.2**. Both FP16 models executed
successfully on the NPU. The bundled public scene consistently produced two
plate candidates on the fixed 720p benchmark input, so every measured positive
frame includes two OCR operations. Numerical OCR output can differ between
FP16 RKNN and ONNX; these tests do not establish a ground-truth accuracy score.

The existing demo was paused during processing-capacity tests and resumed
between runs. Host services and the watchdog were left running. No CPU/NPU
frequency or governor settings were changed. CPU policy `auto` selected the
four A76 cores (kernel capacity 1024), compared with capacity 397 on the four
A55 cores. OpenCV used two threads. The single-worker two-plate comparison
improved from about **6.6 to 7.3 FPS** when using the A76 cluster. Selecting all
three NPU cores for one synchronous model instance did not materially improve
that comparison; independent workers are needed to measure aggregate capacity.

### Processing ceiling

The best configuration sustained **37.083 ANPR frames/sec aggregate** over
**60 seconds**, completing **2,225 frames** after a 10-second warmup. Every frame
contained two detected/OCR crops: **74.167 plate OCR operations/sec**. It used
12 workers, all eight permitted CPUs and one OpenCV thread per worker.

| Workers | CPU policy | OpenCV threads | Timed window | Aggregate FPS |
| ---: | --- | ---: | ---: | ---: |
| 1 | A76 cluster | 2 | 20 s | 7.20 |
| 3 | A76 cluster | 2 | 20 s | 21.60 |
| 6 | A76 cluster | 2 | 20 s | 30.60 |
| 9 | A76 cluster | 2 | 20 s | 33.15 |
| 12 | A76 cluster | 2 | 20 s | 34.40 |
| 18 | A76 cluster | 2 | 20 s | 34.75 |
| 12 | All eight CPUs | 2 | 20 s | 36.70 |
| 18 | All eight CPUs | 2 | 20 s | 36.50 |
| 12 | All eight CPUs | 1 | 20 s | 37.00 |
| 24 | All eight CPUs | 1 | 20 s | 36.20 |
| 12 | All eight CPUs | 1 | **60 s** | **37.083** |

Higher concurrency did not improve throughput. During the sustained run, NPU
cores 0, 1 and 2 averaged **93.1%, 94.0% and 93.7%** load, respectively, with
peaks of 97%. NPU frequency stayed at **1 GHz**; the highest sampled thermal
sensor reached **56.384°C**. Processing latency with 12 workers was
**322.7 ms median / 358.5 ms p95**. The worker pool trades per-frame latency for
aggregate throughput.

This represents approximately **1.236 × 30 FPS** of processing budget. **No
individual tested worker sustained 30 FPS**, so it does not verify a 30-FPS
live camera stream. Live frames would require distribution across workers;
there is no measured margin for two 30-FPS streams with this two-plate workload.

The separate empty-frame control measured **41.7 / 59.75 / 72.6 FPS** with
3 / 6 / 12 workers, respectively, in 20-second windows. Those frames contain
**no detected plates and no OCR operations**; 72.6 FPS is the highest tested
empty-frame rate, rather than the confirmed positive ANPR ceiling. This shows
why a no-plate webcam scene cannot establish full recognition capacity.

The machine-readable results, model/module hashes, configuration rates, stage
timings and NPU/temperature observations are in
[performance/rk3588-720p.json](performance/rk3588-720p.json).

### Webcam and browser delivery

The final 720p webcam run completed **14.118 processed frames/sec** after
discarding 20 warmup frames (280 measured frames). A connected browser received
**14.118 JPEG frames/sec** over a separate 12-second window. The previous
YUYV run reported about **10.2 processed FPS** and delivered **9.7 browser FPS**.
Both views contained **zero detected plates**, so these camera measurements
include detection and preview work but no OCR.

The updated app requested and the driver reported **1280×720 MJPEG at 30 FPS**.
Actual observed capture was **15.08 FPS**, with both captured and processed
frames at 1280×720. Thus the request succeeded but the camera did not demonstrate
30 FPS delivery. Four driver buffers leave space for capture while MJPEG is
decoded; the application still retains only the newest frame for processing.
The browser's fixed 10 FPS delay was removed. Camera exposure controls were
restored after probing, and the final test used their original settings.

The original running app and its source were restored after the temporary
hardware tests. These results describe the candidate image; deploying the
updated image is required to retain its changes on Admiral.

## Scope of the capacity figure

The positive workload is the pinned public two-plate scene, resized with aspect
ratio preserved and padded to exactly **1280×720** before measurement. Detection
uses its unchanged **640×640** model input and OCR its unchanged **48×320** input.
Independent spawned processes assign both model contexts to NPU core 0, 1 or 2
round-robin. Warmup completes for all workers before the shared timed window.
Only fully completed frames inside that window count; deadline-crossing frames
are discarded. Every measured frame must contain the expected plate count.

The figure includes detection, NMS, plate crops, all OCR, annotation and JPEG
quality 80 encoding. Camera capture/decoding, network transmission, initialization,
warmup and preparing the fixed input are excluded. Empty-frame tests exercise
detection and preview work but do no OCR. Throughput depends on models,
precision, plate count, frame content and the pipeline implementation; it is not
a universal silicon limit or a validated camera deployment specification.

Aggregate FPS / 30 expresses a fractional processing budget. A worker must
itself sustain 30 FPS before it counts as a verified 30-FPS stream. The current
synchronous dashboard remains a single worker; the benchmark workers establish
aggregate processing capacity and do not implement live multistream routing.

The command in [README.md](README.md#find-the-device-ceiling) reproduces the
benchmark. Hashes and measurement details are included in the reports. NPU load
is sampled from debugfs when exposed to the container; for this board it was
read on the host and aligned to each measurement window using CLOCK_BOOTTIME.
Another device/BSP or a longer thermal soak still needs its own measurement.
