# RK3588 NPU demos on Admiral

The original `Dockerfile` and `detect_stream.py` build the YOLO11 object detection
stream demo. The separate `Dockerfile.anpr` builds a licence plate recognition
demo using the same Debian Bookworm / Python 3.11 / RKNNLite conventions.

## Licence plate recognition

```text
image / camera / video / RTSP
  -> plate-trained YOLOv5 detector on RK3588 NPU
  -> confidence filter + NMS + map boxes to the processed frame
  -> perspective crop using four plate corners
  -> PP-OCRv4 recognition on RK3588 NPU
  -> CTC decoding -> JSON plate strings and confidence
```

The image embeds the models and sample image. There are no downloads at device
startup. OpenCV handles capture, resizing and cropping; this demo does not use
MPP/RGA acceleration or tracking. Measure end-to-end latency on your actual board
and camera before setting a frame-rate target.

## Build

From the repository root, on a native ARM64 Docker builder (including Apple
Silicon Docker Desktop):

```sh
docker buildx build --platform linux/arm64 -f Dockerfile.anpr \
  -t admrl-anpr:rk3588 --load .
```

An x86 builder needs ARM64 emulation or a remote ARM64 builder. The build compiles
both models ahead of time with RKNN Toolkit2; it does not need an NPU. The final
image uses RKNNLite and the matching native `librknnrt.so`.

To export an OCI archive instead of loading a Docker image:

```sh
docker buildx build --platform linux/arm64 -f Dockerfile.anpr \
  -t admrl-anpr:rk3588 \
  --output type=oci,dest=admrl-anpr-rk3588.oci.tar .
```

## Run on an Admiral RK3588

The default container starts a small dashboard on port **8000** and uses the
webcam at `/dev/video0` when it is present. With no webcam it displays the
bundled sample image and labels it as a sample. Deploy the OCI image as an
Admiral workload with its default entrypoint/arguments and port 8000 reachable.
The AdmiralOS board used for validation has `crun`, without a Docker daemon.
On an RK3588 Linux host with Docker, the equivalent helper is:

```sh
./scripts/run-anpr.sh
```

Open `http://DEVICE_IP:8000` in a browser. The dashboard shows the annotated
camera feed, current recognised strings with detector/OCR scores, recent
reads, processing rate, latency and connection state. It uses the camera
connected to the Admiral device; the viewing browser does not need camera
permissions. All page assets are served by the container, so it works offline.

The `Build and Push Docker Image` workflow publishes pushes to `anpr-rk3588`
using `Dockerfile.anpr` as:
`ghcr.io/admrlos/admrl-demo/systemd-container:anpr-rk3588`.
Use that registry image in the Admiral workload after the workflow succeeds.
On a Docker-equipped RK3588 host:

```sh
ANPR_IMAGE=ghcr.io/admrlos/admrl-demo/systemd-container:anpr-rk3588 ./scripts/run-anpr.sh
```

Other branches and tag refs use the original `Dockerfile`. The separate ANPR
validation workflow builds and tests without publishing.

To publish a local build manually, tag and push to a registry you can access:

```sh
docker tag admrl-anpr:rk3588 ghcr.io/admrlos/admrl-demo/anpr:rk3588
docker push ghcr.io/admrlos/admrl-demo/anpr:rk3588
```

These commands publish under the Admiral organisation; substitute your own
registry/repository when appropriate.

Deploy the ARM64 OCI image through the Admiral dashboard/API, preserving its
entrypoint. Allow access to the host's Rockchip NPU device and allocation devices
and the camera when used. Bind the host's `/proc/device-tree/compatible` into the
same path in the container: RKNNLite uses it to identify the SoC and its
container check also expects the NPU DRM node (normally `/dev/dri/renderD129`).
On deployments using Admiral's `IsolateDevices`
setting, use `IsolateDevices=false` for this hardware demo. Make TCP port 8000
reachable when starting with `--serve`. The host supplies the RKNPU kernel
driver; the container supplies the userspace runtime. Use the same BSP/device
access configuration as the existing working YOLO demo.

For a local Docker test on the board, the helper maps existing `/dev/dri`,
`/dev/dma_heap`, `/dev/rknpu`, `/dev/galcore` and `/dev/video0` devices, binds the
host's device-tree compatibility file, and mounts the current directory at
`/data`:

```sh
# Bundled image test.
./scripts/run-anpr.sh --image /opt/models/sample.jpg

# Your own image, with an annotated image saved locally.
./scripts/run-anpr.sh --image /data/car.jpg --output-dir /data/results

# USB camera, matching the original demo's stream on port 8000.
./scripts/run-anpr.sh --source 0 --serve

# Video or an RTSP stream; finite files stop at EOF unless --loop is set.
./scripts/run-anpr.sh --source /data/traffic.mp4 --max-frames 100 --serve
./scripts/run-anpr.sh --source rtsp://camera.example/stream --serve
```

Set `ANPR_IMAGE` to a registry image name and `ANPR_PORT` to change the helper's
published host port. For another camera device, map it explicitly and use its
path as `--source`.

With `--serve`, `/` returns the dashboard, `/stream.mjpg` returns the annotated
MJPEG stream, `/snapshot.jpg` returns the current annotated JPEG, and `/results`
returns the latest frame's JSON result and runtime state. Standard output contains one JSON result per
processed image/frame; runtime diagnostics go to standard error. An empty plate
list means no plate passed the detector threshold. Each candidate has an
`accepted` flag indicating nonempty text above the OCR threshold. Startup,
model-loading, capture and inference failures return a nonzero exit status.

The web server starts before the models load, so the page can show startup
progress. Pipeline failures are briefly published to the dashboard before the
process exits. The page marks interrupted connections or frames that stop
updating, and retains the last result for inspection.

Camera and RTSP capture continuously keep only the newest frame while inference
runs. This prevents an old-frame backlog when OCR is slower than camera capture.
Webcams request **1280×720, MJPEG, 30 fps** by default. The backend's reported
settings, received dimensions and processing rate appear on the dashboard.
The JSON API also reports the observed capture rate.
Four driver buffers keep capture moving during MJPEG decoding; inference still
receives only the latest available frame.
If the driver ignores the resolution request, camera frames are downscaled to
fit 1280×720 before detection, crops and preview encoding, keeping their aspect
ratio. Smaller camera frames are not enlarged. Image and video file dimensions
are preserved. The MJPEG endpoint sends each newly processed preview without
the original fixed 10 fps delay.

Override capture settings with `--camera-width`, `--camera-height`,
`--camera-fps` and `--camera-fourcc`. For a camera without MJPEG support, use
`--camera-fourcc auto` to keep its default format. A camera may deliver less
than the requested frame rate; the measured processing rate also depends on
plate count, model execution and CPU work.

Displayed FPS measures processed-frame throughput and stays blank for a still
image. The browser's recent-read history holds up to 20 distinct strings and
resets when the page reloads; it does not perform tracking or temporal consensus.

Confidence values are model scores, not calibrated probabilities of a correct
plate. Detection and OCR confidence are returned separately so consumers can
choose thresholds suitable for their camera. Consult `--help` for all options.

Each frame record contains `source`, `frame`, `elapsed_ms` and a `plates` array,
plus `frame_size`, `timings_ms`, measured `fps`, `processed_at` (a monotonic
completion timestamp) and `processing_interval_ms`. Camera records include
`camera.requested`, `camera.negotiated`, `camera.settings_supported` and
`camera.observed`; a successful property setter only indicates backend support.
The observed dimensions and capture rate come from received frames.
The `/results` API also includes `status`, `mode`, `backend`, `sequence`,
`timestamp`, and measured `fps`. Camera frame indices may skip as older captured
frames are replaced by newer ones. Camera URL credentials are redacted from
results and dashboard source labels.
Each plate contains the raw Unicode `text`, a space-stripped `plate`, `bbox`
coordinates `[x0,y0,x1,y1]` in processed-frame pixels, `corners`, `layout`,
`detection_confidence`, `recognition_confidence`, their product `confidence`, and
`accepted`. No region-specific substitutions are applied to the OCR text.

## Measuring performance on the board

Keep the same view and plate count when comparing runs. Capture 20 warmup frames
and 200 measured frames with the dashboard enabled, so preview work is included:

```sh
./scripts/run-anpr.sh --source 0 --serve --max-frames 220 > camera-720p.jsonl
python3 scripts/benchmark_anpr.py camera-720p.jsonl --warmup 20 > camera-720p-summary.json
```

The summary reports actual completion throughput, median/p95 stage timings,
received/processed sizes and the number of plates in each frame. It never
calculates FPS by taking the inverse of inference latency. Capture runs on its
own thread, so capture-read time is reported separately from processing time.
Keep the live dashboard open during the run if comparing browser streaming.
If the demo is already using the camera, stop that workload before running a
second camera benchmark.

`--npu-cores auto` uses the Rockchip automatic single-core assignment.
`--npu-cores all` uses all three RK3588 NPU cores with the same FP16 models;
individual cores are selectable with `0`, `1` or `2`. Compare the same scene
and plate count before choosing a setting. Lower camera resolution reduces CPU
and USB work; the detector still runs at its fixed 640×640 model input.

### Find the device ceiling

The tested RK3588 sustained **37.083 full ANPR frames/sec aggregate** on a
fixed 720p scene with two detected plates per frame, using all three NPU cores.
That is **1.236 × 30 FPS** of processing budget, before capture, decoding and
network costs. The single-worker webcam test reached **14.118 FPS** in a scene
with no plates, while the camera delivered about 15 FPS. These are different
workloads; see [the measured results and limits](VALIDATION.md).

Run the saturated benchmark with the usual demo stopped so it does not compete
for NPU time. For an Admiral workload, supply `--capacity ...` arguments in
place of the default `--serve` arguments, then restore `--serve` for the camera
dashboard. Inside an ANPR container with no competing inference:

```sh
/venv/bin/python /opt/anpr/anpr.py --capacity \
  --streams 1 3 6 12 18 24 --workload plates --duration 20 --warmup 5 \
  --cpu-cores all --opencv-threads 1 > capacity-plates.json

# Repeat the best concurrency for a longer measurement.
/venv/bin/python /opt/anpr/anpr.py --capacity \
  --streams 12 --workload plates --duration 60 --warmup 10 \
  --cpu-cores all --opencv-threads 1 > capacity-plates-sustained.json

# A separate detection-only workload; no plates means no OCR work.
/venv/bin/python /opt/anpr/anpr.py --capacity \
  --streams 3 6 12 --workload empty --duration 20 --warmup 5 \
  --cpu-cores all --opencv-threads 1 > capacity-empty.json
```

`--capacity` must be the first argument. Each worker is an independent spawned
process with its detector and recognizer pinned to one NPU core; workers rotate
across cores 0, 1 and 2. Inputs are exactly 1280×720. The public two-plate sample
is resized with its aspect ratio preserved and padded before timing. The
`plates` workload requires two detected plates on every measured frame; `empty`
requires zero. Model execution, every plate crop/OCR, annotation and JPEG quality
80 encoding are included. Capture, camera decoding, network transmission, input
preparation, initialization and warmup are excluded.

The report counts completed frames in a shared measurement window, checks
worker cleanup, records hashes and CPU affinity, and reports aggregate FPS,
per-worker FPS and median/p95 stage times. `/sys/kernel/debug/rknpu/load` is
sampled when visible; otherwise load statistics are explicitly unavailable.
Increasing workers until aggregate FPS plateaus finds the processing ceiling
for these models and workloads. The `--target-fps` default is 30: aggregate
FPS / 30 is a fractional processing budget, while a verified stream count
requires every tested worker to sustain 30 FPS individually. The synchronous
webcam demo remains a single worker; the benchmark does not turn it into a
multistream scheduler.

By default, RKNN processes use the highest-capacity CPU cluster exposed by the
kernel and two OpenCV threads. On the tested RK3588 this selects its four A76
cores; it improves the single-worker two-plate workload. This only changes the
application's affinity within its allowed CPUs. Use `--cpu-cores all` to retain
the inherited mask, or explicit CPU IDs, and `--opencv-threads` to compare
scheduling. All eight CPUs performed better for the saturated benchmark; the
commands above use that setting. No clocks or governors are changed. ONNX keeps
its inherited CPUs with the default `auto` policy. Recorded CPU capacity and actual affinity make
these comparisons reproducible.

Actual board measurements and their scope are in [VALIDATION.md](VALIDATION.md).

## Models and repeatability

The detector comes from
[we0091234/Chinese_license_plate_detection_recognition](https://github.com/we0091234/Chinese_license_plate_detection_recognition/tree/670b765c09fff11850a7b1c542787ad66c24ae50).
It predicts bounding boxes, four plate corners and single/double-line classes.
Recognition uses the
[Rockchip PP-OCRv4 example](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/PPOCR/PPOCR-Rec)
with its matching character dictionary. This provides Latin letters/digits as
well as Chinese characters. LPRNet's stock Chinese plate alphabet and fixed
plate assumptions make PP-OCR the more flexible choice for this demo.

**The stock detector was trained for Chinese plates.** Its accuracy on other
plate styles, including Australian plates, has not been established. PP-OCR is
a general text recognizer, and plate accuracy also depends on crop quality,
angle, glare and character size. Use representative local footage to assess the
demo, then fine-tune/replace the detector and recognizer for your deployment.
There is no country-specific plate validation or temporal consensus in this
first demo.

Download URLs, Git revisions and SHA-256 checksums are checked into the model
setup files. The converter targets `rk3588`, uses static input shapes, and keeps
FP16 models to avoid an unrepresentative INT8 calibration dataset. Changing
weights requires retaining the documented tensor and preprocessing contracts,
reconverting the models, and rebuilding or mounting the model directory.

The setup script is `scripts/setup_anpr_models.py`. Without `--convert` it
downloads the pinned sources and exports/checks the ONNX models; with `--convert`
it also generates RK3588 FP16 files. `--download-only` fetches and checks the
sources without requiring PyTorch or RKNN; `--cache-dir` preserves checked
downloads between runs. The container build performs this complete
step. To obtain the prepared files separately without setting up the compiler
on your host:

```sh
docker buildx build --platform linux/arm64 -f Dockerfile.anpr \
  --target model-build -t admrl-anpr:models --load .
docker create --name anpr-model-export admrl-anpr:models
docker cp anpr-model-export:/opt/models ./models
docker rm anpr-model-export
```

`models/model_metadata.json` records source revisions, input/output shapes,
preprocessing and checksums of the generated artifacts. The detector contract
is one decoded output `[1,25200,15]` (box, objectness, eight corner coordinates,
two class scores). A generic COCO model or raw YOLO feature-map export is not a
drop-in replacement. The recognizer contract is `[1,40,6625]` softmax output with
the supplied dictionary, CTC blank at index zero, and a final space character.
Both backends use matching `/255` normalization; the detector consumes RGB and
the recognizer consumes BGR. Double-line plates are rearranged into one row.

For example, override all prepared files by adding
`--mount type=bind,src="$PWD/models",dst=/opt/models,readonly` to `docker run`,
or use `--models-dir` with another mounted location. `--detector`,
`--recognizer`, and `--dictionary` also accept individual paths. Input dimensions
must match the converted models.

Model/source license notices are included with the prepared assets. The
detector upstream is GPL-3.0; the Rockchip/PaddleOCR example is Apache-2.0 and the
Rockchip SDK carries its own license. Source links are recorded in
`scripts/model_sources.json`.

## Verification and hardware limits

The CPU validation image and tests verify decoding, crop geometry and the actual
ONNX pipeline without a Rockchip board. CPU inference uses ONNX Runtime only
when explicitly selected; the default runtime requires the NPU. Build and
validation commands and the results of this implementation are in
[VALIDATION.md](VALIDATION.md).

NPU inference and processing throughput were measured on an Admiral RK3588;
see [VALIDATION.md](VALIDATION.md) for the hardware results and workload limits.
A successful build or CPU result alone does not verify NPU performance on another
device. Missing/incompatible NPU access fails at startup.

## Original YOLO11 demo

```sh
docker buildx build --platform linux/arm64 -f Dockerfile \
  -t admrl-yolo11:rk3588 --load .
```

The existing workflow and object detection demo continue to use `Dockerfile`.
