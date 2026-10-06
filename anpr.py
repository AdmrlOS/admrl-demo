#!/usr/bin/env python3
"""Plate YOLOv5 landmarks -> rectified crop -> Rockchip PP-OCRv4 CTC.

The paired model conversion uses mean=0, std=255 for uint8 input. Detector
input is RGB; the official PP-OCR ONNX input is BGR in [0,1]. Keep these
conventions identical on both backends.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import ctypes
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np


_VENDOR_OUTPUT_LOCK = threading.Lock()
_FLUSH_C_STREAMS = ctypes.CDLL(None).fflush


@contextmanager
def vendor_output():
    """Keep Python and buffered native RKNN diagnostics off JSON stdout."""
    with _VENDOR_OUTPUT_LOCK:
        sys.stdout.flush()
        _FLUSH_C_STREAMS(None)
        saved_stdout = os.dup(1)
        try:
            os.dup2(2, 1)
            with redirect_stdout(sys.stderr):
                yield
        finally:
            # C stdio may buffer logs until a later call; flush before restoring
            # fd 1, including when initialization or inference raises.
            _FLUSH_C_STREAMS(None)
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)


@dataclass
class Detection:
    box: np.ndarray
    corners: np.ndarray
    confidence: float
    class_id: int


def letterbox(frame, size=640):
    """Return uint8 RGB input plus reversible scales/padding."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - new_w) // 2, (size - new_h) // 2
    image = np.full((size, size, 3), 114, dtype=np.uint8)
    image[top:top + new_h, left:left + new_w] = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (new_w, new_h))
    return image, (new_w / width, new_h / height, left, top)


def nms(boxes, scores, threshold):
    """Class-agnostic NMS: single/double-row predictions share the same plate."""
    order = scores.argsort()[::-1]
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        rest = order[1:]
        lo = np.maximum(boxes[index, :2], boxes[rest, :2])
        hi = np.minimum(boxes[index, 2:], boxes[rest, 2:])
        intersection = np.prod(np.maximum(0, hi - lo), axis=1)
        union = areas[index] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = rest[iou <= threshold]
    return keep


def decode_detections(outputs, transform, frame_shape, threshold=0.35,
                      iou_threshold=0.45, max_plates=20):
    if len(outputs) != 1:
        raise ValueError("Detector requires the exported decoded YOLO output [1,N,15].")
    prediction = np.asarray(outputs[0], dtype=np.float32)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2 or prediction.shape[1] != 15:
        raise ValueError(f"Unsupported detector output {prediction.shape}; expected [1,N,15].")
    prediction = prediction[np.isfinite(prediction).all(axis=1)]
    class_ids = np.argmax(prediction[:, 13:], axis=1)
    scores = prediction[:, 4] * prediction[np.arange(len(prediction)), 13 + class_ids]
    selected = (scores >= threshold) & (prediction[:, 2:4] > 0).all(axis=1)
    prediction, scores, class_ids = prediction[selected], scores[selected], class_ids[selected]
    if not len(prediction):
        return []
    boxes = np.concatenate((prediction[:, :2] - prediction[:, 2:4] / 2,
                            prediction[:, :2] + prediction[:, 2:4] / 2), axis=1)
    corners = prediction[:, 5:13].reshape(-1, 4, 2).copy()
    sx, sy, left, top = transform
    height, width = frame_shape[:2]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / sx, 0, width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / sy, 0, height)
    corners[:, :, 0] = np.clip((corners[:, :, 0] - left) / sx, 0, width - 1)
    corners[:, :, 1] = np.clip((corners[:, :, 1] - top) / sy, 0, height - 1)
    valid = (boxes[:, 2:] - boxes[:, :2] >= 2).all(axis=1)
    boxes, corners, scores, class_ids = boxes[valid], corners[valid], scores[valid], class_ids[valid]
    return [Detection(boxes[i], corners[i], float(scores[i]), int(class_ids[i]))
            for i in nms(boxes, scores, iou_threshold)[:max_plates]]


def crop_plate(frame, detection, split_double=True):
    """Order corners as upstream does, rectify them, and fall back to bbox."""
    raw = detection.corners.astype(np.float32)
    sums, differences = raw.sum(axis=1), np.diff(raw, axis=1).ravel()
    points = raw[[np.argmin(sums), np.argmin(differences),
                  np.argmax(sums), np.argmax(differences)]]
    width = round(max(np.linalg.norm(points[1] - points[0]),
                      np.linalg.norm(points[2] - points[3])))
    height = round(max(np.linalg.norm(points[3] - points[0]),
                       np.linalg.norm(points[2] - points[1])))
    convex = cv2.isContourConvex(points.reshape(-1, 1, 2))
    if convex and abs(cv2.contourArea(points)) >= 4 and width >= 2 and height >= 2:
        width, height = min(width, 2048), min(height, 1024)
        target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        crop = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(points, target),
                                   (width, height), borderMode=cv2.BORDER_REPLICATE)
    else:
        x0, y0 = np.floor(detection.box[:2]).astype(int)
        x1, y1 = np.ceil(detection.box[2:]).astype(int)
        crop = frame[max(0, y0):min(frame.shape[0], y1),
                     max(0, x0):min(frame.shape[1], x1)].copy()
    if crop.size == 0:
        raise ValueError("Plate crop is empty after clipping to the frame.")
    # Upstream double-row plate rule: overlap the two row crops slightly, then
    # concatenate them horizontally so the CTC recognizer sees one text line.
    if split_double and detection.class_id == 1 and crop.shape[0] >= 6:
        height = crop.shape[0]
        upper, lower = crop[:height * 5 // 12], crop[height // 3:]
        crop = np.concatenate((cv2.resize(upper, (lower.shape[1], lower.shape[0])), lower), axis=1)
    return crop


def ocr_input(crop, width=320, height=48):
    # The Rockchip PPOCR-Rec example stretches BGR crops to this fixed shape.
    return cv2.resize(crop, (width, height))


def load_characters(path):
    characters = Path(path).read_text(encoding="utf-8").splitlines()
    if not characters or any(not character for character in characters):
        raise ValueError("OCR dictionary is empty or contains blank lines.")
    return [""] + characters + [" "]  # CTC blank index 0; Paddle use_space_char.


def decode_ctc(output, characters):
    probabilities = np.asarray(output, dtype=np.float32)
    if probabilities.ndim == 3 and probabilities.shape[0] == 1:
        probabilities = probabilities[0]
    if probabilities.ndim != 2 or probabilities.shape[1] != len(characters):
        raise ValueError(f"OCR output {probabilities.shape} does not match {len(characters)} dictionary classes.")
    if not np.isfinite(probabilities).all() or probabilities.shape[0] == 0:
        raise ValueError("OCR returned invalid or empty probabilities.")
    # This PP-OCR export includes softmax. Reject a mismatched/logit model
    # instead of silently reporting meaningless confidence values.
    if np.min(probabilities) < -1e-4 or np.max(probabilities) > 1.0001:
        raise ValueError("OCR model must output softmax probabilities, not logits.")
    indices = probabilities.argmax(axis=1)
    text, scores, previous = [], [], -1
    for timestep, index in enumerate(indices):
        index = int(index)
        if index != 0 and index != previous:
            text.append(characters[index])
            scores.append(float(probabilities[timestep, index]))
        previous = index
    return "".join(text), float(np.mean(scores)) if scores else 0.0


class Model:
    def __init__(self, path, backend):
        self.backend = backend
        self.runtime = None
        if not Path(path).is_file():
            raise FileNotFoundError(f"Model missing: {path}. Run scripts/setup_anpr_models.py first.")
        print(f"Loading {backend} model: {path}", file=sys.stderr, flush=True)
        if backend == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError("CPU backend needs onnxruntime; install it in your Python environment.") from error
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.runtime = ort.InferenceSession(str(path), sess_options=options,
                                               providers=["CPUExecutionProvider"])
            self.input_name = self.runtime.get_inputs()[0].name
        else:
            try:
                with vendor_output():
                    from rknnlite.api import RKNNLite
            except ImportError as error:
                raise RuntimeError("RKNNLite is unavailable; use the arm64 Admiral image or --backend onnx.") from error
            try:
                with vendor_output():
                    self.runtime = RKNNLite()
                    if self.runtime.load_rknn(str(path)) != 0:
                        raise RuntimeError(f"Could not load {path}; check model conversion and runtime versions.")
                    if self.runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                        raise RuntimeError("NPU initialization failed; use an RK3588 host with a working NPU BSP, "
                                           "pass its /dev/dri devices (or /dev/rknpu), and bind "
                                           "/proc/device-tree/compatible read-only into the container.")
            except BaseException:
                self.close()
                raise

    def infer(self, image):
        if self.backend == "onnx":
            tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            output = self.runtime.run(None, {self.input_name: tensor})
        else:
            with vendor_output():
                output = self.runtime.inference(inputs=[np.ascontiguousarray(image[None])], data_format=["nhwc"])
        if output is None or not len(output):
            raise RuntimeError("Model inference returned no tensors.")
        return output

    def close(self):
        if self.runtime is not None:
            if self.backend == "rknn":
                with vendor_output():
                    self.runtime.release()
            self.runtime = None


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.detector = self.recognizer = None
        extension = ".onnx" if args.backend == "onnx" else ".rknn"
        directory = Path(args.models_dir)
        self.characters = load_characters(args.dictionary or directory / "ppocr_keys_v1.txt")
        try:
            self.detector = Model(args.detector or directory / ("plate_detector" + extension), args.backend)
            self.recognizer = Model(args.recognizer or directory / ("plate_recognizer" + extension), args.backend)
        except BaseException:
            self.close()
            raise

    def run(self, frame):
        start = time.perf_counter()
        image, transform = letterbox(frame, self.args.detector_size)
        detections = decode_detections(self.detector.infer(image), transform, frame.shape,
                                       self.args.threshold, self.args.iou, self.args.max_plates)
        plates = []
        for detection in detections:
            crop = crop_plate(frame, detection, not self.args.no_split_double)
            predictions = self.recognizer.infer(ocr_input(crop, self.args.ocr_width, self.args.ocr_height))
            if len(predictions) != 1:
                raise ValueError("Recognizer requires one [1,T,classes] probability tensor.")
            text, score = decode_ctc(predictions[0], self.characters)
            plates.append({"text": text, "plate": text.replace(" ", ""),
                           "confidence": detection.confidence * score,
                           "detection_confidence": detection.confidence,
                           "recognition_confidence": score,
                           "accepted": bool(text.strip()) and score >= self.args.ocr_threshold,
                           "bbox": [round(float(value), 2) for value in detection.box],
                           "corners": detection.corners.round(2).tolist(),
                           "layout": "double" if detection.class_id == 1 else "single"})
        return {"plates": plates, "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)}

    def close(self):
        for model in (self.recognizer, self.detector):
            if model is not None:
                model.close()


def annotate(frame, result):
    frame = frame.copy()
    for plate in result["plates"]:
        x0, y0, x1, y1 = map(int, plate["bbox"])
        color = (0, 220, 0) if plate["accepted"] else (0, 150, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        # OpenCV's built-in font cannot draw Chinese; JSON retains full Unicode.
        label = plate["plate"].encode("ascii", "replace").decode("ascii")
        cv2.putText(frame, f"{label} {plate['recognition_confidence']:.2f}", (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def redact_source(source):
    value = str(source)
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    query = urlencode([(key, "***" if key.lower() in {"password", "token", "access_token", "api_key", "key", "auth"} else value)
                       for key, value in parse_qsl(parsed.query, keep_blank_values=True)])
    return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, query, parsed.fragment))


def default_webcam_source():
    return "0" if Path("/dev/video0").exists() else None


def source_mode(source):
    value = str(source)
    return "camera" if value.isdecimal() or value.startswith("/dev/video") else "video"


class Preview:
    def __init__(self, stop, mode="image", source="", backend="rknn", web_dir=None):
        self.stop, self.lock = stop, threading.Lock()
        self.web_dir = Path(web_dir) if web_dir else Path(__file__).parent / "web"
        self.jpeg, self.last_processed = None, None
        self.result = {"status": "loading", "mode": mode, "source": redact_source(source),
                       "backend": backend, "frame": None, "sequence": 0,
                       "timestamp": self.timestamp(), "fps": 0.0, "elapsed_ms": None, "plates": []}

    @staticmethod
    def timestamp():
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def update_state(self, status, error=None):
        if status not in {"loading", "live", "sample", "error", "stopped"}:
            raise ValueError(f"Unsupported preview status: {status}")
        with self.lock:
            self.result = dict(self.result, status=status, sequence=self.result["sequence"] + 1,
                               timestamp=self.timestamp())
            if error is not None:
                self.result["error"] = str(error)
            elif status != "error":
                self.result.pop("error", None)

    def publish(self, frame, result):
        okay, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not okay:
            raise RuntimeError("Failed to encode the dashboard preview.")
        now = time.perf_counter()
        with self.lock:
            status = "sample" if self.result["mode"] in {"image", "sample"} else "live"
            if status == "sample" or self.last_processed is None:
                fps = 0.0
            else:
                instantaneous = 1 / max(now - self.last_processed, 1e-6)
                fps = instantaneous if not self.result["fps"] else 0.8 * self.result["fps"] + 0.2 * instantaneous
            self.last_processed = now
            self.result = dict(self.result, **result)
            self.result.update(status=status, source=redact_source(self.result["source"]),
                               sequence=self.result["sequence"] + 1, timestamp=self.timestamp(), fps=round(fps, 2))
            self.result.pop("error", None)
            self.jpeg = jpeg.tobytes()

    def handler(self):
        preview = self
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/index.html": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_body(self, body, content_type):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlsplit(self.path).path
                try:
                    if path == "/results":
                        with preview.lock:
                            body = json.dumps(preview.result, ensure_ascii=False).encode("utf-8")
                        self.send_body(body, "application/json; charset=utf-8")
                    elif path == "/snapshot.jpg":
                        with preview.lock:
                            jpeg = preview.jpeg
                        if jpeg is None:
                            self.send_error(503, "No preview frame available yet")
                        else:
                            self.send_body(jpeg, "image/jpeg")
                    elif path in {"/stream.mjpg", "/video"}:
                        self.send_response(200)
                        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=jpgboundary")
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        while not preview.stop.is_set():
                            with preview.lock:
                                jpeg = preview.jpeg
                            if jpeg is not None:
                                self.wfile.write(b"--jpgboundary\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                                self.wfile.flush()
                            preview.stop.wait(0.1)
                    elif path in assets:
                        name, content_type = assets[path]
                        try:
                            body = (preview.web_dir / name).read_bytes()
                        except OSError:
                            self.send_error(503, "Dashboard asset unavailable")
                        else:
                            self.send_body(body, content_type)
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler


class LatestCapture:
    """Continuously drain a live camera/RTSP stream; hold only its latest frame."""
    def __init__(self, capture, stop, source):
        self.capture, self.stop, self.source = capture, stop, redact_source(source)
        self.condition, self.closed = threading.Condition(), threading.Event()
        self.latest, self.sequence, self.error, self.finished = None, -1, None, False
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while not self.stop.is_set() and not self.closed.is_set():
                okay, frame = self.capture.read()
                if not okay:
                    raise RuntimeError(f"Source stopped returning frames: {self.source}")
                with self.condition:
                    self.latest = frame
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
        finally:
            try:
                self.capture.release()
            except Exception as error:
                with self.condition:
                    if self.error is None:
                        self.error = error
            finally:
                with self.condition:
                    self.finished = True
                    self.condition.notify_all()

    def read(self, after):
        with self.condition:
            while not self.stop.is_set() and not self.closed.is_set():
                if self.sequence > after:
                    return self.sequence, self.latest
                if self.error is not None:
                    raise self.error
                if self.finished:
                    return None
                self.condition.wait(timeout=0.2)
        return None

    def close(self):
        self.closed.set()
        with self.condition:
            self.condition.notify_all()
        # RTSP reads have a five-second timeout. The worker owns release() so
        # OpenCV is never released concurrently with an active native read.
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            print("Capture is still finishing its current read; exiting the daemon worker.", file=sys.stderr, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", nargs="+", help="Image files; emit one JSON record per image then exit.")
    source.add_argument("--source", help="Camera index (e.g. 0), /dev/video0, video file or RTSP URL.")
    parser.add_argument("--models-dir", default="/opt/models")
    parser.add_argument("--backend", choices=("rknn", "onnx"), default="rknn")
    parser.add_argument("--detector", help="Override the prepared detector model path.")
    parser.add_argument("--recognizer", help="Override the prepared recognizer model path.")
    parser.add_argument("--dictionary", help="Matching Paddle OCR dictionary path.")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--ocr-width", type=int, default=320)
    parser.add_argument("--ocr-height", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--ocr-threshold", type=float, default=0.5)
    parser.add_argument("--max-plates", type=int, default=20)
    parser.add_argument("--no-split-double", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 runs until EOF/signal.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Infer every Nth captured frame.")
    parser.add_argument("--loop", action="store_true", help="Replay video files after EOF.")
    parser.add_argument("--output-dir", help="Save annotated images and/or latest.jpg.")
    parser.add_argument("--serve", action="store_true", help="Serve the dashboard at /, MJPEG /stream.mjpg and JSON /results.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    for name in ("threshold", "iou", "ocr_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if min(args.detector_size, args.ocr_width, args.ocr_height, args.max_plates, args.frame_stride) <= 0:
        parser.error("Input dimensions, max-plates and frame-stride must be positive.")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative.")
    if args.loop and args.source is None:
        parser.error("--loop requires --source with a video file.")
    args.auto_source = args.image is None and args.source is None
    if args.auto_source:
        args.source = default_webcam_source()
        args.serve = True
        if args.source is None:
            args.image = [str(Path(args.models_dir) / "sample.jpg")]
    args.mode = source_mode(args.source) if args.source is not None else ("sample" if args.auto_source else "image")
    return args


def main(argv=None):
    args = parse_args(argv)
    stop = threading.Event()
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda *_: stop.set())
    pipeline = cap = live_capture = server = None
    source_label = args.source if args.source is not None else args.image[0]
    preview = Preview(stop, mode=args.mode, source=source_label, backend=args.backend)
    try:
        if args.serve:
            server = ThreadingHTTPServer((args.host, args.port), preview.handler())
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"Dashboard: http://{args.host}:{args.port}/ ; MJPEG: /stream.mjpg ; JSON: /results", file=sys.stderr, flush=True)
        pipeline = Pipeline(args)
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        def process(frame, source, frame_index, destination):
            result = pipeline.run(frame)
            result.update(source=redact_source(source), frame=frame_index)
            if args.serve or args.output_dir:
                annotated = annotate(frame, result)
                if args.serve:
                    preview.publish(annotated, result)
                if args.output_dir and not cv2.imwrite(str(Path(args.output_dir) / destination), annotated):
                    raise RuntimeError("Failed to save annotated output image.")
            print(json.dumps(result, ensure_ascii=False), flush=True)

        if args.image:
            for index, path in enumerate(args.image):
                if stop.is_set():
                    break
                frame = cv2.imread(path)
                if frame is None:
                    raise RuntimeError(f"Cannot read image: {path}")
                process(frame, path, index, f"{index:04d}_{Path(path).stem}.jpg")
            # Keep the requested preview available until shutdown.
            if args.serve:
                while not stop.wait(0.2):
                    pass
        else:
            source = int(args.source) if args.source.isdecimal() else args.source
            video_file = isinstance(source, str) and Path(source).is_file()
            if args.loop and not video_file:
                raise RuntimeError("--loop is supported only for video files.")
            network_stream = isinstance(source, str) and bool(urlsplit(source).scheme) and not video_file
            if network_stream:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            else:
                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source: {redact_source(args.source)}")
            if isinstance(source, int) or str(source).startswith("/dev/video"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            index, inferred, cycle_frames = 0, 0, 0
            if args.mode == "camera" or network_stream:
                live_capture = LatestCapture(cap, stop, args.source)
                cap = None  # The capture worker owns release().
                previous = -1
                while not stop.is_set():
                    captured = live_capture.read(previous)
                    if captured is None:
                        break
                    index, frame = captured
                    previous = index
                    if index % args.frame_stride:
                        continue
                    process(frame, args.source, index, "latest.jpg")
                    inferred += 1
                    if args.max_frames and inferred >= args.max_frames:
                        break
                return 0
            while not stop.is_set():
                okay, frame = cap.read()
                if not okay:
                    if args.loop and cycle_frames:
                        if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                            raise RuntimeError("Could not rewind video for --loop.")
                        cycle_frames = 0
                        continue
                    if not video_file or not cycle_frames:
                        raise RuntimeError(f"Source stopped returning frames: {redact_source(args.source)}")
                    break
                cycle_frames += 1
                if index % args.frame_stride == 0:
                    process(frame, args.source, index, "latest.jpg")
                    inferred += 1
                index += 1
                if args.max_frames and inferred >= args.max_frames:
                    break
        return 0
    except (Exception, KeyboardInterrupt) as error:
        preview.update_state("error", error=error)
        print(f"ANPR error: {error}", file=sys.stderr, flush=True)
        if server is not None:
            # Let the dashboard's poll receive the error, then retain nonzero
            # exit semantics so Admiral can report or restart the failed job.
            stop.wait(3)
        return 1
    finally:
        if preview.result["status"] != "error":
            preview.update_state("stopped")
        stop.set()
        if live_capture is not None:
            live_capture.close()
        if cap is not None:
            cap.release()
        if server is not None:
            server.shutdown()
            server.server_close()
        if pipeline is not None:
            pipeline.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
