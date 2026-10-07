"""HTTP and automatic-input contracts without an NPU or physical webcam."""

import contextlib
from datetime import datetime
import io
import json
import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import socket
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib.parse import urlsplit

import cv2
import numpy as np

import anpr


@contextlib.contextmanager
def running_server(preview):
    server = ThreadingHTTPServer(("127.0.0.1", 0), preview.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        preview.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class DashboardHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.web_dir = Path(self.directory.name)
        self.assets = {
            "index.html": b"<!doctype html><title>ANPR test</title>",
            "app.js": b"console.log('ANPR test');",
            "style.css": b"body { color: white; }",
            "admiral-logo.svg": b'<svg xmlns="http://www.w3.org/2000/svg"><title>Admiral</title></svg>',
        }
        for name, body in self.assets.items():
            (self.web_dir / name).write_bytes(body)
        # A tempting non-public file must remain inaccessible through HTTP.
        (self.web_dir / "private.txt").write_text("private configuration")
        self.preview = anpr.Preview(threading.Event(), mode="camera", source="0",
                                    backend="onnx", web_dir=self.web_dir)
        self.frame = np.full((12, 24, 3), [10, 30, 70], dtype=np.uint8)

    def read_json(self, base):
        with urlopen(base + "/results", timeout=2) as response:
            self.assertEqual(response.headers.get_content_type(), "application/json")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            return json.load(response)

    def test_html_and_assets_have_correct_content_and_mime(self):
        with running_server(self.preview) as base:
            for route, name, mime in (
                ("/", "index.html", {"text/html"}),
                ("/app.js", "app.js", {"application/javascript", "text/javascript"}),
                ("/style.css", "style.css", {"text/css"}),
                ("/admiral-logo.svg", "admiral-logo.svg", {"image/svg+xml"}),
            ):
                with self.subTest(route=route), urlopen(base + route, timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(response.headers.get_content_type(), mime)
                    self.assertEqual(response.read(), self.assets[name])

    def test_only_allowlisted_assets_are_served(self):
        with running_server(self.preview) as base:
            for route in ("/private.txt", "/anpr.py", "/../private.txt",
                          "/%2e%2e/private.txt", "/app.js/../private.txt", "/BRAND-ASSETS.md", "/missing"):
                with self.subTest(route=route), self.assertRaises(HTTPError) as raised:
                    urlopen(base + route, timeout=2)
                self.assertEqual(raised.exception.code, 404)

    def test_loading_state_is_available_before_first_frame(self):
        with running_server(self.preview) as base:
            result = self.read_json(base)
        self.assertEqual(result["status"], "loading")
        self.assertEqual(result["mode"], "camera")
        self.assertEqual(result["source"], "0")
        self.assertEqual(result["backend"], "onnx")
        self.assertIsNone(result["frame"])
        self.assertIsNone(result["elapsed_ms"])
        self.assertEqual(result["plates"], [])
        self.assertGreaterEqual(result["fps"], 0)
        datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))

    def test_snapshot_returns_503_until_a_frame_exists_then_valid_jpeg(self):
        with running_server(self.preview) as base:
            with self.assertRaises(HTTPError) as raised:
                urlopen(base + "/snapshot.jpg", timeout=2)
            self.assertEqual(raised.exception.code, 503)
            self.preview.publish(self.frame, {"plates": [], "frame": 0, "elapsed_ms": 5})
            with urlopen(base + "/snapshot.jpg", timeout=2) as response:
                self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                body = response.read()
            self.assertEqual(body[:2], b"\xff\xd8")
            decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, self.frame.shape)

    def test_stream_returns_a_complete_jpeg_multipart_frame(self):
        self.preview.publish(self.frame, {"plates": [], "frame": 0, "elapsed_ms": 5})
        with running_server(self.preview) as base:
            with urlopen(base + "/stream.mjpg", timeout=2) as response:
                self.assertEqual(response.headers.get_content_type(), "multipart/x-mixed-replace")
                boundary = response.headers.get_param("boundary")
                self.assertTrue(boundary)
                self.assertEqual(response.readline().strip(), b"--" + boundary.encode())
                headers = {}
                while True:
                    line = response.readline()
                    if line == b"\r\n":
                        break
                    key, value = line.decode().strip().split(":", 1)
                    headers[key.lower()] = value.strip()
                self.assertEqual(headers["content-type"], "image/jpeg")
                length = int(headers["content-length"])
                self.assertGreater(length, 0)
                jpeg = response.read(length)
                self.assertEqual(len(jpeg), length)
                self.assertEqual(cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), 1).shape,
                                 self.frame.shape)
                self.assertEqual(response.read(2), b"\r\n")
                self.assertEqual(response.readline(), b"--" + boundary.encode() + b"\r\n")
                # The following boundary completes the first JPEG even when
                # no second frame exists. A new frame must start with headers,
                # not another duplicate boundary line.
                second_frame = np.full(self.frame.shape, 150, dtype=np.uint8)
                self.preview.publish(second_frame, {"plates": [], "frame": 1, "elapsed_ms": 5})
                self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
                length_header = response.readline().decode().strip()
                self.assertTrue(length_header.startswith("Content-Length: "))
                second_length = int(length_header.split(":", 1)[1])
                self.assertEqual(response.readline(), b"\r\n")
                second_jpeg = response.read(second_length)
                np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(second_jpeg, dtype=np.uint8), 1), second_frame)
                self.assertEqual(response.read(2), b"\r\n")
                self.assertEqual(response.readline(), b"--" + boundary.encode() + b"\r\n")

    def test_stream_waits_for_a_new_frame_instead_of_resending_at_ten_fps(self):
        self.preview.publish(self.frame, {"plates": [], "frame": 0, "elapsed_ms": 5})
        with running_server(self.preview) as base:
            address = urlsplit(base)
            connection = socket.create_connection((address.hostname, address.port), timeout=2)
            response = None
            try:
                connection.sendall(b"GET /stream.mjpg HTTP/1.0\r\nHost: localhost\r\n\r\n")
                response = http.client.HTTPResponse(connection)
                response.begin()
                self.assertTrue(response.readline().startswith(b"--jpgboundary"))
                headers = {}
                while True:
                    line = response.readline()
                    if line == b"\r\n":
                        break
                    key, value = line.decode().strip().split(":", 1)
                    headers[key.lower()] = value.strip()
                response.read(int(headers["content-length"]))
                self.assertEqual(response.read(2), b"\r\n")
                self.assertEqual(response.readline(), b"--jpgboundary\r\n")
                # A static sample should leave the connection idle; the old
                # handler sent duplicate JPEGs every 100 ms regardless of input.
                connection.settimeout(.25)
                with self.assertRaises(TimeoutError):
                    response.readline()
            finally:
                if response is not None:
                    response.close()
                connection.close()

    def test_results_publish_unicode_and_runtime_error_transitions(self):
        with running_server(self.preview) as base:
            initial = self.read_json(base)
            self.preview.publish(self.frame, {
                "plates": [{"plate": "京A12345", "confidence": 0.91}],
                "frame": 3, "elapsed_ms": 12.5,
            })
            live = self.read_json(base)
            self.assertEqual(live["status"], "live")
            self.assertEqual(live["plates"][0]["plate"], "京A12345")
            self.assertEqual(live["frame"], 3)
            self.assertEqual(live["elapsed_ms"], 12.5)
            self.assertEqual(live["fps"], 0)
            self.assertGreater(live["sequence"], initial["sequence"])
            self.preview.update_state("error", error="Camera stopped returning frames")
            failed = self.read_json(base)
            self.assertEqual(failed["status"], "error")
            self.assertEqual(failed["error"], "Camera stopped returning frames")
            self.assertGreater(failed["sequence"], live["sequence"])
            self.preview.update_state("stopped")
            stopped = self.read_json(base)
            self.assertEqual(stopped["status"], "stopped")
            self.assertGreater(stopped["sequence"], failed["sequence"])

    def test_sample_status_does_not_claim_a_live_camera(self):
        preview = anpr.Preview(threading.Event(), mode="sample", source="sample.jpg",
                               backend="onnx", web_dir=self.web_dir)
        preview.publish(self.frame, {"plates": [], "frame": 0, "elapsed_ms": 1})
        with running_server(preview) as base:
            result = self.read_json(base)
        self.assertEqual((result["status"], result["mode"]), ("sample", "sample"))
        self.assertEqual(result["fps"], 0)

    def test_live_fps_measures_processed_frame_intervals(self):
        # Camera capture, cropping and OCR all contribute to the actual
        # processed-frame interval; inference duration alone is not FPS.
        with patch.object(anpr.time, "perf_counter", side_effect=[10.0, 10.5, 11.5]):
            self.preview.publish(self.frame, {"plates": [], "frame": 0, "elapsed_ms": 1})
            self.assertEqual(self.preview.result["fps"], 0)
            self.preview.publish(self.frame, {"plates": [], "frame": 1, "elapsed_ms": 1})
            self.assertEqual(self.preview.result["fps"], 2)
            self.preview.publish(self.frame, {"plates": [], "frame": 2, "elapsed_ms": 1})
            self.assertAlmostEqual(self.preview.result["fps"], 1.8)

    def test_stream_credentials_are_not_exposed_in_results(self):
        preview = anpr.Preview(threading.Event(), mode="video",
                               source="rtsp://camera-user:camera-secret@camera.local/live",
                               backend="onnx", web_dir=self.web_dir)
        with running_server(preview) as base:
            result = self.read_json(base)
        self.assertNotIn("camera-user", result["source"])
        self.assertNotIn("camera-secret", result["source"])
        self.assertIn("camera.local", result["source"])


class AutomaticInputTests(unittest.TestCase):
    def test_webcam_discovery_checks_the_mapped_linux_video_device(self):
        with patch.object(anpr.Path, "exists", autospec=True, return_value=True) as exists:
            self.assertEqual(anpr.default_webcam_source(), "0")
        self.assertEqual(exists.call_args.args[0], Path("/dev/video0"))
        with patch.object(anpr.Path, "exists", autospec=True, return_value=False):
            self.assertIsNone(anpr.default_webcam_source())

    def test_no_input_prefers_the_mapped_webcam_and_serves_the_dashboard(self):
        with patch.object(anpr, "default_webcam_source", return_value="0"):
            args = anpr.parse_args([])
        self.assertEqual(args.source, "0")
        self.assertIsNone(args.image)
        self.assertEqual(args.mode, "camera")
        self.assertTrue(args.auto_source)
        self.assertTrue(args.serve)

    def test_no_webcam_uses_a_sample_and_serves_the_dashboard(self):
        with patch.object(anpr, "default_webcam_source", return_value=None):
            args = anpr.parse_args(["--models-dir", "/test-models"])
        self.assertEqual(args.image, ["/test-models/sample.jpg"])
        self.assertIsNone(args.source)
        self.assertEqual(args.mode, "sample")
        self.assertTrue(args.serve)

    def test_explicit_images_keep_one_shot_json_behavior(self):
        pipeline = Mock()
        pipeline.run.return_value = {"plates": [], "elapsed_ms": 1}
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        stdout = io.StringIO()
        with patch.object(anpr, "default_webcam_source") as webcam, \
                patch.object(anpr, "Pipeline", return_value=pipeline), \
                patch.object(anpr.cv2, "imread", return_value=frame), \
                patch.object(anpr, "ThreadingHTTPServer") as server, \
                contextlib.redirect_stdout(stdout):
            self.assertEqual(anpr.main(["--image", "first.jpg", "second.jpg"]), 0)
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([record["source"] for record in records], ["first.jpg", "second.jpg"])
        self.assertEqual([record["frame"] for record in records], [0, 1])
        self.assertEqual(pipeline.run.call_count, 2)
        pipeline.close.assert_called_once()
        webcam.assert_not_called()
        server.assert_not_called()

    def test_automatic_webcam_run_releases_the_worker_and_pipeline(self):
        pipeline, capture, worker = Mock(), Mock(), Mock()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        pipeline.run.return_value = {"plates": [], "elapsed_ms": 1}
        capture.isOpened.return_value = True
        worker.read.return_value = (7, frame, {"capture_read": 30.0, "captured_at": anpr.time.perf_counter(), "capture_fps": 30.0})
        output = io.StringIO()
        with patch.object(anpr, "default_webcam_source", return_value="0"), \
                patch.object(anpr, "Pipeline", return_value=pipeline), \
                patch.object(anpr.cv2, "VideoCapture", return_value=capture) as open_capture, \
                patch.object(anpr, "LatestCapture", return_value=worker) as latest_capture, \
                patch.object(anpr, "ThreadingHTTPServer") as server, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(anpr.main(["--max-frames", "1"]), 0)
        open_capture.assert_called_once_with(0)
        self.assertIs(latest_capture.call_args.args[0], capture)
        self.assertEqual(latest_capture.call_args.args[2], "0")
        self.assertEqual(json.loads(output.getvalue())["frame"], 7)
        worker.close.assert_called_once()
        capture.release.assert_not_called()  # The worker owns the native capture.
        pipeline.close.assert_called_once()
        server.return_value.shutdown.assert_called_once()
        server.return_value.server_close.assert_called_once()

    def test_ignored_webcam_settings_still_bound_every_pipeline_and_preview_pixel(self):
        pipeline, capture, worker = Mock(), Mock(), Mock()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        pipeline.run.return_value = {"plates": [], "elapsed_ms": 1, "timings_ms": {"pipeline_total": 1}}
        capture.isOpened.return_value = True
        capture.set.return_value = True  # Backend reports success; the device ignores it.
        reported = {cv2.CAP_PROP_FRAME_WIDTH: 1920, cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                    cv2.CAP_PROP_FPS: 15, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"YUYV")}
        capture.get.side_effect = reported.__getitem__
        worker.read.return_value = (7, frame, {"capture_read": 66, "captured_at": anpr.time.perf_counter(), "capture_fps": 15})
        stdout = io.StringIO()
        with patch.object(anpr, "Pipeline", return_value=pipeline), \
                patch.object(anpr.cv2, "VideoCapture", return_value=capture), \
                patch.object(anpr, "LatestCapture", return_value=worker), \
                patch.object(anpr, "ThreadingHTTPServer"), \
                patch.object(anpr.Preview, "publish") as publish, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(anpr.main(["--source", "0", "--serve", "--max-frames", "1"]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(pipeline.run.call_args.args[0].shape, (720, 1280, 3))
        self.assertEqual(publish.call_args.args[0].shape, (720, 1280, 3))
        decoded = cv2.imdecode(np.frombuffer(publish.call_args.kwargs["jpeg_bytes"], np.uint8), 1)
        self.assertEqual(decoded.shape, (720, 1280, 3))
        self.assertEqual(result["frame_size"], {"capture": [1920, 1080], "processed": [1280, 720],
                                              "software_scaled": True, "coordinates": "processed"})
        self.assertEqual(result["camera"]["requested"], {"width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG"})
        self.assertEqual(result["camera"]["negotiated"], {"width": 1920, "height": 1080, "fps": 15, "fourcc": "YUYV"})
        self.assertEqual(result["camera"]["observed"], {"width": 1920, "height": 1080, "fps": 15})
        self.assertTrue(all(result["camera"]["settings_supported"].values()))
        capture.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 4)
        self.assertEqual(result["timings_ms"]["capture_read"], 66)
        for stage in ("resize", "annotate", "jpeg", "processing_total", "frame_age"):
            self.assertGreaterEqual(result["timings_ms"][stage], 0)
        self.assertGreater(result["timings_ms"]["processing_total"], result["timings_ms"]["jpeg"])

    def test_rejected_camera_settings_are_reported_and_custom_requests_are_validated(self):
        capture = Mock()
        capture.set.return_value = False
        capture.get.return_value = 0
        args = anpr.parse_args(["--source", "0", "--camera-width", "640", "--camera-height", "480",
                                "--camera-fps", "15", "--camera-fourcc", "auto"])
        metadata = anpr.configure_camera(capture, args)
        self.assertEqual(metadata["requested"], {"width": 640, "height": 480, "fps": 15, "fourcc": None})
        self.assertEqual(metadata["negotiated"], {"width": None, "height": None, "fps": None, "fourcc": None})
        self.assertFalse(metadata["settings_supported"]["width"])
        self.assertIsNone(metadata["settings_supported"]["fourcc"])
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (["--camera-width", "0"], ["--camera-height", "-1"],
                          ["--camera-fps", "nan"], ["--camera-fps", "0"], ["--camera-fourcc", "invalid"]):
                with self.subTest(flags=flags), self.assertRaises(SystemExit):
                    anpr.parse_args(["--source", "0", *flags])

    def test_image_resolution_is_preserved_and_has_no_camera_negotiation(self):
        pipeline = Mock()
        pipeline.run.return_value = {"plates": [], "elapsed_ms": 1}
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        stdout = io.StringIO()
        with patch.object(anpr, "Pipeline", return_value=pipeline), \
                patch.object(anpr.cv2, "imread", return_value=frame), \
                contextlib.redirect_stdout(stdout):
            self.assertEqual(anpr.main(["--image", "high-resolution.jpg"]), 0)
        self.assertIs(pipeline.run.call_args.args[0], frame)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["frame_size"]["processed"], [1920, 1080])
        self.assertFalse(result["frame_size"]["software_scaled"])
        self.assertNotIn("camera", result)
        self.assertEqual(result["fps"], 0)

    def test_headless_video_intervals_measure_completion_throughput(self):
        pipeline, capture = Mock(), Mock()
        pipeline.run.side_effect = [{"plates": [], "elapsed_ms": 1}, {"plates": [], "elapsed_ms": 1}]
        capture.isOpened.return_value = True
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        capture.read.return_value = (True, frame)
        stdout = io.StringIO()
        clocks = [0, .010, .020, .030, .050, .060, .070, .080, .090, .150]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.touch()
            with patch.object(anpr, "Pipeline", return_value=pipeline), \
                    patch.object(anpr.cv2, "VideoCapture", return_value=capture), \
                    patch.object(anpr.time, "perf_counter", side_effect=clocks), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(anpr.main(["--source", str(path), "--max-frames", "2"]), 0)
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertIsNone(records[0]["processing_interval_ms"])
        self.assertEqual(records[0]["fps"], 0)
        self.assertEqual(records[1]["processing_interval_ms"], 100)
        self.assertEqual(records[1]["fps"], 10)
        self.assertEqual(records[1]["timings_ms"]["processing_total"], 70)
        self.assertEqual(records[1]["timings_ms"]["frame_age"], 80)
        self.assertTrue(all(call.args[0] is frame for call in pipeline.run.call_args_list))
        capture.set.assert_not_called()

    def test_dashboard_is_served_during_model_loading_and_reports_init_failure(self):
        previews, servers, states = [], [], []
        preview_class, server_class = anpr.Preview, anpr.ThreadingHTTPServer

        def create_server(*args, **kwargs):
            server = server_class(*args, **kwargs)
            servers.append(server)
            return server

        def read_current_state():
            with urlopen(f"http://127.0.0.1:{servers[0].server_port}/results", timeout=2) as response:
                return json.load(response)

        def create_preview(*args, **kwargs):
            preview = preview_class(*args, **kwargs)
            previews.append(preview)
            original_update = preview.update_state

            def update(status, error=None):
                original_update(status, error)
                if status == "error":
                    # Observe the published error over the real HTTP route
                    # before requesting shutdown; no fixed sleeps required.
                    states.append(read_current_state())
                    preview.stop.set()

            preview.update_state = update
            return preview

        def fail_initialization(_args):
            states.append(read_current_state())
            raise RuntimeError("NPU initialization failed")

        with patch.object(anpr, "Preview", side_effect=create_preview), \
                patch.object(anpr, "ThreadingHTTPServer", side_effect=create_server), \
                patch.object(anpr, "Pipeline", side_effect=fail_initialization), \
                contextlib.redirect_stderr(io.StringIO()):
            code = anpr.main(["--image", "sample.jpg", "--serve", "--host", "127.0.0.1", "--port", "0"])
        self.assertEqual(code, 1)
        self.assertEqual([state["status"] for state in states], ["loading", "error"])
        self.assertEqual(states[1]["error"], "NPU initialization failed")
        self.assertTrue(previews[0].stop.is_set())
        self.assertEqual(servers[0].fileno(), -1)


class LiveCaptureTests(unittest.TestCase):
    def test_slow_consumer_receives_newest_frame_without_a_backlog(self):
        stop, drained, unblock = threading.Event(), threading.Event(), threading.Event()
        frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in range(4)]

        class BurstCapture:
            def __init__(self):
                self.index, self.release_count = 0, 0

            def read(self):
                if self.index < len(frames):
                    frame = frames[self.index]
                    self.index += 1
                    return True, frame
                drained.set()
                unblock.wait(timeout=2)
                return False, None

            def release(self):
                self.release_count += 1

        capture = BurstCapture()
        worker = anpr.LatestCapture(capture, stop, "0")
        try:
            self.assertTrue(drained.wait(timeout=2), "Capture worker did not drain its frame burst")
            sequence, frame = worker.read(-1)
            self.assertEqual(sequence, 3)
            np.testing.assert_array_equal(frame, frames[-1])
            measured_sequence, measured_frame, metrics = worker.read(-1, with_metrics=True)
            self.assertEqual(measured_sequence, sequence)
            np.testing.assert_array_equal(measured_frame, frame)
            self.assertGreaterEqual(metrics["capture_read"], 0)
            self.assertIsInstance(metrics["captured_at"], float)
            self.assertGreater(metrics["capture_fps"], 0)
            unblock.set()
            with self.assertRaisesRegex(RuntimeError, "stopped returning frames"):
                worker.read(sequence)
        finally:
            stop.set()
            unblock.set()
            worker.close()
        self.assertFalse(worker.thread.is_alive())
        self.assertEqual(capture.release_count, 1)

    def test_capture_exception_reaches_consumer_and_release_occurs_once(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        capture = Mock()
        capture.read.side_effect = [(True, frame), RuntimeError("Camera disconnected")]
        worker = anpr.LatestCapture(capture, threading.Event(), "0")
        try:
            sequence, received = worker.read(-1)
            np.testing.assert_array_equal(received, frame)
            with self.assertRaisesRegex(RuntimeError, "Camera disconnected"):
                worker.read(sequence)
        finally:
            worker.close()
        self.assertFalse(worker.thread.is_alive())
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
