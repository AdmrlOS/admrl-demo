"""Model-free contract tests; real ONNX inference is a separate smoke test."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

import anpr


def prediction(confidence=0.9, class_id=0):
    scores = [0.8, 0.2] if class_id == 0 else [0.2, 0.8]
    return [320, 320, 128, 64, confidence, 256, 288, 384, 288,
            384, 352, 256, 352, *scores]


def ctc_probabilities(indices, classes=4):
    probabilities = np.full((1, len(indices), classes), 0.01, dtype=np.float32)
    for timestep, index in enumerate(indices):
        probabilities[0, timestep, index] = 1 - 0.01 * (classes - 1)
    return probabilities


class DetectionTests(unittest.TestCase):
    def test_letterbox_inverse_geometry_and_nms(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        _, transform = anpr.letterbox(frame)
        rows = np.array([[prediction(), prediction(0.7), prediction(0.1)]], dtype=np.float32)
        detections = anpr.decode_detections([rows], transform, frame.shape)
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(detections[0].box, [80, 40, 120, 60], atol=1e-5)
        np.testing.assert_allclose(detections[0].corners,
                                   [[80, 40], [120, 40], [120, 60], [80, 60]], atol=1e-5)
        self.assertAlmostEqual(detections[0].confidence, 0.72, places=5)

    def test_detector_channel_contract_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            anpr.decode_detections([np.zeros((1, 10, 6))], (1, 1, 0, 0), (100, 100))

    def test_nan_and_outside_frame_detections_are_dropped(self):
        rows = np.array([[prediction(), prediction()]], dtype=np.float32)
        rows[0, 0, 5] = np.nan
        rows[0, 1, 0:2] = [-200, -200]
        self.assertEqual(anpr.decode_detections([rows], (1, 1, 0, 0), (100, 100)), [])

    def test_crop_rectification_and_double_row_reordering(self):
        frame = np.zeros((60, 120, 3), dtype=np.uint8)
        frame[:30] = [10, 20, 30]
        frame[30:] = [100, 110, 120]
        detection = anpr.Detection(np.float32([0, 0, 119, 59]),
                                   np.float32([[0, 0], [119, 0], [119, 59], [0, 59]]), 0.9, 1)
        single = anpr.crop_plate(frame, detection, split_double=False)
        double = anpr.crop_plate(frame, detection)
        self.assertEqual(single.shape[:2], (59, 119))
        self.assertEqual(double.shape[:2], (40, 238))
        np.testing.assert_array_equal(double[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(double[-1, -1], [100, 110, 120])

    def test_bad_corners_fall_back_to_clipped_bbox(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        detection = anpr.Detection(np.float32([10, 20, 90, 50]), np.zeros((4, 2)), 0.9, 0)
        self.assertEqual(anpr.crop_plate(frame, detection).shape, (30, 80, 3))

    def test_corner_reordering_preserves_crop_orientation(self):
        frame = np.full((100, 200, 3), [10, 20, 30], dtype=np.uint8)
        frame[20:30, 10:20] = [100, 110, 120]
        points = np.float32([[10, 20], [90, 20], [90, 50], [10, 50]])
        ordered = anpr.Detection(np.float32([10, 20, 90, 50]), points, 0.9, 0)
        shuffled = anpr.Detection(ordered.box, points[[2, 0, 3, 1]], 0.9, 0)
        np.testing.assert_array_equal(anpr.crop_plate(frame, ordered), anpr.crop_plate(frame, shuffled))


class OCRTests(unittest.TestCase):
    def test_ctc_repeats_blank_and_confidence(self):
        text, confidence = anpr.decode_ctc(ctc_probabilities([0, 1, 1, 0, 1, 2, 2, 0]),
                                           ["", "A", "B", " "])
        self.assertEqual(text, "AAB")
        self.assertAlmostEqual(confidence, 0.97, places=6)

    def test_ctc_blank_has_zero_confidence(self):
        text, confidence = anpr.decode_ctc(ctc_probabilities([0, 0]), ["", "A", "B", " "])
        self.assertEqual((text, confidence), ("", 0))

    def test_dictionary_mismatch_and_logits_fail(self):
        with self.assertRaisesRegex(ValueError, "dictionary"):
            anpr.decode_ctc(ctc_probabilities([1]), ["", "A"])
        with self.assertRaisesRegex(ValueError, "logits"):
            anpr.decode_ctc(np.array([[[0, 7, -1, 0]]]), ["", "A", "B", " "])

    def test_dictionary_keeps_unicode_and_appends_blank_space(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dict.txt"
            path.write_text("京\nA\n", encoding="utf-8")
            self.assertEqual(anpr.load_characters(path), ["", "京", "A", " "])

    def test_end_to_end_pipeline_color_contract_and_scores(self):
        pipeline = anpr.Pipeline.__new__(anpr.Pipeline)
        pipeline.args = anpr.parse_args([])
        pipeline.characters = ["", "A", "B", " "]
        pipeline.detector = Mock()
        pipeline.recognizer = Mock()
        pipeline.detector.infer.return_value = [np.array([[prediction()]], dtype=np.float32)]
        pipeline.recognizer.infer.return_value = [ctc_probabilities([0, 1, 1, 0, 2])]
        frame = np.full((100, 200, 3), [10, 20, 30], dtype=np.uint8)
        result = pipeline.run(frame)
        detector_input = pipeline.detector.infer.call_args.args[0]
        recognizer_input = pipeline.recognizer.infer.call_args.args[0]
        np.testing.assert_array_equal(detector_input[320, 320], [30, 20, 10])
        np.testing.assert_array_equal(recognizer_input[20, 20], [10, 20, 30])
        plate = result["plates"][0]
        self.assertEqual(plate["plate"], "AB")
        self.assertTrue(plate["accepted"])
        self.assertAlmostEqual(plate["confidence"], 0.72 * 0.97, places=5)
        self.assertEqual(recognizer_input.shape, (48, 320, 3))


class LifecycleTests(unittest.TestCase):
    def test_native_logs_do_not_contaminate_stdout_and_fd_is_restored(self):
        code = """import ctypes, os, anpr
try:
    with anpr.vendor_output():
        print('python diagnostic')
        os.write(1, b'unbuffered native diagnostic\\n')
        ctypes.CDLL(None).printf(b'buffered native diagnostic\\n')
        raise RuntimeError('initialization failure')
except RuntimeError:
    pass
print('{"plates": []}')
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), {"plates": []})
        for message in ("python diagnostic", "unbuffered native diagnostic", "buffered native diagnostic"):
            self.assertIn(message, result.stderr)

    def test_partial_pipeline_initialization_releases_detector(self):
        detector = Mock()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "ppocr_keys_v1.txt").write_text("A\n")
            args = anpr.parse_args(["--models-dir", directory])
            with patch.object(anpr, "Model", side_effect=[detector, RuntimeError("OCR init failed")]):
                with self.assertRaisesRegex(RuntimeError, "OCR init failed"):
                    anpr.Pipeline(args)
        detector.close.assert_called_once()

    def test_rknn_initialization_failure_releases_native_runtime(self):
        runtime = Mock()
        runtime.load_rknn.return_value = 0
        runtime.init_runtime.return_value = -1
        rknn_class = Mock(return_value=runtime)
        rknn_class.NPU_CORE_AUTO = 0
        api = types.ModuleType("rknnlite.api")
        api.RKNNLite = rknn_class
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.rknn"
            path.touch()
            with patch.dict("sys.modules", {"rknnlite": types.ModuleType("rknnlite"), "rknnlite.api": api}):
                with self.assertRaisesRegex(RuntimeError, "NPU initialization"):
                    anpr.Model(path, "rknn")
        runtime.release.assert_called_once()

    def test_unreadable_image_exits_nonzero_and_closes_pipeline(self):
        pipeline = Mock()
        with patch.object(anpr, "Pipeline", return_value=pipeline), \
                patch.object(anpr.cv2, "imread", return_value=None), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(anpr.main(["--image", "missing.jpg"]), 1)
        pipeline.close.assert_called_once()

    def test_empty_video_fails_without_endless_replay_and_releases_capture(self):
        pipeline, capture = Mock(), Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.touch()
            with patch.object(anpr, "Pipeline", return_value=pipeline), \
                    patch.object(anpr.cv2, "VideoCapture", return_value=capture), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(anpr.main(["--source", str(path), "--loop"]), 1)
        capture.release.assert_called_once()
        pipeline.close.assert_called_once()

    def test_max_frames_and_stdout_are_json_lines(self):
        pipeline, capture = Mock(), Mock()
        pipeline.run.side_effect = [{"plates": [], "elapsed_ms": 1}, {"plates": [], "elapsed_ms": 1}]
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        output = io.StringIO()
        # File playback preserves every frame; live cameras intentionally use
        # a latest-frame buffer so slow OCR does not accumulate stale frames.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.touch()
            with patch.object(anpr, "Pipeline", return_value=pipeline), \
                    patch.object(anpr.cv2, "VideoCapture", return_value=capture), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(anpr.main(["--source", str(path), "--max-frames", "2", "--frame-stride", "2"]), 0)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([record["frame"] for record in records], [0, 2])
        self.assertEqual(capture.read.call_count, 3)
        capture.release.assert_called_once()
        pipeline.close.assert_called_once()

    def test_preview_json_preserves_unicode(self):
        preview = anpr.Preview(threading.Event())
        preview.publish(np.zeros((10, 10, 3), dtype=np.uint8), {"plates": [{"plate": "京A12345"}]})
        server = ThreadingHTTPServer(("127.0.0.1", 0), preview.handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/results", timeout=2) as response:
                result = json.load(response)
            self.assertEqual(result["plates"][0]["plate"], "京A12345")
        finally:
            preview.stop.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
