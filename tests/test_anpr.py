"""Model-free contract tests; real ONNX inference is a separate smoke test."""
import contextlib
import errno
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
    def test_resize_before_channel_conversion_matches_original_pixels(self):
        frame = np.random.default_rng(5).integers(0, 256, (901, 1601, 3), dtype=np.uint8)
        actual, (_, _, left, top) = anpr.letterbox(frame)
        width, height = round(1601 * 640 / 1601), round(901 * 640 / 1601)
        expected = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width, height))
        np.testing.assert_array_equal(actual[top:top + height, left:left + width], expected)

    def test_camera_frame_bound_preserves_aspect_and_does_not_upscale(self):
        landscape = np.zeros((1080, 1920, 3), dtype=np.uint8)
        portrait = np.zeros((1920, 1080, 3), dtype=np.uint8)
        small = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(anpr.bounded_camera_frame(landscape).shape, (720, 1280, 3))
        self.assertEqual(anpr.bounded_camera_frame(portrait).shape, (720, 405, 3))
        self.assertIs(anpr.bounded_camera_frame(small), small)

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

    def test_detection_boxes_refer_to_the_bounded_camera_pixels(self):
        camera = np.zeros((1080, 1920, 3), dtype=np.uint8)
        processed = anpr.bounded_camera_frame(camera)
        _, transform = anpr.letterbox(processed)
        detections = anpr.decode_detections([np.array([[prediction()]], np.float32)], transform, processed.shape)
        np.testing.assert_allclose(detections[0].box, [512, 296, 768, 424])
        self.assertEqual(anpr.crop_plate(processed, detections[0]).shape, (128, 256, 3))

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
        self.assertTrue(all(value >= 0 for value in result["timings_ms"].values()))
        self.assertEqual(result["elapsed_ms"], result["timings_ms"]["pipeline_total"])
        self.assertLessEqual(sum(value for key, value in result["timings_ms"].items()
                                 if key != "pipeline_total"), result["elapsed_ms"] + 0.005)

    def test_pipeline_timings_measure_detector_and_each_recognition_stage(self):
        pipeline = anpr.Pipeline.__new__(anpr.Pipeline)
        pipeline.args = anpr.parse_args(["--image", "test.jpg"])
        pipeline.characters = ["", "A", "B", " "]
        pipeline.detector, pipeline.recognizer = Mock(), Mock()
        pipeline.detector.infer.return_value = [np.array([[prediction()]], dtype=np.float32)]
        pipeline.recognizer.infer.return_value = [ctc_probabilities([1, 2])]
        with patch.object(anpr.time, "perf_counter", side_effect=[index / 100 for index in range(10)]):
            result = pipeline.run(np.zeros((100, 200, 3), dtype=np.uint8))
        for stage in ("detector_preprocess", "detector_inference", "detector_postprocess", "crop",
                      "recognition_preprocess", "recognition_inference", "recognition_postprocess"):
            self.assertEqual(result["timings_ms"][stage], 10)
        self.assertEqual(result["timings_ms"]["pipeline_total"], 90)


class LifecycleTests(unittest.TestCase):
    def test_capacity_dispatch_skips_the_app_models_and_cpu_setup(self):
        capacity = types.ModuleType("scripts.benchmark_capacity")
        capacity.main = Mock(return_value=7)
        with patch.dict("sys.modules", {"scripts.benchmark_capacity": capacity}), \
                patch.object(anpr, "Pipeline") as pipeline, \
                patch.object(anpr, "configure_cpu") as configure_cpu:
            self.assertEqual(anpr.main(["--capacity", "--streams", "1", "2", "--seconds", "10"]), 7)
        capacity.main.assert_called_once_with(["--streams", "1", "2", "--seconds", "10"])
        pipeline.assert_not_called()
        configure_cpu.assert_not_called()

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

    def test_npu_core_choices_are_forwarded_to_the_vendor_runtime(self):
        runtime = Mock()
        runtime.load_rknn.return_value = runtime.init_runtime.return_value = 0
        rknn_class = Mock(return_value=runtime)
        masks = {"auto": ("NPU_CORE_AUTO", 0), "all": ("NPU_CORE_0_1_2", 7),
                 "0": ("NPU_CORE_0", 1), "1": ("NPU_CORE_1", 2), "2": ("NPU_CORE_2", 4)}
        for attribute, value in masks.values():
            setattr(rknn_class, attribute, value)
        api = types.ModuleType("rknnlite.api")
        api.RKNNLite = rknn_class
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.rknn"
            path.touch()
            with patch.dict("sys.modules", {"rknnlite": types.ModuleType("rknnlite"), "rknnlite.api": api}):
                for choice, (_, expected) in masks.items():
                    with self.subTest(cores=choice):
                        model = anpr.Model(path, "rknn", npu_cores=choice)
                        runtime.init_runtime.assert_called_with(core_mask=expected)
                        model.close()
        self.assertEqual(runtime.release.call_count, len(masks))

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


class CPUConfigurationTests(unittest.TestCase):
    def args(self, *flags, backend="rknn"):
        return anpr.parse_args(["--backend", backend, "--image", "test.jpg", *flags])

    def capacity_reader(self, capacities):
        def read(path, *args, **kwargs):
            core = int(path.parent.name.removeprefix("cpu"))
            if core not in capacities:
                raise FileNotFoundError("No kernel CPU capacity")
            return str(capacities[core])
        return read

    def test_auto_selects_only_highest_capacity_allowed_cores_and_own_threads(self):
        initial, expected = {0, 4, 5}, {4, 5}
        with patch.object(anpr.sys, "platform", "linux"), \
                patch.object(anpr.os, "sched_getaffinity", side_effect=[initial, expected], create=True), \
                patch.object(anpr.os, "sched_setaffinity", create=True) as setter, \
                patch.object(anpr.Path, "read_text", autospec=True,
                             side_effect=self.capacity_reader({0: 397, 4: 1024, 5: 1024, 7: 2048})), \
                patch.object(anpr.Path, "iterdir", return_value=[Path("/proc/self/task/100"), Path("/proc/self/task/101")]), \
                patch.object(anpr.cv2, "getNumThreads", return_value=2), \
                patch.object(anpr.cv2, "setNumThreads") as cv_threads:
            result = anpr.configure_cpu(self.args())
        self.assertEqual(result["cores"], [4, 5])
        self.assertEqual(result["selection"], "highest_capacity")
        self.assertEqual(result["policy"], "auto")
        self.assertTrue(result["affinity_applied"])
        self.assertEqual({call.args[0] for call in setter.call_args_list}, {0, 100, 101})
        self.assertTrue(all(call.args[1] == expected for call in setter.call_args_list))
        cv_threads.assert_called_once_with(2)

    def test_equal_or_incomplete_capacity_data_keeps_inherited_affinity(self):
        for capacities in ({0: 397, 1: 397}, {0: 397}, {0: 397, 1: 0}):
            with self.subTest(capacities=capacities), \
                    patch.object(anpr.sys, "platform", "linux"), \
                    patch.object(anpr.os, "sched_getaffinity", return_value={0, 1}, create=True), \
                    patch.object(anpr.os, "sched_setaffinity", create=True) as setter, \
                    patch.object(anpr.Path, "read_text", autospec=True, side_effect=self.capacity_reader(capacities)), \
                    patch.object(anpr.cv2, "getNumThreads", return_value=2), \
                    patch.object(anpr.cv2, "setNumThreads"):
                result = anpr.configure_cpu(self.args())
            self.assertEqual(result["cores"], [0, 1])
            self.assertEqual(result["selection"], "keep")
            setter.assert_not_called()

    def test_onnx_auto_never_changes_affinity(self):
        with patch.object(anpr.sys, "platform", "linux"), \
                patch.object(anpr.os, "sched_getaffinity", return_value={0, 1, 4, 5}, create=True), \
                patch.object(anpr.os, "sched_setaffinity", create=True) as setter, \
                patch.object(anpr.Path, "read_text", autospec=True) as read_capacity, \
                patch.object(anpr.cv2, "getNumThreads", return_value=2), \
                patch.object(anpr.cv2, "setNumThreads"):
            result = anpr.configure_cpu(self.args(backend="onnx"))
        self.assertEqual(result["cores"], [0, 1, 4, 5])
        setter.assert_not_called()
        read_capacity.assert_not_called()

    def test_all_keeps_initial_allowed_set_and_zero_preserves_opencv_thread_setting(self):
        with patch.object(anpr.sys, "platform", "linux"), \
                patch.object(anpr.os, "sched_getaffinity", return_value={4, 5}, create=True), \
                patch.object(anpr.os, "sched_setaffinity", create=True) as setter, \
                patch.object(anpr.Path, "read_text", autospec=True) as read_capacity, \
                patch.object(anpr.Path, "iterdir", return_value=[]), \
                patch.object(anpr.cv2, "getNumThreads", return_value=8), \
                patch.object(anpr.cv2, "setNumThreads") as cv_threads:
            result = anpr.configure_cpu(self.args("--cpu-cores", "all", "--opencv-threads", "0"))
        setter.assert_called_once_with(0, {4, 5})
        read_capacity.assert_not_called()
        cv_threads.assert_not_called()
        self.assertEqual(result["cores"], [4, 5])
        self.assertEqual(result["opencv_threads"], 8)

    def test_explicit_affinity_cannot_broaden_the_inherited_allowed_set(self):
        with patch.object(anpr.sys, "platform", "linux"), \
                patch.object(anpr.os, "sched_getaffinity", return_value={4, 5}, create=True), \
                patch.object(anpr.os, "sched_setaffinity", create=True) as setter, \
                patch.object(anpr.cv2, "getNumThreads", return_value=2), \
                patch.object(anpr.cv2, "setNumThreads") as cv_threads:
            with self.assertRaisesRegex(ValueError, "inherited allowed set"):
                anpr.configure_cpu(self.args("--cpu-cores", "0,4"))
        setter.assert_not_called()
        cv_threads.assert_not_called()

    def test_permission_denial_reports_actual_affinity_and_continues(self):
        with patch.object(anpr.sys, "platform", "linux"), \
                patch.object(anpr.os, "sched_getaffinity", return_value={0, 4}, create=True), \
                patch.object(anpr.os, "sched_setaffinity", side_effect=PermissionError(errno.EPERM, "Not permitted"), create=True), \
                patch.object(anpr.Path, "read_text", autospec=True, side_effect=self.capacity_reader({0: 397, 4: 1024})), \
                patch.object(anpr.Path, "iterdir", return_value=[]), \
                patch.object(anpr.cv2, "getNumThreads", return_value=2), \
                patch.object(anpr.cv2, "setNumThreads"), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            result = anpr.configure_cpu(self.args())
        self.assertEqual(result["cores"], [0, 4])
        self.assertFalse(result["affinity_applied"])
        self.assertIn("Not permitted", result["warning"])
        self.assertIn("CPU configuration", stderr.getvalue())

    def test_unavailable_platform_keeps_cpu_placement(self):
        with patch.object(anpr.sys, "platform", "darwin"), \
                patch.object(anpr.os, "sched_setaffinity", create=True) as setter, \
                patch.object(anpr.cv2, "getNumThreads", return_value=2), \
                patch.object(anpr.cv2, "setNumThreads"):
            result = anpr.configure_cpu(self.args())
        setter.assert_not_called()
        self.assertIsNone(result["cores"])
        self.assertEqual(result["selection"], "keep")

    def test_cpu_cli_validates_explicit_ids_and_thread_count(self):
        args = self.args("--cpu-cores", "4, 5", "--opencv-threads", "4")
        self.assertEqual(args.cpu_cores, "4,5")
        self.assertEqual(args.opencv_threads, 4)
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in (["--cpu-cores", "4,4"], ["--cpu-cores", "4,-1"],
                          ["--cpu-cores", "4,"], ["--cpu-cores", "unknown"], ["--opencv-threads", "-1"]):
                with self.subTest(flags=flags), self.assertRaises(SystemExit):
                    self.args(*flags)


if __name__ == "__main__":
    unittest.main()
