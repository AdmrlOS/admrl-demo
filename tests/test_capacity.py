"""Capacity reporting uses completed work and verified workloads, without an NPU."""

import unittest

import numpy as np

from scripts import benchmark_capacity as capacity


def worker(frames, plates=2, preflight=None):
    return {"completed_frames": frames, "preflight_plate_count": plates if preflight is None else preflight,
            "plate_count_distribution": {str(plates): frames} if frames else {}}


class CapacityReportingTests(unittest.TestCase):
    def summarize(self, workers, backend="rknn", plates=2, duration=20):
        return capacity.summarize_configuration(workers, duration, 30, backend, plates)

    def test_rates_use_completed_frame_count_over_common_window(self):
        report = self.summarize([worker(120), worker(100)])
        self.assertEqual(report["completed_frames"], 220)
        self.assertEqual(report["aggregate_processing_fps"], 11)
        self.assertEqual(report["per_stream_fps"], [6, 5])
        self.assertEqual(report["minimum_per_stream_fps"], 5)
        self.assertAlmostEqual(report["aggregate_target_fps_equivalent"], 11 / 30)

    def test_aggregate_30_fps_does_not_verify_individual_streams(self):
        report = self.summarize([worker(400), worker(400)])
        self.assertEqual(report["aggregate_processing_fps"], 40)
        self.assertFalse(report["all_streams_meet_target"])
        self.assertFalse(report["verified_npu_streams_meet_target"])

    def test_weakest_stream_decides_target_even_when_average_meets_it(self):
        report = self.summarize([worker(800), worker(400)])
        self.assertEqual(report["aggregate_processing_fps"], 60)
        self.assertFalse(report["all_streams_meet_target"])

    def test_boundary_30_fps_passes_for_every_stream(self):
        report = self.summarize([worker(600), worker(600), worker(600)])
        self.assertTrue(report["verified_npu_streams_meet_target"])

    def test_preflight_and_every_measured_plate_count_must_match(self):
        for item in (worker(1000, preflight=1), worker(1000, plates=0)):
            report = self.summarize([item])
            self.assertFalse(report["workload_valid"])
            self.assertFalse(report["all_streams_meet_target"])
        item = worker(1000)
        item["plate_count_distribution"] = {"2": 999, "1": 1}
        self.assertFalse(self.summarize([item])["workload_valid"])

    def test_empty_scene_has_a_separate_verified_workload(self):
        self.assertTrue(self.summarize([worker(600, plates=0)], plates=0)["workload_valid"])

    def test_zero_completions_or_no_workers_cannot_verify_capacity(self):
        for workers in ([], [worker(0)]):
            self.assertFalse(self.summarize(workers)["workload_valid"])

    def test_onnx_can_measure_throughput_but_cannot_establish_npu_capacity(self):
        config = self.summarize([worker(1000)], backend="onnx")
        self.assertTrue(config["all_streams_meet_target"])
        self.assertFalse(config["verified_npu_streams_meet_target"])
        self.assertEqual(capacity.capacity_summary([config], "onnx", 30)["largest_tested_stream_count_meeting_target"], 0)

    def test_ceiling_and_largest_verified_configuration_are_distinct(self):
        passing = self.summarize([worker(620), worker(620)])
        saturated = self.summarize([worker(500)] * 3)
        report = capacity.capacity_summary([passing, saturated], "rknn", 30)
        self.assertEqual(report["maximum_measured_aggregate_fps"], 75)
        self.assertEqual(report["best_tested_concurrency"], 3)
        self.assertEqual(report["largest_tested_stream_count_meeting_target"], 2)

    def test_invalid_workloads_do_not_inflate_measured_ceiling(self):
        valid = self.summarize([worker(120)])
        invalid = self.summarize([worker(1000, plates=0)])
        report = capacity.capacity_summary([valid, invalid], "rknn", 30)
        self.assertEqual(report["maximum_measured_aggregate_fps"], 6)
        self.assertEqual(report["largest_tested_stream_count_meeting_target"], 0)

    def test_invalid_measurement_parameters_are_rejected(self):
        for duration, target in ((0, 30), (-1, 30), (float("inf"), 30), (20, float("nan"))):
            with self.subTest(duration=duration, target=target), self.assertRaises(ValueError):
                capacity.summarize_configuration([worker(1)], duration, target, "rknn", 2)
        with self.assertRaises(ValueError):
            self.summarize([worker(-1)])

    def test_only_completions_inside_the_common_deadline_are_counted(self):
        self.assertFalse(capacity.completed_in_window(9.999, 10, 30))
        self.assertTrue(capacity.completed_in_window(10, 10, 30))
        self.assertTrue(capacity.completed_in_window(30, 10, 30))
        self.assertFalse(capacity.completed_in_window(30.001, 10, 30))


class CapacityInputTests(unittest.TestCase):
    def test_exact_720p_letterbox_preserves_portrait_aspect_and_background(self):
        source = np.full((100, 50, 3), [11, 22, 33], np.uint8)
        frame = capacity.prepare_frame(source)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)
        np.testing.assert_array_equal(frame[:, 460:820], np.broadcast_to([11, 22, 33], (720, 360, 3)))
        self.assertTrue(np.all(frame[:, :460] == 114))
        self.assertTrue(np.all(frame[:, 820:] == 114))

    def test_auto_cpu_selection_uses_kernel_capacity_and_allowed_affinity(self):
        allowed = [2, 3, 4, 5]
        capacities = {2: 397, 3: 397, 4: 1024, 5: 1024}
        self.assertEqual(capacity.select_cpu_cores("auto", "rknn", allowed, capacities), [4, 5])
        self.assertEqual(capacity.select_cpu_cores("all", "rknn", allowed, capacities), allowed)
        self.assertEqual(capacity.select_cpu_cores("auto", "onnx", allowed, capacities), allowed)
        self.assertEqual(capacity.select_cpu_cores("auto", "rknn", allowed, {4: 1024}), allowed)
        with self.assertRaises(ValueError):
            capacity.select_cpu_cores([0], "rknn", allowed, capacities)

    def test_all_three_npu_utilization_values_are_parsed(self):
        self.assertEqual(capacity.parse_npu_load("NPU load: Core0: 99%, Core1: 80%, Core2: 0%"),
                         {0: 99, 1: 80, 2: 0})
        with self.assertRaises(ValueError):
            capacity.parse_npu_load("Core0: 99%")


class CapacityCleanupTests(unittest.TestCase):
    class Process:
        def __init__(self, pid, states):
            self.pid, self.states, self.calls = pid, iter(states), []

        def is_alive(self):
            return next(self.states)

        def join(self, timeout):
            self.calls.append(("join", timeout))

        def terminate(self):
            self.calls.append(("terminate",))

        def kill(self):
            self.calls.append(("kill",))

    def test_stuck_worker_is_terminated_then_killed_and_verified(self):
        process = self.Process(42, [True, True, False])
        self.assertEqual(capacity.cleanup_workers([process]), [42])
        self.assertEqual(process.calls, [("join", 2), ("terminate",), ("join", 2), ("kill",), ("join", 2)])

    def test_every_worker_is_cleaned_up_before_remaining_pids_raise(self):
        stuck = self.Process(42, [True, True, True])
        stopped = self.Process(43, [False, False, False])
        with self.assertRaisesRegex(RuntimeError, "42"):
            capacity.cleanup_workers([stuck, stopped])
        self.assertEqual(stopped.calls, [("join", 2)])

    def test_sigterm_raises_exit_to_unwind_parent_cleanup(self):
        with self.assertRaises(SystemExit) as error:
            capacity.abort_on_signal(15, None)
        self.assertEqual(error.exception.code, 143)


if __name__ == "__main__":
    unittest.main()
