import unittest

from scripts.benchmark_anpr import summarise


class BenchmarkTests(unittest.TestCase):
    def record(self, interval, elapsed, plates=0):
        return {"processing_interval_ms": interval, "timings_ms": {"pipeline_total": elapsed},
                "frame_size": {"capture": [1920, 1080], "processed": [1280, 720]},
                "mode": "camera", "backend": "rknn", "plates": [{"plate": "private"}] * plates}

    def test_warmup_is_excluded_and_throughput_includes_work_outside_inference(self):
        records = [self.record(None, 900), self.record(999, 900),
                   self.record(100, 10, 2), self.record(100, 20), self.record(200, 30)]
        report = summarise(records, warmup=2)
        self.assertEqual(report["frames"], 3)
        self.assertAlmostEqual(report["processed_fps"], 6.667, places=3)
        self.assertEqual(report["timings_ms"]["pipeline_total"]["median"], 20)
        self.assertEqual(report["capture_sizes"], [(1920, 1080)])
        self.assertEqual(report["processed_sizes"], [(1280, 720)])
        self.assertEqual(report["plate_count_frames"], {0: 2, 2: 1})
        self.assertNotIn("private", str(report))

    def test_missing_intervals_do_not_turn_inverse_latency_into_fps(self):
        report = summarise([self.record(None, 5), self.record(None, 5)], warmup=0)
        self.assertIsNone(report["processed_fps"])
        self.assertIsNone(report["measurement_seconds"])

    def test_empty_window_and_negative_warmup_fail(self):
        with self.assertRaises(ValueError):
            summarise([self.record(None, 5)], warmup=1)
        with self.assertRaises(ValueError):
            summarise([], warmup=-1)

    def test_invalid_stage_values_are_excluded(self):
        record = self.record(None, float("nan"))
        record["timings_ms"].update(valid=4, unavailable=None, impossible=-1)
        self.assertEqual(summarise([record], 0)["timings_ms"], {
            "valid": {"samples": 1, "mean": 4, "median": 4, "p95": 4, "min": 4, "max": 4}})


if __name__ == "__main__":
    unittest.main()
