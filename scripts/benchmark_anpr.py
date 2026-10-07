#!/usr/bin/env python3
"""Summarise real ANPR frame records without treating inference latency as FPS."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    lo, hi = math.floor(index), math.ceil(index)
    percentile = ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {"samples": len(values), "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3), "p95": round(percentile, 3),
            "min": round(ordered[0], 3), "max": round(ordered[-1], 3)}


def summarise(records, warmup=10):
    if warmup < 0:
        raise ValueError("Warmup must be nonnegative.")
    measured = records[warmup:]
    if not measured:
        raise ValueError("No measured frames remain after warmup.")
    stages = {}
    for record in measured:
        for key, value in record.get("timings_ms", {}).items():
            if finite_number(value) and value >= 0:
                stages.setdefault(key, []).append(value)
    # The first retained interval started before the measurement window.
    intervals = [record.get("processing_interval_ms") for record in measured[1:]]
    complete_intervals = (bool(intervals)
                          and all(finite_number(value) and value > 0 for value in intervals))
    throughput = len(intervals) * 1000 / sum(intervals) if complete_intervals else None
    sizes = lambda name: sorted({tuple(record["frame_size"][name]) for record in measured
                                 if isinstance(record.get("frame_size", {}).get(name), list)})
    plates = Counter(len(record.get("plates", [])) for record in measured)
    return {"frames": len(measured), "warmup_frames": warmup,
            "backend": sorted({record.get("backend", "unknown") for record in measured}),
            "mode": sorted({record.get("mode", "unknown") for record in measured}),
            "processed_fps": round(throughput, 3) if throughput is not None else None,
            "measurement_seconds": round(sum(intervals) / 1000, 3) if complete_intervals else None,
            "throughput_note": "Measured completion intervals; includes capture waits and preview work."
                               if complete_intervals else "Completion intervals unavailable; no FPS estimate from latency.",
            "capture_sizes": sizes("capture"), "processed_sizes": sizes("processed"),
            "plate_count_frames": dict(sorted(plates.items())),
            "timings_ms": {key: distribution(values) for key, values in sorted(stages.items())},
            "camera": measured[-1].get("camera")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSONL emitted by anpr.py --max-frames.")
    parser.add_argument("--warmup", type=int, default=10, help="Discard this many initial frame records.")
    args = parser.parse_args(argv)
    try:
        records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("Each JSONL record must be an object.")
        report = summarise(records, args.warmup)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
