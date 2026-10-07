#!/usr/bin/env python3
"""Measure saturated ANPR processing capacity on identical public 720p inputs.

Each stream uses an independent spawned process and one NPU core, round-robin.
Capture, decode and network transmission are excluded: this is a processing
upper bound for these models, not a claim about a complete camera deployment.
"""

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import re
import signal
import sys
import time
import traceback

import cv2
import numpy as np


def completed_in_window(completed_at, start, deadline):
    return start <= completed_at <= deadline


def summarize_configuration(workers, duration, target_fps, backend, expected_plates):
    """Use completed frames in the common window, never inverse latency."""
    if duration <= 0 or not math.isfinite(duration) or target_fps <= 0 or not math.isfinite(target_fps):
        raise ValueError("Duration and target FPS must be finite and positive")
    if any(worker["completed_frames"] < 0 for worker in workers):
        raise ValueError("Completed frame counts must be nonnegative")
    rates = [worker["completed_frames"] / duration for worker in workers]
    valid = bool(workers) and all(
        worker["completed_frames"] > 0
        and worker["preflight_plate_count"] == expected_plates
        and worker["plate_count_distribution"] == {str(expected_plates): worker["completed_frames"]}
        for worker in workers)
    aggregate = sum(rates)
    return {"stream_count": len(workers), "measurement_seconds": duration,
            "completed_frames": sum(worker["completed_frames"] for worker in workers),
            "aggregate_processing_fps": aggregate,
            "minimum_per_stream_fps": min(rates, default=0),
            "per_stream_fps": rates, "workload_valid": valid,
            "all_streams_meet_target": valid and bool(rates) and min(rates) >= target_fps,
            "verified_npu_streams_meet_target": backend == "rknn" and valid and bool(rates) and min(rates) >= target_fps,
            "aggregate_target_fps_equivalent": aggregate / target_fps}


def capacity_summary(configurations, backend, target_fps):
    measured = [item for item in configurations if item.get("workload_valid") and "aggregate_processing_fps" in item]
    best = max(measured, key=lambda item: item["aggregate_processing_fps"], default=None)
    qualified = [item["stream_count"] for item in measured if item["verified_npu_streams_meet_target"]] if backend == "rknn" else []
    return {"backend": backend, "target_fps_per_stream": target_fps,
            "maximum_measured_aggregate_fps": best["aggregate_processing_fps"] if best else 0,
            "best_tested_concurrency": best["stream_count"] if best else None,
            "largest_tested_stream_count_meeting_target": max(qualified, default=0),
            "note": "Aggregate FPS / target is a fractional processing budget, not verified real-time streams; only tested per-stream rates verify the target. ONNX results do not establish NPU capacity."}


def prepare_frame(image):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A readable BGR sample image is required")
    scale = min(1280 / image.shape[1], 720 / image.shape[0])
    width, height = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    frame = np.full((720, 1280, 3), 114, dtype=np.uint8)
    x, y = (1280 - width) // 2, (720 - height) // 2
    frame[y:y + height, x:x + width] = resized
    return frame


def percentiles(values):
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def process_frame(module, pipeline, frame):
    start = time.perf_counter()
    result = pipeline.run(frame)
    inferred = time.perf_counter()
    annotated = module.annotate(frame, result)
    drawn = time.perf_counter()
    okay, _ = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    completed = time.perf_counter()
    if not okay:
        raise RuntimeError("JPEG encoding failed")
    stages = dict(result.get("timings_ms", {}))
    stages.update(pipeline_wall=(inferred - start) * 1000,
                  annotation=(drawn - inferred) * 1000,
                  jpeg=(completed - drawn) * 1000,
                  processing_total=(completed - start) * 1000)
    return len(result["plates"]), stages, completed


def worker(index, options, frame, signals, starts, messages):
    pipeline = None
    try:
        # Set only this child process's CPU affinity, including its existing
        # native threads. No device CPU or frequency configuration is changed.
        if options["cpu_cores"]:
            for task in Path("/proc/self/task").iterdir():
                try:
                    os.sched_setaffinity(int(task.name), options["cpu_cores"])
                except ProcessLookupError:
                    pass  # A native thread can exit between enumeration and binding.
        cv2.setNumThreads(options["opencv_threads"])
        spec = importlib.util.spec_from_file_location("capacity_anpr", options["module"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
            args = module.parse_args(["--image", options["sample"], "--models-dir", options["models_dir"],
                                      "--backend", options["backend"], "--npu-cores", str(index % 3)])
            pipeline = module.Pipeline(args)
            preflight, _, _ = process_frame(module, pipeline, frame)
            expected = 0 if options["workload"] == "empty" else 2
            if preflight != expected:
                raise RuntimeError(f"Workload preflight expected {expected} plates, detected {preflight}")
            messages.put(("ready", index, None))
            signals["warmup"].wait()
            warmup_frames = 0
            while not signals["abort"].is_set() and time.perf_counter() < starts[0] + options["warmup"]:
                process_frame(module, pipeline, frame)
                warmup_frames += 1
            messages.put(("warmed", index, None))
            signals["measure"].wait()
            start, deadline = starts[1], starts[1] + options["duration"]
            counts, stages = Counter(), defaultdict(list)
            discarded = 0
            while not signals["abort"].is_set() and time.perf_counter() < deadline:
                plate_count, timings, completed = process_frame(module, pipeline, frame)
                if not completed_in_window(completed, start, deadline):
                    discarded += 1
                    break
                counts[plate_count] += 1
                for key, value in timings.items():
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        stages[key].append(value)
            report = {"worker": index, "npu_core": index % 3 if options["backend"] == "rknn" else None, "opencv_threads": cv2.getNumThreads(),
                      "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                      "preflight_plate_count": preflight, "warmup_frames": warmup_frames,
                      "completed_frames": sum(counts.values()), "discarded_deadline_crossing_frames": discarded,
                      "measurement_seconds": options["duration"],
                      "processing_fps": sum(counts.values()) / options["duration"],
                      "plate_count_distribution": {str(key): value for key, value in sorted(counts.items())},
                      "timings_ms": {key: percentiles(values) for key, values in sorted(stages.items())},
                      "_stage_samples": dict(stages)}
            messages.put(("result", index, report))
    except BaseException as error:
        traceback.print_exc(file=sys.stderr)
        messages.put(("error", index, f"{type(error).__name__}: {error}"))
    finally:
        if pipeline is not None:
            with contextlib.redirect_stdout(sys.stderr):
                pipeline.close()


def parse_npu_load(text):
    loads = {int(core): float(load) for core, load in re.findall(r"Core\s*(\d)\s*:\s*(\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE)}
    if set(loads) != {0, 1, 2}:
        raise ValueError("NPU load file did not expose all three core percentages")
    return loads


class NPULoadSampler:
    def __init__(self, start, duration):
        self.start, self.deadline, self.next = start, start + duration, start
        self.samples, self.error = defaultdict(list), None

    def __call__(self):
        now = time.perf_counter()
        if self.error or now < self.next or now > self.deadline:
            return
        self.next = now + 0.5
        try:
            loads = parse_npu_load(Path("/sys/kernel/debug/rknpu/load").read_text())
            for core, value in loads.items():
                self.samples[core].append(value)
        except (OSError, ValueError) as error:
            self.error = f"{type(error).__name__}: {error}"

    def report(self):
        return {"available": bool(self.samples), "error": self.error,
                "cores": {str(core): {"mean": float(np.mean(values)), "max": max(values), "sample_count": len(values)}
                          for core, values in sorted(self.samples.items())}}


def await_phase(messages, phase, count, timeout, sample=None):
    results, deadline = {}, time.monotonic() + timeout
    while len(results) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {phase} workers ({len(results)}/{count})")
        if sample is not None:
            sample()
        try:
            kind, index, payload = messages.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            continue
        if kind == "error":
            raise RuntimeError(f"Worker {index}: {payload}")
        if kind != phase or index in results:
            raise RuntimeError(f"Unexpected worker message: {kind}, {index}")
        results[index] = payload
    return [results[index] for index in range(count)]


class WorkerCleanupError(RuntimeError):
    """Surviving workers make subsequent benchmark configurations unsafe."""


def cleanup_workers(processes):
    forced, remaining, failures = [], [], []
    for process in processes:
        if process.pid is None:
            continue
        process.join(timeout=2)
        if process.is_alive():
            forced.append(process.pid)
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if process.is_alive():
            remaining.append(process.pid)
        elif process.pid not in forced and getattr(process, "exitcode", 0) not in (0, None):
            failures.append((process.pid, process.exitcode))
    if remaining:
        raise WorkerCleanupError(f"Benchmark workers failed to exit after kill: PIDs {remaining}")
    if failures:
        raise RuntimeError(f"Benchmark workers exited unsuccessfully: PID/exitcode {failures}")
    return forced


def benchmark(options, frame, count):
    context = mp.get_context("spawn")  # RKNN runtime must not inherit a forked context.
    messages = context.Queue()
    signals = {name: context.Event() for name in ("warmup", "measure", "abort")}
    starts = context.Array("d", [0, 0])
    processes = [context.Process(target=worker, args=(index, options, frame, signals, starts, messages))
                 for index in range(count)]
    report = None
    try:
        for process in processes:
            process.start()
        await_phase(messages, "ready", count, 90)
        starts[0] = time.perf_counter()
        signals["warmup"].set()
        await_phase(messages, "warmed", count, options["warmup"] + 30)
        starts[1] = time.perf_counter()
        signals["measure"].set()
        sampler = NPULoadSampler(starts[1], options["duration"])
        workers = await_phase(messages, "result", count, options["duration"] + 30, sampler)
        report = summarize_configuration(workers, options["duration"], options["target_fps"],
                                         options["backend"], 0 if options["workload"] == "empty" else 2)
        pooled = defaultdict(list)
        for item in workers:
            for key, values in item.pop("_stage_samples").items():
                pooled[key].extend(values)
        report["timings_ms"] = {key: percentiles(values) for key, values in sorted(pooled.items())}
        report["hardware_npu_load_percent"] = sampler.report()
        report["workers"] = workers
        return report
    finally:
        for signal in signals.values():
            signal.set()
        try:
            forced = cleanup_workers(processes)
        finally:
            messages.close()
            messages.join_thread()
        if report is not None:
            report["workers_requiring_forced_cleanup"] = forced


def cpu_cores(value):
    if value in ("auto", "all"):
        return value
    try:
        cores = sorted({int(core) for core in value.split(",")})
        if not cores or min(cores) < 0:
            raise ValueError()
        return cores
    except ValueError:
        raise argparse.ArgumentTypeError("Use auto, all or a comma-separated list of nonnegative CPU IDs") from None


def select_cpu_cores(requested, backend, allowed, capacities):
    """Prefer the fastest kernel capacity group, without hardcoding CPU IDs."""
    if isinstance(requested, list):
        if not set(requested) <= set(allowed):
            raise ValueError("Requested CPU cores are outside this process's affinity")
        return requested
    if requested == "all" or backend != "rknn" or not capacities or set(capacities) != set(allowed):
        return allowed
    fastest = max(capacities.values())
    return [core for core in allowed if capacities[core] == fastest]


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Stream counts must be positive")
    return number


def abort_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def main(argv=None):
    # External bounded runs must release spawned RKNN contexts before exiting.
    signal.signal(signal.SIGTERM, abort_on_signal)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=positive_integer, nargs="+", default=[1, 2, 3, 6])
    parser.add_argument("--workload", choices=("empty", "plates"), default="plates")
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--target-fps", type=float, default=30)
    parser.add_argument("--backend", choices=("rknn", "onnx"), default="rknn")
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--cpu-cores", type=cpu_cores, default="auto")
    parser.add_argument("--models-dir", type=Path, default=Path("/opt/models"))
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--module", type=Path, default=Path(__file__).resolve().parents[1] / "anpr.py")
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) for value in (args.duration, args.warmup, args.target_fps)) or args.duration <= 0 or args.warmup < 0 or args.target_fps <= 0 or args.opencv_threads < 1:
        parser.error("Duration, target FPS and OpenCV threads must be positive; warmup must be nonnegative")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    capacities = {}
    for core in allowed:
        try:
            capacities[core] = int(Path(f"/sys/devices/system/cpu/cpu{core}/cpu_capacity").read_text().strip())
        except (OSError, ValueError):
            pass
    try:
        selected = select_cpu_cores(args.cpu_cores, args.backend, allowed, capacities)
    except ValueError as error:
        parser.error(str(error))
    sample = (args.sample or args.models_dir / "sample.jpg").resolve()
    frame = prepare_frame(cv2.imread(str(sample))) if args.workload == "plates" else np.zeros((720, 1280, 3), dtype=np.uint8)
    extension = ".rknn" if args.backend == "rknn" else ".onnx"
    report = {"scope": "Saturated processing upper bound for the selected models; not generic RK3588 silicon capacity",
              "backend": args.backend, "model_precision": "FP16 RKNN, unquantized" if args.backend == "rknn" else "ONNX floating point",
              "workload": args.workload, "expected_plates_per_frame": 0 if args.workload == "empty" else 2,
              "frame_size": [1280, 720], "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
              "included": ["detector", "all detected plate OCR", "annotation", "JPEG quality 80"],
              "excluded": ["initialization", "warmup", "input preparation", "capture", "decode", "network"],
              "platform": platform.platform(), "opencv_version": cv2.__version__,
              "opencv_threads_requested": args.opencv_threads, "cpu_cores_requested": args.cpu_cores,
              "cpu_cores_selected": selected or None, "kernel_cpu_capacities": capacities,
              "npu_assignment": "Independent worker processes, core 0/1/2 round-robin",
              "model_sha256": {name: checksum(args.models_dir / (name + extension)) for name in ("plate_detector", "plate_recognizer")},
              "model_asset_sha256": {name: checksum(args.models_dir / name) for name in ("model_metadata.json", "ppocr_keys_v1.txt") if (args.models_dir / name).is_file()},
              "module_sha256": checksum(args.module), "warmup_seconds": args.warmup,
              "configuration_order": args.streams, "configurations": [], "errors": []}
    options = vars(args).copy()
    options.update(module=str(args.module.resolve()), models_dir=str(args.models_dir.resolve()),
                   sample=str(sample), cpu_cores=selected)
    for count in args.streams:
        print(f"Benchmarking {count} streams, {args.workload}: {args.warmup}s warmup + {args.duration}s measurement", file=sys.stderr, flush=True)
        try:
            result = benchmark(options, frame, count)
            report["configurations"].append(result)
            print(f"{count} streams: {result['aggregate_processing_fps']:.2f} aggregate FPS; weakest stream {result['minimum_per_stream_fps']:.2f} FPS", file=sys.stderr, flush=True)
        except Exception as error:
            report["errors"].append({"stream_count": count, "error": f"{type(error).__name__}: {error}"})
            traceback.print_exc(file=sys.stderr)
            if isinstance(error, WorkerCleanupError):
                break  # Surviving workers could contaminate later measurements.
    report["capacity"] = capacity_summary(report["configurations"], args.backend, args.target_fps)
    print(json.dumps(report, indent=2), flush=True)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
