#!/usr/bin/env python3
"""Fetch checked model sources, export a plate detector, optionally compile RKNN.

Export runs the checksum-pinned upstream PyTorch exporter. Conversion belongs on
Linux x86_64/aarch64 with RKNN-Toolkit2 2.3.2; inference needs only Toolkit Lite2.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


SOURCES_PATH = Path(__file__).with_name("model_sources.json")
DETECTOR_ROOT = "Chinese_license_plate_detection_recognition-670b765c09fff11850a7b1c542787ad66c24ae50"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source, destination):
    """Never reuse an unchecked or partial download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == source["sha256"]:
        return
    print(f"Downloading {source['url']}", flush=True)
    request = urllib.request.Request(source["url"], headers={"User-Agent": "Admiral-ANPR-demo/1.0"})
    temp_path = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual = sha256(temp_path)
        if actual != source["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {destination.name}: expected {source['sha256']}, got {actual}")
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)


def unpack(archive, directory):
    # Explicitly reject traversal, links, and device entries on Python 3.10/3.11.
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsafe archive entry: {member.name}")
        handle.extractall(directory)


def inspect_models(output_dir):
    import onnx
    import onnxruntime
    import numpy as np

    expected = {
        "plate_detector.onnx": ([1, 3, 640, 640], [1, 25200, 15]),
        "plate_recognizer.onnx": ([1, 3, 48, 320], [1, 40, 6625]),
    }
    for name, (expected_input, expected_output) in expected.items():
        model = onnx.load(str(output_dir / name))
        input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        # Paddle's supplied graph does not declare output dimensions. Verify
        # the real graph numerically rather than treating absent metadata as 0.
        session = onnxruntime.InferenceSession(str(output_dir / name), providers=["CPUExecutionProvider"])
        output_shape = list(session.run(None, {session.get_inputs()[0].name: np.zeros(expected_input, dtype=np.float32)})[0].shape)
        if input_shape != expected_input or output_shape != expected_output:
            raise RuntimeError(f"Unexpected {name} IO: input {input_shape}, output {output_shape}; expected {expected[name]}")
        if not model.graph.output[0].type.tensor_type.HasField("shape"):
            # Repair only the in-memory checker copy. The downloaded file and
            # its checksum stay intact for the official RKNN importer.
            for size in output_shape:
                model.graph.output[0].type.tensor_type.shape.dim.add().dim_value = size
        onnx.checker.check_model(model)
        print(f"Verified {name}: {input_shape} -> {output_shape}", flush=True)


def convert_model(onnx_path, rknn_path):
    from rknn.api import RKNN

    toolkit_version = importlib.metadata.version("rknn-toolkit2")
    if toolkit_version.split("+", 1)[0] != "2.3.2":
        raise RuntimeError(f"Use RKNN-Toolkit2 2.3.2; found {toolkit_version}")
    compiler = RKNN(verbose=False)
    try:
        # Both ONNX graphs consume [0,1]. RKNN takes uint8 NHWC images and does
        # exactly the same /255 normalization as the ONNX CPU path.
        compiler.config(target_platform="rk3588", mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
        ret = compiler.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load failed for {onnx_path.name}: {ret}")
        ret = compiler.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed for {onnx_path.name}: {ret}")
        ret = compiler.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"RKNN export failed for {onnx_path.name}: {ret}")
    finally:
        compiler.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--cache-dir", type=Path, help="Optional directory for checksum-verified downloads")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--convert", action="store_true", help="Export ONNX and compile FP16 .rknn models for RK3588")
    modes.add_argument("--download-only", action="store_true", help="Retain checked detector source archive/weights, OCR ONNX, dictionary and sample without exporting")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = json.loads(SOURCES_PATH.read_text())

    with tempfile.TemporaryDirectory(prefix="admiral-anpr-models-") as scratch:
        scratch_dir = Path(scratch)
        cache_dir = args.cache_dir.resolve() if args.cache_dir else scratch_dir
        archive = cache_dir / "plate_detector_source.tar.gz"
        download(sources["detector_source"], archive)
        download(sources["recognizer"], output_dir / "plate_recognizer.onnx")
        download(sources["dictionary"], output_dir / "ppocr_keys_v1.txt")
        download(sources["recognizer_license"], output_dir / "plate_recognizer_LICENSE.txt")
        unpack(archive, scratch_dir)
        source_dir = scratch_dir / DETECTOR_ROOT
        for key, hash_key in (("weights_path", "weights_sha256"), ("sample_path", "sample_sha256")):
            if sha256(source_dir / sources["detector_source"][key]) != sources["detector_source"][hash_key]:
                raise RuntimeError(f"Unexpected detector source asset: {sources['detector_source'][key]}")
        shutil.copyfile(source_dir / "imgs/single_blue.jpg", output_dir / "sample.jpg")
        shutil.copyfile(source_dir / "LICENSE", output_dir / "plate_detector_LICENSE.txt")
        if args.download_only:
            shutil.copyfile(archive, output_dir / "plate_detector_source.tar.gz")
            shutil.copyfile(source_dir / "weights/plate_detect.pt", output_dir / "plate_detector.pt")
            (output_dir / "model_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
            print(f"Checked model sources ready in {output_dir}", flush=True)
            return
        env = dict(os.environ, OMP_NUM_THREADS="1", MPLCONFIGDIR=str(scratch_dir / "matplotlib"))
        subprocess.run(
            [sys.executable, "export.py", "--weights", "weights/plate_detect.pt", "--img_size", "640", "640", "--batch_size", "1"],
            cwd=source_dir, env=env, check=True,
        )
        shutil.copyfile(source_dir / "weights/plate_detect.onnx", output_dir / "plate_detector.onnx")

    inspect_models(output_dir)
    if args.convert:
        for stem in ("plate_detector", "plate_recognizer"):
            convert_model(output_dir / f"{stem}.onnx", output_dir / f"{stem}.rknn")

    metadata = {
        "rknn_conversion": {"target": "rk3588", "toolkit_version": "2.3.2", "precision": "FP16", "do_quantization": False, "mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]], "performed": args.convert},
        "detector": {"input": [1, 3, 640, 640], "output": [1, 25200, 15], "color": "RGB", "scale": "divide by 255", "layout": "cx,cy,w,h,obj,x1,y1,x2,y2,x3,y3,x4,y4,single,double"},
        "recognizer": {"input": [1, 3, 48, 320], "output": [1, 40, 6625], "color": "BGR", "scale": "divide by 255", "dictionary_entries": 6623, "decode": "CTC greedy, blank index 0, dictionary lines then space"},
        "sources": sources,
        "artifacts": {path.name: sha256(path) for path in output_dir.iterdir() if path.suffix in (".onnx", ".rknn", ".txt", ".jpg")},
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"ANPR models ready in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
