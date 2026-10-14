#!/usr/bin/env python
"""
Batch inference: depth.npz (file or folder) -> per-scan output folder -> FloorNet.

Example:
  python infer.py --input ./scans --output ./results
  python infer.py --input ./scans/x.npz --output ./results

Layout (flat input folder with x.npz, y.npz):
  results/
    x/
      x.ply
      x.tfrecords
      x_floornet.npz
      x_floorplan_pred.png
      ...
    y/
      ...
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from middleware.model_cache import DEFAULT_CHECKPOINT_NAME, ensure_checkpoint
from middleware.pipeline import convert_npz_scan


def discover_npz_files(input_path, recursive=False):
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".npz"):
            raise ValueError("Input file must be .npz: %s" % input_path)
        return [(input_path, "")]
    if not os.path.isdir(input_path):
        raise ValueError("Input path not found: %s" % input_path)

    entries = []
    if recursive:
        for root, _dirs, files in os.walk(input_path):
            for name in sorted(files):
                if name.lower().endswith(".npz"):
                    full = os.path.join(root, name)
                    rel = os.path.relpath(root, input_path)
                    if rel == ".":
                        rel = ""
                    entries.append((full, rel))
    else:
        for name in sorted(os.listdir(input_path)):
            full = os.path.join(input_path, name)
            if os.path.isfile(full) and name.lower().endswith(".npz"):
                entries.append((full, ""))
    if not entries:
        raise ValueError("No .npz files found under %s" % input_path)
    return entries


def output_scan_dir(output_root, rel_subdir, stem):
    if rel_subdir:
        return os.path.join(output_root, rel_subdir, stem)
    return os.path.join(output_root, stem)


def run_infer_worker(tfrecord, out_dir, scene_name, checkpoint_dir, gpu, solver, python_exe):
    worker = os.path.join(REPO_ROOT, "middleware", "infer_worker.py")
    cmd = [
        python_exe,
        worker,
        "--tfrecord",
        tfrecord,
        "--out-dir",
        out_dir,
        "--scene-name",
        scene_name,
        "--checkpoint-dir",
        checkpoint_dir,
        "--gpu",
        str(gpu),
        "--solver",
        solver,
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="FloorNet batch inference from depth.npz")
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a .npz file or a folder containing .npz files (e.g. x.npz, y.npz)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="outputs",
        help="Output root directory (each scan gets a subfolder named after the file stem)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a folder, include .npz files in subdirectories (preserves relative paths)",
    )
    p.add_argument(
        "--convert-only",
        action="store_true",
        help="Only build PLY / TFRecord / NPZ; skip neural network inference",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint directory (default: checkpoint/%s, auto-downloaded if missing)"
        % DEFAULT_CHECKPOINT_NAME,
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download checkpoint from GitHub when missing",
    )
    p.add_argument(
        "--python",
        default=None,
        help="Python executable for TensorFlow inference (default: same as this script)",
    )
    p.add_argument("--gpu", default="0", help="GPU id for inference")
    p.add_argument(
        "--solver",
        choices=("ip", "qp"),
        default="ip",
        help="Floorplan reconstruction solver (ip=PuLP, qp=Gurobi)",
    )
    p.add_argument("--conf-min", type=float, default=0.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--views", default=None, help="Comma-separated view indices, e.g. 0,1,2")
    p.add_argument("--num-points", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-align", action="store_true")
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failed scan instead of continuing",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_root = os.path.abspath(args.output)
    os.makedirs(output_root, exist_ok=True)

    checkpoint_name = DEFAULT_CHECKPOINT_NAME
    if args.checkpoint:
        checkpoint_dir = os.path.abspath(args.checkpoint)
    else:
        checkpoint_dir = os.path.join(REPO_ROOT, "checkpoint", checkpoint_name)

    if not args.convert_only:
        if args.no_download:
            if not os.path.isfile(os.path.join(checkpoint_dir, "checkpoint.ckpt.index")):
                print("Checkpoint missing and --no-download set.", file=sys.stderr)
                return 1
        else:
            ensure_checkpoint(REPO_ROOT, checkpoint_name)

    view_indices = None
    if args.views:
        view_indices = [int(x.strip()) for x in args.views.split(",") if x.strip()]

    python_exe = args.python or sys.executable
    scans = discover_npz_files(args.input, recursive=args.recursive)
    ok, failed = [], []

    for npz_path, rel_subdir in scans:
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        scan_out = output_scan_dir(output_root, rel_subdir, stem)
        os.makedirs(scan_out, exist_ok=True)
        print("\n=== %s -> %s ===" % (npz_path, scan_out))
        try:
            paths = convert_npz_scan(
                npz_path,
                scan_out,
                scene_name=stem,
                conf_min=args.conf_min,
                stride=args.stride,
                view_indices=view_indices,
                num_points=args.num_points,
                seed=args.seed,
                align=not args.no_align,
            )
            if not args.convert_only:
                run_infer_worker(
                    paths["tfrecord"],
                    scan_out,
                    stem,
                    checkpoint_dir,
                    args.gpu,
                    args.solver,
                    python_exe,
                )
            ok.append(scan_out)
        except Exception as e:
            print("FAILED:", npz_path, e, file=sys.stderr)
            failed.append((npz_path, str(e)))
            if args.fail_fast:
                break

    print("\nDone. %d succeeded, %d failed." % (len(ok), len(failed)))
    for path in ok:
        print("  ok:", path)
    for path, err in failed:
        print("  fail:", path, "-", err)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
