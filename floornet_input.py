#!/usr/bin/env python
"""
FloorNet input middleware: depth.npz -> point cloud -> FloorNet TFRecord.

Example:
  python floornet_input.py depth.npz --name my_scan
  python floornet_input.py depth.npz --ply out/my_scan.ply --tfrecord data/my_scan_val.tfrecords

Then run inference (TensorFlow 1.x / project train.py):
  python train.py --task=test --dataset= --checkpoint_dir=checkpoint/floornet_hybrid1_branch0123_wsf \\
      --keyname=floornet_hybrid1_branch0123_wsf --numTestingImages=1
  (Point custom val TFRecord via editing filenames in evaluate.py or symlink as data/Tango_val.tfrecords)
"""

from __future__ import print_function

import argparse
import os
import sys

from middleware.pipeline import convert_npz_scan


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert depth.npz to PLY + FloorNet-ready TFRecord (input middleware)."
    )
    p.add_argument("input_npz", help="Path to depth.npz (world_points_from_depth + images, etc.)")
    p.add_argument(
        "--name",
        default=None,
        help="Scene id used in TFRecord image_path (default: input file stem)",
    )
    p.add_argument(
        "--out-dir",
        default="data/middleware",
        help="Output directory for generated artifacts",
    )
    p.add_argument("--ply", default=None, help="PLY output path (default: <out-dir>/<name>.ply)")
    p.add_argument(
        "--tfrecord",
        default=None,
        help="TFRecord path (default: data/<name>_val.tfrecords for FloorNet val naming)",
    )
    p.add_argument(
        "--format",
        choices=("all", "ply", "tfrecord", "npz"),
        default="all",
        help="Which outputs to write",
    )
    p.add_argument(
        "--conf-min",
        type=float,
        default=0.0,
        help="Minimum depth_conf to keep a point",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Pixel stride when fusing views (2 = half resolution, faster/smaller)",
    )
    p.add_argument(
        "--views",
        default=None,
        help="Comma-separated view indices to use (default: all), e.g. 0,1,2",
    )
    p.add_argument(
        "--num-points",
        type=int,
        default=50000,
        help="Points sampled for FloorNet (fixed at 50k in the original network)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling")
    p.add_argument(
        "--no-align",
        action="store_true",
        help="Skip PCA alignment (use if world frame is already top-down)",
    )
    p.add_argument(
        "--dataset-flag",
        type=int,
        default=1,
        help="TFRecord flags[0] dataset id (1 = Tango/real in FloorNet)",
    )
    p.add_argument(
        "--symlink-val",
        action="store_true",
        help="Also symlink TFRecord to data/Tango_val.tfrecords for quick test.py runs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_npz)
    if not os.path.isfile(input_path):
        print("Input not found:", input_path, file=sys.stderr)
        return 1

    name = args.name or os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    view_indices = None
    if args.views:
        view_indices = [int(x.strip()) for x in args.views.split(",") if x.strip()]

    want_ply = args.format in ("all", "ply")
    want_tf = args.format in ("all", "tfrecord")
    want_npz = args.format in ("all", "npz")

    scan_dir = out_dir
    if want_ply and args.ply:
        scan_dir = os.path.dirname(os.path.abspath(args.ply)) or out_dir

    print("Loading", input_path)
    paths = convert_npz_scan(
        input_path,
        scan_dir,
        scene_name=name,
        conf_min=args.conf_min,
        stride=max(1, args.stride),
        view_indices=view_indices,
        num_points=args.num_points,
        seed=args.seed,
        align=not args.no_align,
        dataset_flag=args.dataset_flag,
    )
    print("Fused point cloud from views", paths["meta"].get("views_used"))

    if want_ply:
        ply_path = args.ply or paths["ply"]
        if os.path.abspath(ply_path) != os.path.abspath(paths["ply"]):
            from middleware.floornet_pack import write_ply
            from middleware.depth_npz import load_depth_npz, fuse_multiview_point_cloud

            data = load_depth_npz(input_path)
            raw_points, _ = fuse_multiview_point_cloud(
                data, conf_min=args.conf_min, view_indices=view_indices, stride=max(1, args.stride)
            )
            write_ply(ply_path, raw_points, rgb_in_01=True)
        print("Wrote PLY:", ply_path if args.ply else paths["ply"])

    if want_npz:
        print("Wrote FloorNet bundle:", paths["floornet_npz"])

    if want_tf:
        tf_path = os.path.abspath(args.tfrecord or paths["tfrecord"])
        if args.tfrecord and os.path.abspath(args.tfrecord) != os.path.abspath(paths["tfrecord"]):
            from middleware.floornet_pack import write_tfrecord, pack_for_floornet
            from middleware.depth_npz import load_depth_npz, fuse_multiview_point_cloud

            data = load_depth_npz(input_path)
            raw_points, _ = fuse_multiview_point_cloud(
                data, conf_min=args.conf_min, view_indices=view_indices, stride=max(1, args.stride)
            )
            packed = pack_for_floornet(
                raw_points, num_points=args.num_points, seed=args.seed, align=not args.no_align
            )
            os.makedirs(os.path.dirname(tf_path) or ".", exist_ok=True)
            write_tfrecord(tf_path, name, packed, dataset_flag=args.dataset_flag)
        print("Wrote TFRecord:", tf_path)
        print(
            "  -> Run FloorNet test/evaluate with this file in data/ "
            "(e.g. rename or set val filenames to include it)."
        )
        if args.symlink_val:
            link = os.path.join("data", "Tango_val.tfrecords")
            os.makedirs("data", exist_ok=True)
            if os.path.lexists(link):
                os.remove(link)
            try:
                os.symlink(os.path.relpath(tf_path, "data"), link)
                print("Symlinked", link, "->", tf_path)
            except OSError:
                import shutil

                shutil.copy2(tf_path, link)
                print("Copied to", link, "(symlink not supported on this FS)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
