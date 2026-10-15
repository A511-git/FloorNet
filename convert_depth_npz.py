#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_depth_npz.py

Converts a depth.npz file containing world_points_from_depth, depth_conf, images, etc.
into a 3D Point Cloud (.ply file) and optionally prepares it for FloorNet.

Input data structure expected in depth.npz:
  - world_points_from_depth : (V, H, W, 3) float32 (3D coordinates in world space)
  - depth_conf              : (V, H, W) float32 (confidence score per pixel)
  - images                  : (V, 3, H, W) or (V, H, W, 3) float32 (RGB colors)
  - depth                   : (V, H, W, 1) or (V, H, W) float32 (depth map)
  - extrinsic               : (V, 3, 4) float32 (camera extrinsics)
  - covisibility_scores     : (V,) float32
"""

from __future__ import print_function

import os
import sys
import argparse
import numpy as np


def inspect_npz(npz_path):
    """Inspect and print the contents, shapes, and dtypes of an npz file."""
    if not os.path.exists(npz_path):
        raise IOError("File not found: %s" % npz_path)

    data = np.load(npz_path)
    print("=== Contents of %s ===" % npz_path)
    for key in sorted(data.files):
        arr = data[key]
        print("  %-25s shape=%-18s dtype=%s" % (key, str(arr.shape), arr.dtype))
    print("=========================================")
    return data


def extract_point_cloud(npz_data, min_conf=0.0, views=None, num_points=0, seed=0):
    """
    Extracts 3D points and RGB colors from depth.npz data.

    Returns:
        points: (N, 3) float32 array of [X, Y, Z] in world space
        colors: (N, 3) uint8 array of [R, G, B] (or None if images not available)
    """
    if "world_points_from_depth" not in npz_data:
        raise KeyError("Missing 'world_points_from_depth' array in npz file.")

    raw_points = npz_data["world_points_from_depth"]  # (V, H, W, 3)
    has_conf = "depth_conf" in npz_data
    has_depth = "depth" in npz_data
    has_images = "images" in npz_data

    # Ensure shape is (V, H, W, 3)
    if raw_points.ndim == 3:
        raw_points = np.expand_dims(raw_points, 0)

    num_views, H, W, _ = raw_points.shape

    # Select views if specified
    if views is None:
        view_indices = list(range(num_views))
    else:
        view_indices = [v for v in views if 0 <= v < num_views]

    # Process RGB images if present
    raw_images = None
    if has_images:
        imgs = npz_data["images"]
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, 0)
        # If (V, 3, H, W), transpose to (V, H, W, 3)
        if imgs.shape[1] == 3 and imgs.ndim == 4:
            imgs = np.transpose(imgs, (0, 2, 3, 1))

        # Normalize to uint8 0..255
        if imgs.dtype != np.uint8:
            if imgs.max() <= 1.0:
                raw_images = np.clip(imgs * 255.0, 0, 255).astype(np.uint8)
            else:
                raw_images = np.clip(imgs, 0, 255).astype(np.uint8)
        else:
            raw_images = imgs

    all_pts_list = []
    all_colors_list = []

    for v in view_indices:
        pts_v = raw_points[v]  # (H, W, 3)

        # Basic validity: must be finite and non-zero
        valid_mask = np.isfinite(pts_v).all(axis=-1)
        valid_mask &= (np.abs(pts_v).sum(axis=-1) > 1e-4)

        # Depth filter
        if has_depth:
            d_v = npz_data["depth"][v]
            if d_v.ndim == 3:
                d_v = d_v.squeeze(-1)
            valid_mask &= (d_v > 0) & np.isfinite(d_v)

        # Confidence filter
        if has_conf and min_conf > 0.0:
            conf_v = npz_data["depth_conf"][v]
            valid_mask &= (conf_v >= min_conf)

        valid_pts = pts_v[valid_mask]
        all_pts_list.append(valid_pts)

        if raw_images is not None:
            colors_v = raw_images[v][valid_mask]
            all_colors_list.append(colors_v)

    if len(all_pts_list) == 0 or sum(len(p) for p in all_pts_list) == 0:
        raise ValueError("No valid 3D points found after filtering (check min_conf).")

    points = np.concatenate(all_pts_list, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors_list, axis=0) if raw_images is not None else None

    # Downsampling / Sampling (e.g. FloorNet requires 50,000 points)
    if num_points > 0:
        total_pts = len(points)
        rng = np.random.RandomState(seed)
        if total_pts >= num_points:
            indices = rng.choice(total_pts, size=num_points, replace=False)
        else:
            # If fewer points available, sample with replacement to reach requested count
            indices = rng.choice(total_pts, size=num_points, replace=True)

        points = points[indices]
        if colors is not None:
            colors = colors[indices]

    return points, colors


def write_ply(filename, points, colors=None, binary=True):
    """
    Writes 3D points and optional RGB colors to a .ply file.
    Uses fast binary writing by default.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    num_pts = len(points)

    if binary:
        # Binary Little Endian write (fast and compact)
        if colors is not None:
            dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            data = np.empty(num_pts, dtype=dtype)
            data['x'] = points[:, 0]
            data['y'] = points[:, 1]
            data['z'] = points[:, 2]
            data['red'] = colors[:, 0]
            data['green'] = colors[:, 1]
            data['blue'] = colors[:, 2]
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                "element vertex %d\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n" % num_pts
            )
        else:
            dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
            data = np.empty(num_pts, dtype=dtype)
            data['x'] = points[:, 0]
            data['y'] = points[:, 1]
            data['z'] = points[:, 2]
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                "element vertex %d\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n" % num_pts
            )

        with open(filename, 'wb') as f:
            f.write(header.encode('ascii'))
            data.tofile(f)
    else:
        # ASCII write
        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % num_pts)
            f.write("property float x\nproperty float y\nproperty float z\n")
            if colors is not None:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")

            if colors is not None:
                for p, c in zip(points, colors):
                    f.write("%.4f %.4f %.4f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))
            else:
                for p in points:
                    f.write("%.4f %.4f %.4f\n" % (p[0], p[1], p[2]))

    print("Saved PLY: %s (%d points)" % (filename, num_pts))


def compute_floornet_density_mapping(points, grid_size=256, padding_ratio=0.05):
    """
    Computes FloorNet top-view 2D density mapping:
    - 50,000 points (X, Y, Z)
    - 50,000 2D grid indices pointing into the (grid_size x grid_size) top-down density image.

    Matches FloorNet's top-view projection formula:
      maxRange = max(max(X) - min(X), max(Y) - min(Y))
      padding = padding_ratio * maxRange
      index = round((Y - minY + pad) / (maxRange + 2*pad) * 256) * 256 + ...
    """
    xs = points[:, 0]
    ys = points[:, 1]

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    max_range = max(max_x - min_x, max_y - min_y)
    if max_range <= 0:
        max_range = 1.0

    padding = padding_ratio * max_range
    scale = (max_range + 2.0 * padding)

    grid_x = np.clip(np.round(((xs - min_x + padding) / scale) * (grid_size - 1)), 0, grid_size - 1).astype(np.int64)
    grid_y = np.clip(np.round(((ys - min_y + padding) / scale) * (grid_size - 1)), 0, grid_size - 1).astype(np.int64)

    point_indices = grid_y * grid_size + grid_x

    # Also compute 2D top-down density heatmap
    density_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    for gx, gy in zip(grid_x, grid_y):
        density_map[gy, gx] += 1.0

    return point_indices, density_map


def main():
    parser = argparse.ArgumentParser(
        description="Convert depth.npz to 3D point cloud (.ply) and FloorNet point cloud inputs."
    )
    parser.add_argument("input", help="Path to input depth.npz file")
    parser.add_argument("-o", "--output", default=None, help="Output .ply file path (default: <input_stem>.ply)")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum depth confidence threshold (default: 0.0)")
    parser.add_argument("--views", type=str, default=None, help="Comma-separated view indices to use, e.g. '0,1,2'")
    parser.add_argument("--num-points", type=int, default=50000, help="Number of points to sample (default: 50000 for FloorNet, 0 for all points)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default: 0)")
    parser.add_argument("--ascii", action="store_true", help="Save PLY in ASCII format instead of binary")
    parser.add_argument("--check", action="store_true", help="Inspect and print npz contents without converting")
    parser.add_argument("--floornet-pack", action="store_true", help="Also save FloorNet-compatible pack (.npz with 50k points, 2D indices, and density map)")

    args = parser.parse_args()

    # If --check flag is passed, just inspect and exit
    if args.check:
        inspect_npz(args.input)
        return 0

    # Load npz
    npz_data = inspect_npz(args.input)

    # Views parsing
    views = None
    if args.views:
        views = [int(v.strip()) for v in args.views.split(",") if v.strip()]

    # Extract point cloud
    print("Extracting 3D points (min_conf=%.2f, views=%s, num_points=%d)..." % (args.min_conf, str(views), args.num_points))
    points, colors = extract_point_cloud(
        npz_data,
        min_conf=args.min_conf,
        views=views,
        num_points=args.num_points,
        seed=args.seed
    )
    print("Extracted %d valid 3D points (has colors: %s)" % (len(points), str(colors is not None)))

    # Output path resolution
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        output_ply = base + ".ply"
    else:
        output_ply = args.output

    # Write PLY
    write_ply(output_ply, points, colors=colors, binary=(not args.ascii))

    # Optional: package for FloorNet
    if args.floornet_pack:
        point_indices, density_map = compute_floornet_density_mapping(points)
        pack_file = os.path.splitext(output_ply)[0] + "_floornet.npz"
        np.savez_compressed(
            pack_file,
            points=points,
            point_indices=point_indices,
            density_map=density_map,
            colors=colors if colors is not None else np.zeros((0, 3), dtype=np.uint8)
        )
        print("Saved FloorNet pack: %s (points: %s, indices: %s, density_map: %s)" % (
            pack_file, str(points.shape), str(point_indices.shape), str(density_map.shape)
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
