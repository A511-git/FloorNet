#!/usr/bin/env python
"""
prepare_floornet_tfrecords.py

Unified converter that turns EITHER:
  1. A 3D Point Cloud (.ply file, e.g. Scenes_reception.ply)
  OR
  2. A Depth predictions file (.npz file, e.g. Scenes_reception_predictions.npz)
into FloorNet-ready TFRecords (.tfrecords) for direct model inference.

Features:
  - Auto-detects input file format (.ply vs .npz)
  - Built-in fast binary & ASCII PLY reader (no external 3D libraries required)
  - Samples exactly 50,000 points (FloorNet input specification)
  - Computes 2D top-down density mapping (50,000 point_indices into 256x256 grid)
  - Writes standard FloorNet TFRecord format compatible with RecordReader.py
"""

from __future__ import print_function

import os
import sys
import argparse
import re
import numpy as np


NUM_POINTS = 50000
HEIGHT = 256
WIDTH = 256
MAX_NUM_CORNERS = 300
NUM_INPUT_CHANNELS = 7  # 6 stored (XYZ + RGB), RecordReader appends 1 constant channel


# =====================================================================
# PLY File Reader (Supports both Binary Little-Endian and ASCII PLY)
# =====================================================================

def load_ply(filename):
    """
    Loads 3D points and RGB colors from a .ply file.
    Returns:
        points: (N, 3) float32 array
        colors: (N, 3) float32 array in range [0.0, 1.0]
    """
    if not os.path.exists(filename):
        raise IOError("PLY file not found: %s" % filename)

    print("Loading PLY file: %s" % filename)
    with open(filename, 'rb') as f:
        # Parse PLY Header
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        is_binary = False
        num_vertices = 0
        properties = []

        in_vertex_element = False
        for line in header_lines:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == 'format':
                if 'binary_little_endian' in tokens[1]:
                    is_binary = True
                elif 'ascii' in tokens[1]:
                    is_binary = False
                else:
                    raise ValueError("Unsupported PLY format: %s" % tokens[1])
            elif tokens[0] == 'element':
                if tokens[1] == 'vertex':
                    num_vertices = int(tokens[2])
                    in_vertex_element = True
                else:
                    in_vertex_element = False
            elif tokens[0] == 'property' and in_vertex_element:
                prop_type = tokens[1]
                prop_name = tokens[2]
                properties.append((prop_name, prop_type))

        if num_vertices == 0:
            raise ValueError("No vertices found in PLY header: %s" % filename)

        print("  Vertices count: %d, format: %s" % (num_vertices, "binary" if is_binary else "ascii"))

        # Map PLY property types to numpy dtypes
        type_map = {
            'char': 'i1', 'int8': 'i1',
            'uchar': 'u1', 'uint8': 'u1',
            'short': 'i2', 'int16': 'i2',
            'ushort': 'u2', 'uint16': 'u2',
            'int': 'i4', 'int32': 'i4',
            'uint': 'u4', 'uint32': 'u4',
            'float': 'f4', 'float32': 'f4',
            'double': 'f8', 'float64': 'f8'
        }

        if is_binary:
            dtype_list = []
            for name, ptype in properties:
                np_t = type_map.get(ptype, 'f4')
                dtype_list.append((name, '<' + np_t))

            vertex_dtype = np.dtype(dtype_list)
            data = np.fromfile(f, dtype=vertex_dtype, count=num_vertices)

            x = data['x'].astype(np.float32)
            y = data['y'].astype(np.float32)
            z = data['z'].astype(np.float32)
            points = np.stack([x, y, z], axis=-1)

            # Colors if available
            color_names = [name for name, _ in properties if name in ['red', 'green', 'blue', 'r', 'g', 'b']]
            if len(color_names) >= 3:
                r_name = 'red' if 'red' in data.dtype.names else 'r'
                g_name = 'green' if 'green' in data.dtype.names else 'g'
                b_name = 'blue' if 'blue' in data.dtype.names else 'b'
                r = data[r_name].astype(np.float32)
                g = data[g_name].astype(np.float32)
                b = data[b_name].astype(np.float32)
                colors = np.stack([r, g, b], axis=-1)
                if colors.max() > 1.0:
                    colors /= 255.0
            else:
                colors = np.full_like(points, 0.5, dtype=np.float32)

        else:
            # ASCII PLY parsing
            lines = f.readlines()
            points_list = []
            colors_list = []
            has_colors = False

            prop_names = [name for name, _ in properties]
            x_idx = prop_names.index('x')
            y_idx = prop_names.index('y')
            z_idx = prop_names.index('z')

            r_idx = prop_names.index('red') if 'red' in prop_names else (prop_names.index('r') if 'r' in prop_names else -1)
            g_idx = prop_names.index('green') if 'green' in prop_names else (prop_names.index('g') if 'g' in prop_names else -1)
            b_idx = prop_names.index('blue') if 'blue' in prop_names else (prop_names.index('b') if 'b' in prop_names else -1)

            if r_idx >= 0 and g_idx >= 0 and b_idx >= 0:
                has_colors = True

            for line in lines[:num_vertices]:
                vals = line.decode('ascii', errors='ignore').split()
                if not vals:
                    continue
                points_list.append([float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])])
                if has_colors:
                    colors_list.append([float(vals[r_idx]), float(vals[g_idx]), float(vals[b_idx])])

            points = np.array(points_list, dtype=np.float32)
            if has_colors:
                colors = np.array(colors_list, dtype=np.float32)
                if colors.max() > 1.0:
                    colors /= 255.0
            else:
                colors = np.full_like(points, 0.5, dtype=np.float32)

    # Filter out NaNs and Infs
    valid_mask = np.isfinite(points).all(axis=-1)
    points = points[valid_mask]
    colors = colors[valid_mask]

    return points, colors


# =====================================================================
# NPZ File Reader (Extracts 3D World Points from Depth predictions)
# =====================================================================

def load_npz(filename, min_conf=0.0, views=None):
    """
    Loads 3D world points and colors from a depth.npz file.
    Returns:
        points: (N, 3) float32 array
        colors: (N, 3) float32 array in range [0.0, 1.0]
    """
    if not os.path.exists(filename):
        raise IOError("NPZ file not found: %s" % filename)

    print("Loading NPZ file: %s" % filename)
    data = np.load(filename)

    if "world_points_from_depth" not in data:
        raise KeyError("Expected 'world_points_from_depth' array in %s" % filename)

    raw_pts = data["world_points_from_depth"]
    if raw_pts.ndim == 3:
        raw_pts = np.expand_dims(raw_pts, 0)

    num_views, H, W, _ = raw_pts.shape
    view_indices = list(range(num_views)) if views is None else [v for v in views if 0 <= v < num_views]

    has_conf = "depth_conf" in data
    has_depth = "depth" in data
    has_images = "images" in data

    raw_images = None
    if has_images:
        imgs = data["images"]
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, 0)
        if imgs.shape[1] == 3 and imgs.ndim == 4:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        raw_images = imgs.astype(np.float32)
        if raw_images.max() > 1.0:
            raw_images /= 255.0

    all_pts, all_cols = [], []
    for v in view_indices:
        pts_v = raw_pts[v]
        mask = np.isfinite(pts_v).all(axis=-1) & (np.abs(pts_v).sum(axis=-1) > 1e-4)

        if has_depth:
            d_v = data["depth"][v]
            if d_v.ndim == 3:
                d_v = d_v.squeeze(-1)
            mask &= (d_v > 0) & np.isfinite(d_v)

        if has_conf and min_conf > 0.0:
            mask &= (data["depth_conf"][v] >= min_conf)

        all_pts.append(pts_v[mask])
        if raw_images is not None:
            all_cols.append(raw_images[v][mask])

    points = np.concatenate(all_pts, axis=0).astype(np.float32)
    colors = np.concatenate(all_cols, axis=0).astype(np.float32) if raw_images is not None else np.full_like(points, 0.5)

    return points, colors


# =====================================================================
# Preprocessing: 50k Sampling & 2D Top-Down Projection Mapping
# =====================================================================

# =====================================================================
# Preprocessing: 50k Sampling & 2D Top-Down Projection Mapping
# =====================================================================

def prepare_points_and_indices(
    points,
    colors,
    num_points=50000,
    seed=0,
    up_axis="auto",
    padding=0.05,
    percentile_cap=98.0,
    gamma=0.8,
    dilate_kernel=1,
    robust_bounds=True
):
    """
    1. Samples exactly num_points (XYZ + RGB = 6 channels).
    2. Auto-detects/aligns vertical up-axis (e.g. Y-up vs Z-up) to prevent top-down view from being halved or projected sideways.
    3. Computes the 2D density indices mapping each point to the 256x256 grid.
    4. Applies percentile capping and gamma contrast curve to prevent dense clusters from washing out walls.
    """
    total_pts = len(points)
    if total_pts == 0:
        raise ValueError("Cannot prepare empty point cloud.")

    rng = np.random.RandomState(seed)
    if total_pts >= num_points:
        idx = rng.choice(total_pts, size=num_points, replace=False)
    else:
        idx = rng.choice(total_pts, size=num_points, replace=True)

    sampled_pts = points[idx].copy()
    sampled_cols = colors[idx].copy()

    # Detect / handle vertical up-axis
    rx = np.ptp(sampled_pts[:, 0])
    ry = np.ptp(sampled_pts[:, 1])
    rz = np.ptp(sampled_pts[:, 2])

    if up_axis == "auto":
        # In indoor spaces, ceiling height is usually smaller than room floor dimensions
        if ry < rz and ry < rx * 0.85:
            detected_up = "y"
        else:
            detected_up = "z"
    else:
        detected_up = up_axis.lower()

    # If Y is up (common in ARKit, RoomFormer, OpenGL), permute (X, Z, Y) so (X, Y) is ground plane and Z is height
    if detected_up == "y":
        aligned_pts = np.stack([sampled_pts[:, 0], sampled_pts[:, 2], sampled_pts[:, 1]], axis=-1)
    else:
        aligned_pts = sampled_pts

    # FloorNet channel arrangement: [X, Y, Z, R, G, B]
    features_6ch = np.concatenate([aligned_pts, sampled_cols], axis=-1).astype(np.float32)

    # Compute 2D top-view projection index map
    xs = aligned_pts[:, 0]
    ys = aligned_pts[:, 1]

    if robust_bounds:
        # Filter spatial outliers (e.g. 0.5% and 99.5%) to prevent stray points from distorting the bounding box
        min_x, max_x = np.percentile(xs, 0.5), np.percentile(xs, 99.5)
        min_y, max_y = np.percentile(ys, 0.5), np.percentile(ys, 99.5)
    else:
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

    max_range = max(max_x - min_x, max_y - min_y)
    if max_range <= 0:
        max_range = 1.0

    pad = padding * max_range
    scale = max_range + 2.0 * pad

    grid_x = np.clip(np.round(((xs - min_x + pad) / scale) * (WIDTH - 1)), 0, WIDTH - 1).astype(np.int64)
    grid_y = np.clip(np.round(((ys - min_y + pad) / scale) * (HEIGHT - 1)), 0, HEIGHT - 1).astype(np.int64)

    point_indices = (grid_y * WIDTH + grid_x).astype(np.int64)

    # Compute enhanced top-down density map
    coordinates = np.stack([grid_x, grid_y], axis=-1)
    unique_coords, counts = np.unique(coordinates, return_counts=True, axis=0)

    if percentile_cap is not None and percentile_cap > 0 and len(counts) > 0:
        cap_val = max(float(np.percentile(counts, percentile_cap)), 1.0)
        norm_counts = np.clip(counts.astype(np.float32), 0, cap_val) / cap_val
    else:
        max_val = max(float(counts.max()), 1.0)
        norm_counts = counts.astype(np.float32) / max_val

    if gamma is not None and gamma > 0 and gamma != 1.0:
        norm_counts = np.power(norm_counts, gamma)

    density_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    density_map[unique_coords[:, 1], unique_coords[:, 0]] = norm_counts
    density_uint8 = (np.clip(density_map, 0, 1.0) * 255.0).astype(np.uint8)

    if dilate_kernel > 1:
        k = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        density_uint8 = cv2.dilate(density_uint8, k, iterations=1)

    return features_6ch, point_indices, density_uint8


# =====================================================================
# TFRecord Protobuf Serializer
# =====================================================================

def write_tfrecord(output_path, scene_name, features_6ch, point_indices):
    """
    Writes a single FloorNet example into a .tfrecords file.
    """
    import tensorflow as tf

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def _bytes_feature(val):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

    def _float_feature(val):
        return tf.train.Feature(float_list=tf.train.FloatList(value=val))

    def _int64_feature(val):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

    # Empty placeholder arrays for inference (ground truth is unknown at test time)
    dummy_corners = np.zeros(MAX_NUM_CORNERS * 3, dtype=np.int64)
    dummy_icon_heatmap = np.zeros(HEIGHT * WIDTH, dtype=np.uint8).tobytes()
    dummy_room_heatmap = np.zeros(HEIGHT * WIDTH, dtype=np.uint8).tobytes()
    dummy_flags = np.array([0, 1], dtype=np.int64)

    # Dummy image feature vector for optional image branch
    sizes = [WIDTH, WIDTH // 2, WIDTH // 4, WIDTH // 8, WIDTH // 16, WIDTH // 32]
    num_channels = [7, 64, 64, 64, 128, 256]
    img_feature_len = sum([s * s * c for s, c in zip(sizes, num_channels)[1:]])
    dummy_image_feature = np.zeros(img_feature_len, dtype=np.float32)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_path': _bytes_feature(scene_name.encode('utf-8')),
        'points': _float_feature(features_6ch.reshape(-1)),
        'point_indices': _int64_feature(point_indices.reshape(-1)),
        'corner': _int64_feature(dummy_corners),
        'num_corners': _int64_feature([0]),
        'icon': _bytes_feature(dummy_icon_heatmap),
        'room': _bytes_feature(dummy_room_heatmap),
        'image': _float_feature(dummy_image_feature),
        'flags': _int64_feature(dummy_flags),
    }))

    # Use compatible writer
    writer_cls = getattr(tf.io, 'TFRecordWriter', None) or getattr(tf.python_io, 'TFRecordWriter')
    with writer_cls(output_path) as writer:
        writer.write(example.SerializeToString())

    print("Successfully generated TFRecord: %s" % output_path)
    print("  Scene name: %s" % scene_name)
    print("  Points tensor shape: (50000, 6)")
    print("  Point indices shape: (50000,)")
    print("  File size: %.2f MB" % (os.path.getsize(output_path) / (1024.0 * 1024.0)))


# =====================================================================
# Main CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert .ply (point cloud) or .npz (depth predictions) into a FloorNet TFRecord for inference."
    )
    parser.add_argument("input", help="Path to input .ply or .npz file")
    parser.add_argument("-o", "--output", default=None, help="Output .tfrecords file path (default: data/<name>_val.tfrecords)")
    parser.add_argument("--scene-name", default=None, help="Scene identifier name (default: derived from filename)")
    parser.add_argument("--up-axis", choices=["auto", "z", "y"], default="auto",
                        help="Vertical axis of point cloud ('auto', 'z', or 'y', default: 'auto'). Prevents sideways/halved top-down views.")
    parser.add_argument("--percentile-cap", type=float, default=98.0,
                        help="Point count saturation percentile cap (default: 98.0). Prevents dense cluster outliers from washing out walls.")
    parser.add_argument("--gamma", type=float, default=0.8,
                        help="Gamma contrast curve (< 1.0 boosts faint wall pixels, default: 0.8).")
    parser.add_argument("--dilate-kernel", type=int, default=1,
                        help="Dilation kernel size (e.g. 2) to thicken faint/sparse wall dots into solid lines (default: 1).")
    parser.add_argument("--no-robust-bounds", dest="robust_bounds", action="store_false",
                        help="Disable robust 0.5%-99.5% percentile bounding box outlier rejection.")

    parser.set_defaults(robust_bounds=True)
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    ext = os.path.splitext(input_path)[1].lower()

    scene_name = args.scene_name or os.path.splitext(os.path.basename(input_path))[0]

    # 1. Load Point Cloud
    if ext == ".ply":
        points, colors = load_ply(input_path)
    elif ext == ".npz":
        points, colors = load_npz(input_path, min_conf=args.min_conf)
    else:
        raise ValueError("Unsupported input format '%s'. Please provide a .ply or .npz file." % ext)

    print("Loaded %d points from %s" % (len(points), input_path))

    # 2. Sample 50k points and compute 2D projection indices
    print("Sampling 50,000 points and computing 2D top-down grid mapping (up_axis=%s)..." % args.up_axis)
    features_6ch, point_indices, density_img = prepare_points_and_indices(
        points,
        colors,
        num_points=NUM_POINTS,
        seed=args.seed,
        up_axis=args.up_axis,
        percentile_cap=args.percentile_cap,
        gamma=args.gamma,
        dilate_kernel=args.dilate_kernel,
        robust_bounds=args.robust_bounds
    )

    # 3. Resolve output path
    if args.output:
        output_tfrecord = os.path.abspath(args.output)
    else:
        output_tfrecord = os.path.abspath(os.path.join("data", scene_name + "_val.tfrecords"))

    # 4. Serialize to TFRecord
    write_tfrecord(output_tfrecord, scene_name, features_6ch, point_indices)

    print("\nNext step: Run FloorNet inference on this scene!")
    print("Example command:")
    print("  python train.py --task=evaluate --dataset=%s --separateIconLoss" % scene_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
