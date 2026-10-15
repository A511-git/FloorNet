#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer.py - FloorNet Batch & Single-File Inference Pipeline

Processes input 3D Point Clouds (.ply) or Depth predictions (.npz), runs FloorNet
neural network inference and vector floorplan reconstruction, and exports
complete per-scene bundles preserving folder hierarchy.

Per-scene output bundle:
  <output_dir>/<relative_path>/<scene_name>/
    |-- pointcloud.ply              (Sampled 50k point cloud with RGB)
    |-- floorplan.txt               (Reconstructed vector wall lines)
    |-- floorplan_representation.txt(Full vector floorplan with rooms, doors, icons)
    |-- doors.txt                   (Detected door coordinates & types)
    |-- icons.txt                   (Detected furniture icons & types)
    |-- floorplan.png               (High-resolution 1000x1000 rendered floorplan)
    |-- density.png                 (256x256 top-down projection density map)
    \-- [optional] scene.tfrecords  (FloorNet dataset protobuf record)
"""

from __future__ import print_function

import os
import sys
import argparse
import copy
import time
import numpy as np
import cv2

# Ensure FloorNet root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import sigmoid, softmax, writePointCloud
from floorplan_utils import (
    NUM_POINTS,
    HEIGHT,
    WIDTH,
    NUM_WALL_CORNERS,
    NUM_ICONS,
    NUM_ROOMS,
    drawResultImageFinal,
    drawResultImage,
    resizeResult,
    writeRepresentation,
    getLabelIconMap,
    calcLineDim,
)
from prepare_floornet_tfrecords import (
    load_ply,
    load_npz,
    prepare_points_and_indices,
    write_tfrecord,
)


SUPPORTED_EXTENSIONS = ('.ply', '.npz')
DEFAULT_CHECKPOINT = os.path.join(ROOT_DIR, 'checkpoint', 'floornet_hybrid1_branch0123_wsf')


def safe_makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


# =====================================================================
# Scan Discovery & Directory Tree Preservation
# =====================================================================

def discover_scans(input_path, recursive=True):
    """
    Finds all valid .ply and .npz scans.
    Returns:
        List of tuples: (full_file_path, rel_subfolder, scene_stem)
    """
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise IOError("Input path does not exist: %s" % input_path)

    scans = []

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported file extension '%s'. Must be .ply or .npz" % ext)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        # Clean up suffix like '_predictions' if present
        clean_stem = stem.replace('_predictions', '')
        scans.append((input_path, "", clean_stem))
        return scans

    # If directory: walk files
    for root, dirs, files in os.walk(input_path):
        if not recursive and root != input_path:
            continue
        rel_root = os.path.relpath(root, input_path)
        if rel_root == ".":
            rel_root = ""

        # Sort for deterministic processing order
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, f)
                stem = os.path.splitext(f)[0].replace('_predictions', '')
                scans.append((full_path, rel_root, stem))

    if len(scans) == 0:
        raise ValueError("No .ply or .npz files found in %s" % input_path)

    return scans


# =====================================================================
# Model Session & Graph Management
# =====================================================================

class FloorNetModel(object):
    """
    Encapsulates TensorFlow graph and restored session to avoid
    re-initialization overhead during batch processing.
    """
    def __init__(self, checkpoint_dir, gpu_id='0', branches='0123', hybrid='1'):
        import tensorflow as tf
        from train import build_graph

        self.tf = tf
        self.gpu_id = str(gpu_id)
        self.branches = branches
        self.hybrid = hybrid
        self.checkpoint_dir = checkpoint_dir

        if self.gpu_id == '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id

        # Setup FloorNet options namespace
        class Options(object):
            pass

        self.options = Options()
        self.options.gpu_id = ['0'] if self.gpu_id != '-1' else ['0']
        self.options.branches = self.branches
        self.options.hybrid = self.hybrid
        self.options.batchSize = 1
        self.options.numPoints = NUM_POINTS
        self.options.numInputChannels = 7
        self.options.numFinalChannels = 256
        self.options.outputLayers = 'two'
        self.options.kernelSize = 11
        self.options.cornerLossType = 'sigmoid'
        self.options.slice = False
        self.options.checkpoint_dir = self.checkpoint_dir
        self.options.augmentation = 'wsf'
        self.options.sumScale = 10
        self.options.poolingTypes = 'sssmm'
        self.options.debug = -1
        self.options.prefix = 'floornet'
        self.options.loss = '01234'

        print("Building TensorFlow graph for FloorNet (branches=%s)..." % self.branches)
        self.tf.reset_default_graph()

        # Placeholders for direct in-memory execution
        self.points_ph = self.tf.placeholder(self.tf.float32, shape=(1, NUM_POINTS, 7), name='input_points')
        self.indices_ph = self.tf.placeholder(self.tf.int32, shape=(1, NUM_POINTS), name='input_indices')
        self.flags_ph = self.tf.placeholder(self.tf.int64, shape=(1, 2), name='input_flags')

        input_dict = {
            'points': self.points_ph,
            'point_indices': self.indices_ph,
            'flags': self.flags_ph,
        }

        self.pred_dict, self.debug_dict = build_graph(self.options, input_dict)

        config = self.tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.sess = self.tf.Session(config=config)
        self.sess.run(self.tf.global_variables_initializer())

        # Restore weights
        var_to_restore = [v for v in self.tf.global_variables() if 'is_training' not in v.name]
        loader = self.tf.train.Saver(var_to_restore)

        ckpt_file = os.path.join(self.checkpoint_dir, 'checkpoint.ckpt')
        if not (os.path.exists(ckpt_file + '.index') or os.path.exists(ckpt_file)):
            # Check if there is an index inside directory
            ckpt_state = self.tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt_state and ckpt_state.model_checkpoint_path:
                ckpt_file = ckpt_state.model_checkpoint_path
            else:
                raise IOError("Checkpoint files not found in: %s" % self.checkpoint_dir)

        print("Restoring checkpoint: %s" % ckpt_file)
        loader.restore(self.sess, ckpt_file)
        print("Model restored successfully.")

    def predict(self, points_6ch, point_indices):
        """
        Runs neural network forward pass on 50k points + indices.
        Returns:
            pred_wc: Wall corner heatmaps (256, 256, 13)
            pred_oc: Door corner heatmaps (256, 256, 4)
            pred_ic: Icon corner heatmaps (256, 256, 4)
            pred_icon: Icon segmentation probabilities (256, 256, 13)
            pred_room: Room segmentation probabilities (256, 256, 16)
        """
        # Append 1 constant channel to points (7 channels total)
        points_7ch = np.concatenate([points_6ch, np.ones((NUM_POINTS, 1), dtype=np.float32)], axis=-1)

        feed_dict = {
            self.points_ph: np.expand_dims(points_7ch, 0),
            self.indices_ph: np.expand_dims(point_indices.astype(np.int32), 0),
            self.flags_ph: np.array([[0, 1]], dtype=np.int64),
        }

        preds = self.sess.run(self.pred_dict, feed_dict=feed_dict)

        corners = preds['corner'][0]  # (256, 256, 21)
        pred_wc = sigmoid(corners[:, :, :NUM_WALL_CORNERS])
        pred_oc = sigmoid(corners[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4])
        pred_ic = sigmoid(corners[:, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8])

        pred_icon = softmax(preds['icon'][0])
        pred_room = softmax(preds['room'][0])

        return pred_wc, pred_oc, pred_ic, pred_icon, pred_room

    def close(self):
        if self.sess:
            self.sess.close()


# =====================================================================
# Floorplan Geometry & Visual Export
# =====================================================================

def export_reconstruction(output_dir, scene_name, result_pred, density_img):
    """
    Exports vector floorplan, doors, icons, and renders visualization.
    """
    safe_makedirs(output_dir)
    labelIconMap = getLabelIconMap()

    # 1. Export doors.txt
    door_lines = result_pred.get('door', [[], []])[1]
    door_points = result_pred.get('door', [[], []])[0]
    doors_file = os.path.join(output_dir, 'doors.txt')
    with open(doors_file, 'w') as f:
        f.write("# Door segments: x1 y1 x2 y2\n")
        for line in door_lines:
            p1 = door_points[line[0]]
            p2 = door_points[line[1]]
            f.write("%d\t%d\t%d\t%d\tdoor\n" % (int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))

    # 2. Export icons.txt
    icon_boxes = result_pred.get('icon', [[], [], []])[1]
    icon_labels = result_pred.get('icon', [[], [], []])[2]
    icon_points = result_pred.get('icon', [[], [], []])[0]
    icons_file = os.path.join(output_dir, 'icons.txt')
    with open(icons_file, 'w') as f:
        f.write("# Icons: min_x min_y max_x max_y type\n")
        for box, lbl in zip(icon_boxes, icon_labels):
            pts = np.array([icon_points[pt_idx][:2] for pt_idx in box]).astype(np.int32)
            mins = pts.min(0)
            maxs = pts.max(0)
            type_name = labelIconMap.get(lbl + 1, 'icon_%d' % lbl)
            f.write("%d\t%d\t%d\t%d\t%s\n" % (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1]), type_name))

    # 3. High-resolution rendered 2D floorplan image & full representation
    wall_data = result_pred.get('wall', [[], []])
    if len(wall_data) >= 2 and len(wall_data[1]) > 0:
        new_w, new_h = 1000, 1000
        res_copy = copy.deepcopy(result_pred)
        resizeResult(res_copy, new_w, new_h, WIDTH, HEIGHT)

        try:
            rendered_img = drawResultImageFinal(new_w, new_h, res_copy)
            img_path = os.path.join(output_dir, 'floorplan.png')
            cv2.imwrite(img_path, rendered_img)
        except Exception as e:
            # Fallback to standard 256x256 renderer
            try:
                rendered_img, _ = drawResultImage(WIDTH, HEIGHT, result_pred)
                cv2.imwrite(os.path.join(output_dir, 'floorplan.png'), rendered_img)
            except Exception:
                pass

        rep_path = os.path.join(output_dir, 'floorplan_representation.txt')
        writeRepresentation(rep_path, new_w, new_h, res_copy)


# =====================================================================
# Main Inference Driver
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FloorNet Inference: Process .ply / .npz files or folders and export vector floorplans."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input .ply/.npz file or folder of scans.")
    parser.add_argument("-o", "--output", default="output",
                        help="Output directory where per-scene folders are saved (default: 'output').")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to FloorNet model checkpoint directory.")
    parser.add_argument("--gpu", default="0",
                        help="GPU device id to use ('0', '1', or '-1' for CPU).")
    parser.add_argument("--solver", choices=["ip", "qp"], default="ip",
                        help="Floorplan reconstruction solver: 'ip' (PuLP/CBC open-source, default) or 'qp' (Gurobi, requires license).")
    parser.add_argument("--wall-threshold", type=float, default=0.5,
                        help="Corner heatmap detection threshold for walls (default: 0.5).")
    parser.add_argument("--min-conf", type=float, default=0.1,
                        help="Confidence threshold for depth.npz points (default: 0.1).")
    parser.add_argument("--views", default=None,
                        help="Comma-separated view indices for depth.npz, e.g. '0,1,2'.")
    parser.add_argument("--num-points", type=int, default=50000,
                        help="Number of points to sample (default: 50000).")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false",
                        help="Do not scan subfolders recursively.")
    parser.add_argument("--save-tfrecord", action="store_true",
                        help="Also save a FloorNet .tfrecords file inside each scene output folder.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Abort batch execution immediately if any scene fails.")

    # Density projection & orientation helper flags (preventing halved/lost top-down views)
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

    parser.set_defaults(recursive=True, robust_bounds=True)
    args = parser.parse_args()

    # Select integer programming solver
    if args.solver == "ip":
        from IP import reconstructFloorplan
    else:
        try:
            from QP import reconstructFloorplan
        except ImportError:
            print("Warning: Gurobi (gurobipy) not available for QP solver. Falling back to IP solver.")
            from IP import reconstructFloorplan

    # 1. Discover scans
    scans = discover_scans(args.input, recursive=args.recursive)
    total_scans = len(scans)
    print("Discovered %d scan(s) to process." % total_scans)

    # 2. Initialize Model once
    model = FloorNetModel(checkpoint_dir=args.checkpoint, gpu_id=args.gpu)

    output_root = os.path.abspath(args.output)
    safe_makedirs(output_root)

    successful = 0
    failed = []

    start_time = time.time()

    for idx, (scan_path, rel_subdir, scene_name) in enumerate(scans, 1):
        # Target scene directory maintaining input hierarchy
        if rel_subdir:
            scene_dir = os.path.join(output_root, rel_subdir, scene_name)
        else:
            scene_dir = os.path.join(output_root, scene_name)

        safe_makedirs(scene_dir)
        print("\n[%d/%d] Processing: %s" % (idx, total_scans, scan_path))
        print("  -> Output folder: %s" % scene_dir)

        scan_t0 = time.time()
        try:
            ext = os.path.splitext(scan_path)[1].lower()

            # Load points
            if ext == '.ply':
                raw_pts, raw_cols = load_ply(scan_path)
            elif ext == '.npz':
                views = [int(v.strip()) for v in args.views.split(",")] if args.views else None
                raw_pts, raw_cols = load_npz(scan_path, min_conf=args.min_conf, views=views)
            else:
                raise ValueError("Unsupported format: %s" % ext)

            # Sample 50k points & 2D density mapping with robust orientation & contrast enhancement
            features_6ch, point_indices, density_vis = prepare_points_and_indices(
                raw_pts,
                raw_cols,
                num_points=args.num_points,
                seed=0,
                up_axis=args.up_axis,
                percentile_cap=args.percentile_cap,
                gamma=args.gamma,
                dilate_kernel=args.dilate_kernel,
                robust_bounds=args.robust_bounds
            )

            # Save point cloud (.ply) into scene output folder
            pts_out_path = os.path.join(scene_dir, 'pointcloud.ply')
            # Extract 3D coords and 0..255 uint8 colors for PLY writing
            pts_xyz = features_6ch[:, :3]
            cols_rgb = np.clip(features_6ch[:, 3:6] * 255.0, 0, 255).astype(np.uint8)
            writePointCloud(pts_out_path, np.concatenate([pts_xyz, cols_rgb / 255.0], axis=-1))

            # Save top-down density image for visual inspection
            cv2.imwrite(os.path.join(scene_dir, 'density.png'), density_vis)

            # Run Neural Network prediction
            pred_wc, pred_oc, pred_ic, pred_icon, pred_room = model.predict(features_6ch, point_indices)

            # Reconstruct floorplan polygon via Quadratic/Integer Programming
            solver_prefix = (scene_dir + os.sep).replace("\\", "/")
            try:
                result_pred = reconstructFloorplan(
                    pred_wc,
                    pred_oc,
                    pred_ic,
                    pred_icon,
                    pred_room,
                    densityImage=density_vis,
                    output_prefix=solver_prefix,
                    debug_prefix=solver_prefix,
                    heatmapValueThresholdWall=args.wall_threshold
                )
            except TypeError:
                result_pred = reconstructFloorplan(
                    pred_wc,
                    pred_oc,
                    pred_ic,
                    pred_icon,
                    pred_room,
                    densityImage=density_vis,
                    debug_prefix=solver_prefix,
                    heatmapValueThresholdWall=args.wall_threshold
                )

            # Export final rendered image, representation, doors, and icons
            if result_pred and isinstance(result_pred, dict):
                export_reconstruction(scene_dir, scene_name, result_pred, density_vis)

            # Optional: save TFRecord
            if args.save_tfrecord:
                tfrec_path = os.path.join(scene_dir, scene_name + '.tfrecords')
                write_tfrecord(tfrec_path, scene_name, features_6ch, point_indices)

            elapsed = time.time() - scan_t0
            print("  Done in %.2fs" % elapsed)
            successful += 1

        except Exception as err:
            elapsed = time.time() - scan_t0
            print("  FAILED (%s) after %.2fs: %s" % (scan_path, elapsed, str(err)), file=sys.stderr)
            failed.append((scan_path, str(err)))
            if args.fail_fast:
                model.close()
                raise

    model.close()
    total_elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("Inference Pipeline Finished in %.2fs" % total_elapsed)
    print("  Total Scans: %d" % total_scans)
    print("  Successful:  %d" % successful)
    print("  Failed:      %d" % len(failed))
    if failed:
        print("\nFailed Scans:")
        for path, err in failed:
            print("  - %s: %s" % (path, err))
    print("=" * 50)

    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
