"""
Run FloorNet on a single TFRecord and write prediction artifacts.
Invoked by infer.py (same or compatible Python with TensorFlow 1.x).
"""

from __future__ import print_function

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _save_predictions(out_dir, scene_name, pred, debug, inp, solver_module):
    import cv2
    import numpy as np

    from floorplan_utils import (
        HEIGHT,
        NUM_ICONS,
        NUM_ROOMS,
        NUM_WALL_CORNERS,
        WIDTH,
        drawResultImageFinal,
    )
    from utils import drawSegmentationImage, segmentation2Heatmaps, sigmoid, softmax

    os.makedirs(out_dir, exist_ok=True)
    reconstruct_floorplan = solver_module.reconstructFloorplan

    pred_wc = sigmoid(pred["corner"][0, :, :, :NUM_WALL_CORNERS])
    pred_oc = sigmoid(pred["corner"][0, :, :, NUM_WALL_CORNERS : NUM_WALL_CORNERS + 4])
    pred_ic = sigmoid(
        pred["corner"][0, :, :, NUM_WALL_CORNERS + 4 : NUM_WALL_CORNERS + 8]
    )
    pred_icon = softmax(pred["icon"][0])
    pred_room = softmax(pred["room"][0])

    density = np.minimum(debug["x0_topdown"][0, :, :, -1] * 255, 255).astype(np.uint8)
    density_rgb = np.stack([density, density, density], axis=2)
    cv2.imwrite(os.path.join(out_dir, scene_name + "_density.png"), density_rgb)

    threshold = np.ones((HEIGHT, WIDTH, 1), dtype=np.float32) * 0.5
    corner_img = drawSegmentationImage(np.concatenate([threshold, pred_wc], axis=2), blackIndex=0)
    corner_img[corner_img == 0] = density_rgb[corner_img == 0]
    cv2.imwrite(os.path.join(out_dir, scene_name + "_corner_pred.png"), corner_img)

    icon_img = drawSegmentationImage(np.argmax(pred_icon, axis=-1), blackIndex=0)
    icon_img[icon_img == 0] = density_rgb[icon_img == 0]
    cv2.imwrite(os.path.join(out_dir, scene_name + "_icon_pred.png"), icon_img)

    room_img = drawSegmentationImage(np.argmax(pred_room, axis=-1), blackIndex=0)
    room_img[room_img == 0] = density_rgb[room_img == 0]
    cv2.imwrite(os.path.join(out_dir, scene_name + "_room_pred.png"), room_img)

    np.savez_compressed(
        os.path.join(out_dir, scene_name + "_network_pred.npz"),
        corner=pred["corner"][0],
        icon=pred["icon"][0],
        room=pred["room"][0],
        density=debug["x0_topdown"][0],
    )

    print("Running floorplan reconstruction (%s)..." % solver_module.__name__)
    result_pred = reconstruct_floorplan(
        pred_wc,
        pred_oc,
        pred_ic,
        pred_icon,
        pred_room,
        density,
        gt=False,
        debug_prefix=os.path.join(out_dir, scene_name + "_reconstruct_debug"),
    )
    if result_pred:
        new_size = 1000
        from floorplan_utils import resizeResult

        resizeResult(result_pred, new_size, new_size, WIDTH, HEIGHT)
        result_image = drawResultImageFinal(new_size, new_size, result_pred)
        cv2.imwrite(os.path.join(out_dir, scene_name + "_floorplan_pred.png"), result_image)
        print("Wrote floorplan:", os.path.join(out_dir, scene_name + "_floorplan_pred.png"))
    else:
        print("Warning: reconstruction returned empty result for", scene_name)


def run_inference(tfrecord_path, out_dir, scene_name, checkpoint_dir, gpu="0", solver="ip"):
    os.chdir(REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from middleware.tf_setup import configure

    configure()
    import tensorflow.compat.v1 as tf
    import tflearn  # noqa: F401 — shim installed by configure()

    from RecordReader import getDatasetVal
    from train import build_graph, build_loss, parse_args

    if solver == "ip":
        import IP as solver_module
    elif solver == "qp":
        import QP as solver_module
    else:
        raise ValueError("solver must be ip or qp, got %s" % solver)

    argv_bak = sys.argv
    sys.argv = [argv_bak[0]]
    options = parse_args()
    sys.argv = argv_bak

    options.checkpoint_dir = os.path.abspath(checkpoint_dir)
    options.batchSize = 1
    options.branches = "0123"
    options.hybrid = "1"
    options.gpu_id = [g.strip() for g in str(gpu).split(",") if g.strip()]
    options.numTestingImages = 1
    options.augmentation = ""

    tfrecord_path = os.path.abspath(tfrecord_path)
    out_dir = os.path.abspath(out_dir)

    tf.reset_default_graph()
    dataset = getDatasetVal([tfrecord_path], "", False, options.batchSize)
    iterator = dataset.make_one_shot_iterator()
    input_dict, gt_dict = iterator.get_next()
    pred_dict, debug_dict = build_graph(options, input_dict)
    dataset_flag = input_dict["flags"][0, 0]
    loss, loss_list = build_loss(
        options, pred_dict, gt_dict, dataset_flag, debug_dict, input_dict["flags"]
    )

    var_to_restore = [v for v in tf.global_variables() if "is_training" not in v.name]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tflearn.is_training(False)
        loader = tf.train.Saver(var_to_restore)
        ckpt = os.path.join(options.checkpoint_dir, "checkpoint.ckpt")
        loader.restore(sess, ckpt)
        pred, debug, inp = sess.run([pred_dict, debug_dict, input_dict])

    _save_predictions(out_dir, scene_name, pred, debug, inp, solver_module)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="FloorNet single-TFRecord inference worker")
    parser.add_argument("--tfrecord", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--solver", choices=("ip", "qp"), default="ip")
    args = parser.parse_args()

    try:
        run_inference(
            args.tfrecord,
            args.out_dir,
            args.scene_name,
            args.checkpoint_dir,
            gpu=args.gpu,
            solver=args.solver,
        )
    except ImportError as e:
        print(
            "Inference requires tensorflow>=2.15, tf-slim, opencv, pulp (see requirements.txt).",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
