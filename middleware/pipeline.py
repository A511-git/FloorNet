"""Convert one depth.npz scan into a per-scan output directory."""

import os

from .depth_npz import fuse_multiview_point_cloud, load_depth_npz
from .floornet_pack import pack_for_floornet, write_floornet_npz, write_ply, write_tfrecord


def convert_npz_scan(
    npz_path,
    out_scan_dir,
    scene_name=None,
    conf_min=0.0,
    stride=1,
    view_indices=None,
    num_points=50000,
    seed=0,
    align=True,
    dataset_flag=1,
):
    """
    Write into out_scan_dir:
      <scene>.ply, <scene>_floornet.npz, <scene>.tfrecords
    """
    npz_path = os.path.abspath(npz_path)
    out_scan_dir = os.path.abspath(out_scan_dir)
    os.makedirs(out_scan_dir, exist_ok=True)

    if scene_name is None:
        scene_name = os.path.splitext(os.path.basename(npz_path))[0]

    data = load_depth_npz(npz_path)
    raw_points, meta = fuse_multiview_point_cloud(
        data,
        conf_min=conf_min,
        view_indices=view_indices,
        stride=max(1, stride),
    )
    packed = pack_for_floornet(
        raw_points,
        num_points=num_points,
        seed=seed,
        align=align,
    )

    ply_path = os.path.join(out_scan_dir, scene_name + ".ply")
    npz_bundle = os.path.join(out_scan_dir, scene_name + "_floornet.npz")
    tfrecord_path = os.path.join(out_scan_dir, scene_name + ".tfrecords")

    write_ply(ply_path, raw_points, rgb_in_01=True)
    write_floornet_npz(npz_bundle, scene_name, packed, raw_points=raw_points, meta=meta)
    write_tfrecord(tfrecord_path, scene_name, packed, dataset_flag=dataset_flag)

    return {
        "scene_name": scene_name,
        "ply": ply_path,
        "floornet_npz": npz_bundle,
        "tfrecord": tfrecord_path,
        "meta": meta,
    }
