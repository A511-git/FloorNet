"""Load multi-view depth bundles (e.g. VGGT-style depth.npz) and fuse into one point cloud."""

import numpy as np


def _as_scalar_int(x):
    if isinstance(x, np.ndarray):
        return int(np.asarray(x).reshape(-1)[0])
    return int(x)


def load_depth_npz(path):
    """
    Load depth.npz and return a dict of arrays.

    Expected keys (all optional except one of world_points_from_depth / depth):
      world_points_from_depth (V, H, W, 3)
      images (V, 3, H, W) or (V, H, W, 3)
      depth (V, H, W, 1) or (V, H, W)
      depth_conf (V, H, W)
      extrinsic, pose_enc, pose_enc_list, covisibility_scores, ref_idx
    """
    data = dict(np.load(path, allow_pickle=False))
    return data


def _normalize_images(images):
    """Return (V, H, W, 3) float32 RGB in [0, 1]."""
    img = np.asarray(images, dtype=np.float32)
    if img.ndim != 4:
        raise ValueError("images must be 4D (views, C, H, W) or (views, H, W, C), got %s" % (img.shape,))
    if img.shape[1] == 3:
        img = np.transpose(img, (0, 2, 3, 1))
    elif img.shape[-1] != 3:
        raise ValueError("images channel dim must be 3, got shape %s" % (img.shape,))
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def fuse_multiview_point_cloud(
    data,
    conf_min=0.0,
    use_world_points=True,
    view_indices=None,
    stride=1,
):
    """
    Fuse per-pixel 3D points from all views into Nx6 XYZRGB (RGB in [0, 1]).

    Parameters
    ----------
    data : dict from load_depth_npz
    conf_min : minimum depth_conf to keep a point (0 = keep all valid)
    use_world_points : if True, use world_points_from_depth; else unproject depth (not implemented)
    view_indices : optional list of view indices to use (default: all)
    stride : subsample pixels by this step along H and W (1 = dense)
    """
    if use_world_points:
        if "world_points_from_depth" not in data:
            raise KeyError("NPZ missing world_points_from_depth; cannot build point cloud.")
        world = np.asarray(data["world_points_from_depth"], dtype=np.float32)
    else:
        raise NotImplementedError("Depth-only unprojection is not implemented; use world_points_from_depth.")

    if world.ndim != 4 or world.shape[-1] != 3:
        raise ValueError("world_points_from_depth must be (V, H, W, 3), got %s" % (world.shape,))

    num_views = world.shape[0]
    if view_indices is None:
        view_indices = list(range(num_views))
    else:
        view_indices = [int(v) for v in view_indices]

    conf = None
    if "depth_conf" in data:
        conf = np.asarray(data["depth_conf"], dtype=np.float32)
        if conf.ndim == 4:
            conf = conf[..., 0]

    images = None
    if "images" in data:
        images = _normalize_images(data["images"])

    depth = None
    if "depth" in data:
        depth = np.asarray(data["depth"], dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]

    chunks_xyz = []
    chunks_rgb = []

    for v in view_indices:
        wv = world[v]
        if stride > 1:
            wv = wv[::stride, ::stride]

        flat = wv.reshape(-1, 3)
        valid = np.isfinite(flat).all(axis=1)
        if depth is not None:
            dv = depth[v]
            if stride > 1:
                dv = dv[::stride, ::stride]
            valid &= np.isfinite(dv.reshape(-1)) & (dv.reshape(-1) > 1e-6)
        if conf is not None:
            cv = conf[v]
            if stride > 1:
                cv = cv[::stride, ::stride]
            valid &= np.isfinite(cv.reshape(-1)) & (cv.reshape(-1) >= conf_min)

        xyz = flat[valid]
        if xyz.shape[0] == 0:
            continue

        if images is not None:
            iv = images[v]
            if stride > 1:
                iv = iv[::stride, ::stride]
            rgb = iv.reshape(-1, 3)[valid]
        else:
            rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)

        chunks_xyz.append(xyz)
        chunks_rgb.append(rgb)

    if not chunks_xyz:
        raise ValueError("No valid points after filtering; try lowering --conf-min or check inputs.")

    xyz = np.concatenate(chunks_xyz, axis=0)
    rgb = np.concatenate(chunks_rgb, axis=0)
    points = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    meta = {
        "num_views": num_views,
        "views_used": view_indices,
        "num_points_raw": points.shape[0],
        "ref_idx": data.get("ref_idx"),
    }
    if meta["ref_idx"] is not None:
        meta["ref_idx"] = _as_scalar_int(meta["ref_idx"])
    return points, meta
