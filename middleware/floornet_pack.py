"""Align, normalize, and package point clouds for FloorNet (TFRecord / NPZ)."""

import os

import numpy as np

HEIGHT = 256
WIDTH = 256
NUM_POINTS = 50000
NUM_INPUT_CHANNELS = 6  # stored in TFRecord (7th channel is ones in RecordReader)
MAX_NUM_CORNERS = 300

# Must match floorplan_utils.py SIZES / NUM_CHANNELS for dummy image features
_SIZES = [WIDTH, WIDTH // 2, WIDTH // 4, WIDTH // 8, WIDTH // 16, WIDTH // 32]
_NUM_CHANNELS = [7, 64, 64, 64, 128, 256]
_DUMMY_IMAGE_FEATURE_SIZE = sum(
    size * size * num_channels
    for size, num_channels in list(zip(_SIZES, _NUM_CHANNELS))[1:]
)


def align_points_pca(xyz):
    """
    Rotate so the smallest-variance axis is vertical (Z); plan uses X,Y.
    Returns aligned_xyz, rotation_matrix (3x3), centroid (3,).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    if centered.shape[0] < 3:
        return xyz.astype(np.float32), np.eye(3, dtype=np.float32), centroid.astype(np.float32)

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    vertical = eigvecs[:, order[0]]
    h1 = eigvecs[:, order[2]]
    h2 = eigvecs[:, order[1]]
    r = np.stack([h1, h2, vertical], axis=1)
    if np.linalg.det(r) < 0:
        r[:, 0] *= -1
    if r[2, 2] < 0:
        r[:, 2] *= -1
    aligned = centered @ r
    return aligned.astype(np.float32), r.astype(np.float32), centroid.astype(np.float32)


def scale_points_floornet(points):
    """
    Match RecordWriterTango normalization: square footprint in XY, values in [0, 1].
    points: NxC, modifies XYZ in place conceptually; returns scaled copy.
    """
    out = np.array(points, dtype=np.float32, copy=True)
    xyz = out[:, :3]
    mins = xyz.min(0, keepdims=True)
    maxs = xyz.max(0, keepdims=True)
    max_range = float((maxs - mins)[:, :2].max())
    if max_range < 1e-8:
        max_range = 1.0
    padding = max_range * 0.05
    mins = (maxs + mins) / 2.0 - max_range / 2.0
    mins -= padding
    max_range += padding * 2.0
    out[:, :3] = (xyz - mins) / max_range
    return out


def compute_point_indices(points_xy):
    """FloorNet 256x256 grid indices: row * WIDTH + col from normalized x,y in [0,1]."""
    coords = np.clip(np.round(points_xy[:, :2] * HEIGHT).astype(np.int32), 0, HEIGHT - 1)
    return (coords[:, 1] * WIDTH + coords[:, 0]).astype(np.int64)


def sample_points(points, num_points=NUM_POINTS, seed=0):
    n = points.shape[0]
    rng = np.random.RandomState(seed)
    if n < num_points:
        extra = rng.choice(n, num_points - n, replace=True)
        return np.concatenate([points, points[extra]], axis=0)
    if n > num_points:
        idx = rng.choice(n, num_points, replace=False)
        return points[idx]
    return points


def pack_for_floornet(points, num_points=NUM_POINTS, seed=0, align=True):
    """
    points: Nx6 float32, XYZ world + RGB in [0, 1].
    Returns dict with floornet arrays and metadata.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[1] < 6:
        raise ValueError("points need at least 6 columns (XYZRGB), got %s" % (points.shape,))

    rotation = np.eye(3, dtype=np.float32)
    centroid = points[:, :3].mean(axis=0).astype(np.float32)
    if align:
        aligned, rotation, centroid = align_points_pca(points[:, :3])
        points = points.copy()
        points[:, :3] = aligned

    points = sample_points(points, num_points=num_points, seed=seed)
    points = scale_points_floornet(points)

    # Match RecordWriterTango color scaling (expects ~0-255 before this step)
    rgb = points[:, 3:6]
    if rgb.max() <= 1.0:
        points[:, 3:6] = rgb - 0.5
    else:
        points[:, 3:6] = rgb / 255.0 - 0.5

    store = points[:, :NUM_INPUT_CHANNELS].astype(np.float32)
    indices = compute_point_indices(store)

    return {
        "points": store,
        "point_indices": indices,
        "rotation": rotation,
        "centroid": centroid,
        "num_points": num_points,
    }


def write_ply(path, points, rgb_in_01=True):
    """Write ASCII PLY from Nx6 (or Nx3) array."""
    points = np.asarray(points)
    xyz = points[:, :3]
    if points.shape[1] >= 6:
        rgb = points[:, 3:6]
        if rgb_in_01:
            if rgb.max() <= 1.0 and rgb.min() >= -0.6:
                # may already be floornet-normalized
                if rgb.min() < 0:
                    rgb = np.clip(rgb + 0.5, 0, 1)
                rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            else:
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % xyz.shape[0])
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(xyz.shape[0]):
            f.write(
                "%.6f %.6f %.6f %d %d %d\n"
                % (xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb_u8[i, 0], rgb_u8[i, 1], rgb_u8[i, 2])
            )


def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return _make_feature(bytes_list=[value])


def _int64_feature(value):
    return _make_feature(int64_list=[int(v) for v in np.asarray(value).reshape(-1)])


def _float_feature(value):
    return _make_feature(float_list=[float(v) for v in np.asarray(value).reshape(-1)])


def _make_feature(**kwargs):
    try:
        import tensorflow as tf

        if "bytes_list" in kwargs:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=kwargs["bytes_list"]))
        if "int64_list" in kwargs:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=kwargs["int64_list"]))
        if "float_list" in kwargs:
            return tf.train.Feature(float_list=tf.train.FloatList(value=kwargs["float_list"]))
    except ImportError:
        pass
    return _SimpleFeature(**kwargs)


class _SimpleFeature(object):
    """Minimal TFRecord feature without TensorFlow."""

    def __init__(self, bytes_list=None, int64_list=None, float_list=None):
        self.bytes_list = bytes_list
        self.int64_list = int64_list
        self.float_list = float_list


def _features_to_simple_dict(features):
    out = {}
    for name, feat in features.items():
        if feat.bytes_list is not None:
            out[name] = ("bytes", feat.bytes_list)
        elif feat.float_list is not None:
            out[name] = ("float", feat.float_list)
        elif feat.int64_list is not None:
            out[name] = ("int64", feat.int64_list)
        else:
            raise ValueError("empty feature for %s" % name)
    return out


def _serialize_example_simple(features):
    """Serialize Example using TensorFlow if present, else pure-Python protobuf."""
    try:
        import tensorflow as tf

        tf_features = {}
        for k, v in features.items():
            if isinstance(v, tf.train.Feature):
                tf_features[k] = v
            elif isinstance(v, _SimpleFeature):
                if v.bytes_list is not None:
                    tf_features[k] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=v.bytes_list)
                    )
                elif v.int64_list is not None:
                    tf_features[k] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=v.int64_list)
                    )
                elif v.float_list is not None:
                    tf_features[k] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=v.float_list)
                    )
            else:
                tf_features[k] = v
        example = tf.train.Example(features=tf.train.Features(feature=tf_features))
        return example.SerializeToString()
    except ImportError:
        from .tfrecord_simple import serialize_example

        return serialize_example(_features_to_simple_dict(features))


def write_tfrecord(path, scene_id, packed, dataset_flag=1):
    """
    Write one FloorNet-compatible TFRecord example (dummy labels for inference).
    """
    points = packed["points"]
    indices = packed["point_indices"]
    icon = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    room = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    corner_pad = np.zeros((MAX_NUM_CORNERS, 3), dtype=np.int64)
    flags = np.array([dataset_flag, 0], dtype=np.int64)
    image_feat = np.zeros(_DUMMY_IMAGE_FEATURE_SIZE, dtype=np.float32)

    features = {
        "image_path": _bytes_feature(scene_id),
        "points": _float_feature(points.reshape(-1)),
        "point_indices": _int64_feature(indices.reshape(-1)),
        "corner": _int64_feature(corner_pad.reshape(-1)),
        "num_corners": _int64_feature([0]),
        "icon": _bytes_feature(icon.tobytes()),
        "room": _bytes_feature(room.tobytes()),
        "image": _float_feature(image_feat.reshape(-1)),
        "flags": _int64_feature(flags),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    record = _serialize_example_simple(features)
    from .tfrecord_simple import write_tfrecord_file

    write_tfrecord_file(path, record)


def write_floornet_npz(path, scene_id, packed, raw_points=None, meta=None):
    """Sidecar bundle for debugging / custom loaders."""
    payload = {
        "scene_id": scene_id,
        "points": packed["points"],
        "point_indices": packed["point_indices"],
        "rotation": packed["rotation"],
        "centroid": packed["centroid"],
    }
    if raw_points is not None:
        payload["points_raw_xyzrgb"] = raw_points
    if meta is not None:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **payload)
