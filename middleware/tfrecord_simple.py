"""Minimal TFRecord Example writer (no TensorFlow required)."""

import struct


def _encode_varint(value):
    value = int(value)
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        out.append(bits | (0x80 if value else 0))
        if not value:
            break
    return bytes(out)


def _encode_key(field_number, wire_type):
    return _encode_varint((field_number << 3) | wire_type)


def _encode_length_delimited(payload):
    return _encode_key(2, 2) + _encode_varint(len(payload)) + payload


def _encode_float_packed(values):
    payload = struct.pack("<%df" % len(values), *values)
    return _encode_key(1, 2) + _encode_varint(len(payload)) + payload


def _encode_int64_packed(values):
    # int64_list uses non-packed repeated varint in TF Example convention
    parts = []
    for v in values:
        parts.append(_encode_key(1, 0) + _encode_varint(int(v)))
    return b"".join(parts)


def _encode_bytes_list(values):
    parts = []
    for v in values:
        if isinstance(v, str):
            v = v.encode("utf-8")
        parts.append(_encode_key(1, 2) + _encode_varint(len(v)) + v)
    return b"".join(parts)


def _encode_feature_bytes(values):
    inner = _encode_bytes_list(values)
    return _encode_key(1, 2) + _encode_varint(len(inner)) + inner


def _encode_feature_float(values):
    inner = _encode_float_packed(values)
    return _encode_key(2, 2) + _encode_varint(len(inner)) + inner


def _encode_feature_int64(values):
    inner = _encode_int64_packed(values)
    return _encode_key(3, 2) + _encode_varint(len(inner)) + inner


def _encode_map_entry(name, feature_bytes):
    # MapEntry { string key = 1; Feature value = 2; }
    key_b = _encode_key(1, 2) + _encode_varint(len(name)) + name
    val_b = _encode_key(2, 2) + _encode_varint(len(feature_bytes)) + feature_bytes
    entry = key_b + val_b
    return _encode_key(1, 2) + _encode_varint(len(entry)) + entry


def serialize_example(feature_dict):
    """
    feature_dict: name -> ("bytes"|"float"|"int64", sequence of values)
    """
    entries = []
    for name, (kind, values) in feature_dict.items():
        if not isinstance(name, bytes):
            name = name.encode("utf-8")
        if kind == "bytes":
            feat = _encode_feature_bytes(values)
        elif kind == "float":
            feat = _encode_feature_float(values)
        elif kind == "int64":
            feat = _encode_feature_int64(values)
        else:
            raise ValueError("unknown feature kind %s" % kind)
        entries.append(_encode_map_entry(name, feat))

    features_msg = b"".join(entries)
    # Example.features field 1
    example = _encode_key(1, 2) + _encode_varint(len(features_msg)) + features_msg
    return example


def write_tfrecord_file(path, record_bytes):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(record_bytes)))
        f.write(record_bytes)
