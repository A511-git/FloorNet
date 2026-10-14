"""Download and verify FloorNet checkpoints."""

import os
import sys
import urllib.request

DEFAULT_CHECKPOINT_NAME = "floornet_hybrid1_branch0123_wsf"
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/art-programmer/FloorNet/master/checkpoint/"
)

CHECKPOINT_FILES = [
    "checkpoint",
    "checkpoint.ckpt.index",
    "checkpoint.ckpt.meta",
    "checkpoint.ckpt.data-00000-of-00001",
]

# Expected size of the weights shard (bytes) from upstream repo.
MIN_DATA_BYTES = 20_000_000


def checkpoint_dir(repo_root=None, name=DEFAULT_CHECKPOINT_NAME):
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "checkpoint", name)


def is_checkpoint_ready(checkpoint_path):
    if not os.path.isdir(checkpoint_path):
        return False
    index_path = os.path.join(checkpoint_path, "checkpoint.ckpt.index")
    data_path = os.path.join(checkpoint_path, "checkpoint.ckpt.data-00000-of-00001")
    if not os.path.isfile(index_path):
        return False
    if not os.path.isfile(data_path):
        return False
    if os.path.getsize(data_path) < MIN_DATA_BYTES:
        return False
    return True


def _download_file(url, dest_path, label=None):
    label = label or os.path.basename(dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    def reporthook(block_count, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_count * block_size, total_size)
        pct = 100.0 * done / total_size
        sys.stdout.write("\r  %s %.1f%%" % (label, pct))
        sys.stdout.flush()

    print("Downloading %s ..." % label)
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    sys.stdout.write("\n")
    sys.stdout.flush()


def ensure_checkpoint(repo_root=None, name=DEFAULT_CHECKPOINT_NAME, force=False):
    """
    Ensure checkpoint files exist under checkpoint/<name>/.
    Downloads missing files from the official GitHub repo if needed.
    """
    path = checkpoint_dir(repo_root, name)
    if not force and is_checkpoint_ready(path):
        return path

    os.makedirs(path, exist_ok=True)
    base_url = GITHUB_RAW_BASE + name + "/"
    for filename in CHECKPOINT_FILES:
        dest = os.path.join(path, filename)
        if not force and os.path.isfile(dest):
            if filename.startswith("checkpoint.ckpt.data") and os.path.getsize(dest) < MIN_DATA_BYTES:
                pass
            else:
                continue
        url = base_url + filename
        _download_file(url, dest, filename)

    if not is_checkpoint_ready(path):
        raise RuntimeError("Checkpoint download incomplete: %s" % path)
    print("Checkpoint ready:", path)
    return path
