#!/usr/bin/env python3
"""
Fashion-MNIST loader with no torchvision dependency.

Downloads the four idx.gz files from the first mirror that answers, caches them
under ./_fashion_mnist/, and returns integer tensors padded to the coupling
half-width M so the network stays fully reversible: the padded input goes
straight into v0 = [x_pad, 0] with no projection layer to store.
"""

import gzip
import hashlib
import os
import struct
import urllib.request

import torch

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fashion_mnist")

FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}

MIRRORS = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
]

# Official Zalando checksums for the four gz files.
SHA256 = {
    "train-images-idx3-ubyte.gz":
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz":
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz":
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz":
        "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}


def _fetch(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    last = None
    for base in MIRRORS:
        try:
            req = urllib.request.Request(base + name,
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            if len(blob) < 1000:
                raise IOError(f"short read {len(blob)} bytes")
            with open(path, "wb") as fh:
                fh.write(blob)
            return path
        except Exception as exc:  # try the next mirror
            last = f"{base}: {type(exc).__name__}: {exc}"
    raise RuntimeError(f"could not fetch {name}; last error {last}")


def _verify(path):
    """Returns (matches_official, actual_digest). Never raises on mismatch."""
    name = os.path.basename(path)
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    actual = d.hexdigest()
    return actual == SHA256.get(name), actual


def _read_idx(path):
    with gzip.open(path, "rb") as fh:
        magic, count = struct.unpack(">II", fh.read(8))
        if magic == 2051:  # images
            rows, cols = struct.unpack(">II", fh.read(8))
            buf = fh.read(count * rows * cols)
            t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
            return t.reshape(count, rows * cols).to(torch.int64)
        if magic == 2049:  # labels
            buf = fh.read(count)
            return torch.frombuffer(bytearray(buf), dtype=torch.uint8).to(torch.int64)
        raise ValueError(f"unexpected idx magic {magic} in {path}")


def load(M=832, input_shift=0, train_limit=None, test_limit=None, verify=True):
    """
    Returns (xtr, ytr, xte, yte, info).

    M            pad the 784 pixels out to this width with zeros; the padding
                 is part of the reversible state and needs no projection.
    input_shift  right-shift raw 0..255 pixels by this many bits before use.
    """
    paths = {k: _fetch(v) for k, v in FILES.items()}
    checks = {}
    if verify:
        for k, p in paths.items():
            match, digest = _verify(p)
            checks[os.path.basename(p)] = {"official_sha256_match": match,
                                           "sha256": digest}

    xtr = _read_idx(paths["train_x"])
    ytr = _read_idx(paths["train_y"])
    xte = _read_idx(paths["test_x"])
    yte = _read_idx(paths["test_y"])

    if train_limit:
        xtr, ytr = xtr[:train_limit], ytr[:train_limit]
    if test_limit:
        xte, yte = xte[:test_limit], yte[:test_limit]

    if input_shift:
        xtr = xtr >> input_shift
        xte = xte >> input_shift

    pixels = xtr.shape[1]
    if M < pixels:
        raise ValueError(f"M={M} is smaller than the {pixels} input pixels")
    pad = M - pixels
    if pad:
        xtr = torch.cat([xtr, torch.zeros((xtr.shape[0], pad), dtype=torch.int64)], 1)
        xte = torch.cat([xte, torch.zeros((xte.shape[0], pad), dtype=torch.int64)], 1)

    info = {
        "pixels": pixels,
        "M": M,
        "pad": pad,
        "input_shift": input_shift,
        "train": tuple(xtr.shape),
        "test": tuple(xte.shape),
        "max_pixel": int(xtr.max().item()),
        "classes": int(ytr.max().item()) + 1,
        "file_checks": checks,
    }
    return xtr, ytr, xte, yte, info


if __name__ == "__main__":
    import json
    _xtr, _ytr, _xte, _yte, _info = load()
    print(json.dumps(_info, indent=2))
