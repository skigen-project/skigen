"""Shared helpers for the Skigen parity reference generators.

Every ``generate_<estimator>_reference.py`` script imports this module to:

* enforce the pinned scikit-learn minor version (``sklearn_version.txt``);
* obtain small, deterministic fixtures with fixed seeds;
* write expected arrays as plain CSV under ``data/<estimator>/``.

The CSV files are checked into the repository so that CI never has to invoke
Python at test time — only the explicit ``skigen_regenerate_parity`` target
runs these scripts.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")


def check_sklearn_version():
    """Abort regeneration if the installed sklearn minor differs from the pin."""
    import sklearn

    with open(os.path.join(HERE, "sklearn_version.txt")) as fh:
        pinned = fh.read().strip()
    installed = sklearn.__version__
    pinned_mm = ".".join(pinned.split(".")[:2])
    installed_mm = ".".join(installed.split(".")[:2])
    if pinned_mm != installed_mm:
        raise SystemExit(
            f"sklearn version mismatch: pinned {pinned_mm}.x, "
            f"installed {installed}. Update tests/parity/sklearn_version.txt "
            f"and review default changes before regenerating."
        )
    return installed


def out_dir(name):
    d = os.path.join(DATA_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def save(name, **arrays):
    """Write each named array as ``data/<name>/<key>.csv`` (2-D, %.18e)."""
    d = out_dir(name)
    for key, value in arrays.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        np.savetxt(os.path.join(d, key + ".csv"), arr, delimiter=",", fmt="%.18e")
    return d


def save_params(name, params):
    """Record the constructor parameters used, for documentation/auditing."""
    d = out_dir(name)
    with open(os.path.join(d, "params.json"), "w") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)


def blobs(n=60, d=3, centers=2, seed=0):
    """Two/three Gaussian blobs — deterministic, well separated."""
    rng = np.random.default_rng(seed)
    per = n // centers
    offsets = np.linspace(-4.0, 4.0, centers)
    X = np.vstack(
        [rng.normal(loc=off, scale=0.6, size=(per, d)) for off in offsets]
    )
    y = np.concatenate([np.full(per, k) for k in range(centers)])
    return X.astype(np.float64), y.astype(np.int64)


def regression_data(n=60, d=4, seed=0):
    """Linear regression fixture with mild noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = np.array([1.5, -2.0, 0.5, 3.0][:d])
    y = X @ true_w + 0.1 * rng.normal(size=n)
    return X.astype(np.float64), y.astype(np.float64)


def counts_data(n=40, d=5, seed=0):
    """Non-negative integer count matrix for Multinomial/chi2 fixtures."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 6, size=(n, d))
    y = (X[:, 0] + X[:, 1] > X[:, 2] + X[:, 3]).astype(np.int64)
    return X.astype(np.float64), y
