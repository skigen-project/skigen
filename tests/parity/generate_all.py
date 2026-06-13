"""Driver that regenerates every Skigen parity fixture.

Invoked by the ``skigen_regenerate_parity`` CMake target. Imports each
``generate_<estimator>_reference.py`` module and runs its ``main()``. The
scikit-learn minor version is checked once up-front against
``sklearn_version.txt`` so regeneration fails loudly on a version drift.
"""

import importlib
import os
import sys

from _common import check_sklearn_version

GENERATORS = [
    "generate_core_reference",
    "generate_naive_bayes_reference",
    "generate_isotonic_reference",
    "generate_covariance_reference",
    "generate_feature_selection_reference",
    "generate_bayesian_linear_reference",
    "generate_decomposition_reference",
    "generate_neighbors_reference",
    "generate_ensemble_reference",
    "generate_svm_reference",
    "generate_neural_network_reference",
    "generate_manifold_reference",
    "generate_calibration_reference",
    "generate_model_selection_reference",
]


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    version = check_sklearn_version()
    print(f"Regenerating parity fixtures against scikit-learn {version}")
    for name in GENERATORS:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError:
            print(f"  SKIP   {name} (not present)")
            continue
        mod.main()
        print(f"  OK     {name}")
    print("Done.")


if __name__ == "__main__":
    main()
