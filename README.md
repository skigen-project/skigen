<p align="center">
  <img src="doc/website/static/img/skigen-logo.svg" alt="Skigen" width="400">
</p>

<p align="center">
  <em>Energy-efficient machine learning, native to C++ and Eigen.</em>
</p>

<p align="center">
  <a href="https://github.com/skigen-project/skigen/actions"><img src="https://github.com/skigen-project/skigen/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://skigen-project.github.io"><img src="https://img.shields.io/badge/docs-website-green.svg" alt="Docs"></a>
</p>

-----------------

## About

The rapid expansion of AI has put extraordinary pressure on global compute and memory resources. A meaningful share of this cost comes not from the algorithms, but from the software infrastructure executing them — interpreter overhead, garbage collection, runtime dispatch, and the GIL collectively consume energy and memory that contribute nothing to the actual computation.

Skigen is a header-only machine learning library built on [Eigen](https://eigen.tuxfamily.org/) that brings the [scikit-learn](https://scikit-learn.org/) API — `fit()`, `transform()`, `predict()` — to native C++. It compiles every operation down to vectorized machine code, targeting the latest C++ standards to make full use of modern language features: concepts, constexpr, and zero-cost abstractions.

scikit-learn established the definitive API for machine learning. Skigen adopts that API for environments where native execution matters: embedded systems, real-time pipelines, edge inference, and large-scale deployments where energy and memory are engineering constraints, not afterthoughts.

## Design

| Principle | Implementation |
|---|---|
| **Eigen-native** | Headers, namespaces, and include patterns mirror Eigen. `#include <Skigen/Dense>` feels like `#include <Eigen/Dense>`. |
| **Header-only** | Drop `Skigen/` next to `Eigen/` — no compiled libraries, no linker flags. |
| **Templatized** | All estimators accept `Scalar` (default: `double`). Switch to `float` for 2× SIMD throughput on the same hardware. |
| **Zero-copy** | Inputs use `Eigen::Ref<const MatrixType>` — supports sub-blocks and memory-mapped data without copying. |
| **CRTP** | Static polymorphism via the Curiously Recurring Template Pattern. Zero vtable overhead. |
| **Bit-level parity** | Results match scikit-learn for identical inputs and default parameters. Verified via cross-language parity tests. |

## Quick Start

```cpp
#include <Eigen/Dense>
#include <Skigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd X(4, 2);
    X << 1, 10,
         2, 20,
         3, 30,
         4, 40;

    Skigen::StandardScaler scaler;
    Eigen::MatrixXd Z = scaler.fit_transform(X);

    std::cout << "Standardized:\n" << Z << "\n";

    // Round-trip back to original scale
    Eigen::MatrixXd X_back = scaler.inverse_transform(Z);
    std::cout << "Recovered:\n" << X_back << "\n";

    return 0;
}
```

## Build & Test

```bash
git clone https://github.com/skigen-project/skigen.git
cd skigen

cmake -B build -DSKIGEN_BUILD_TESTS=ON
cmake --build build
./build/tests/skigen_tests
```

Eigen is detected automatically if installed system-wide, or from a sibling `../eigen` source tree.

## Include Pattern

Skigen follows the same convention as Eigen — module headers without file extensions:

```cpp
#include <Skigen/Core>           // Base classes, traits, concepts
#include <Skigen/Preprocessing>  // Scalers, normalizers, ...
#include <Skigen/Dense>          // Everything bundled
```

## Repository Layout

```
Skigen/                     # Header library (Eigen-style)
├── Core                    # Module header — base classes, traits, concepts
├── Preprocessing           # Module header — scalers, normalizers, ...
├── LinearModel             # Module header — regression, classification
├── Decomposition           # Module header — PCA, TruncatedSVD
├── Cluster                 # Module header — KMeans, MiniBatchKMeans
├── Neighbors               # Module header — KNN classifier/regressor
├── Tree                    # Module header — decision trees
├── ModelSelection          # Module header — train/test split, cross-validation
├── Pipeline                # Module header — compile-time pipeline composition
├── Metrics                 # Module header — regression, classification, pairwise
├── Dense                   # Convenience header — bundles all modules
└── src/                    # Internal headers (.h)
    ├── Core/               # Traits, Concepts, Base, Validation, EigenHelpers
    ├── Preprocessing/      # StandardScaler, MinMaxScaler, MaxAbsScaler, ...
    ├── LinearModel/        # LinearRegression, Ridge, Lasso, ElasticNet, ...
    ├── Decomposition/      # PCA, TruncatedSVD
    ├── Cluster/            # KMeans, MiniBatchKMeans
    ├── Neighbors/          # KNeighborsClassifier, KNeighborsRegressor
    ├── Tree/               # DecisionTreeClassifier, DecisionTreeRegressor
    ├── ModelSelection/     # TrainTestSplit, CrossValidation
    ├── Pipeline/           # Pipeline (compile-time)
    └── Metrics/            # Regression, Classification, Pairwise
tests/                      # Unit + parity tests
benchmarks/                 # Performance benchmarks
examples/                   # Usage examples
doc/                        # Requirements + Docusaurus website
```

## Requirements

| | Minimum |
|---|---|
| **C++ Standard** | Latest (currently C++23) |
| **Compilers** | GCC ≥ 13, Clang ≥ 17, MSVC ≥ 19.38 |
| **Eigen** | ≥ 3.4 |
| **CMake** | ≥ 3.20 (for building tests/examples) |

## License

Skigen is released under the [MIT License](LICENSE).

## Acknowledgments

Skigen builds on [Eigen](https://eigen.tuxfamily.org/) for numerical computation and adopts the API conventions established by [scikit-learn](https://scikit-learn.org/).
