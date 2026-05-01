# Skigen — EES (Energy-Efficient Software) Rules

## Context

This file is machine-readable project context for AI agents working on Skigen.

## Identity

- **Project:** Skigen
- **Language:** C++23, header-only
- **Dependency:** Eigen (sole numerical backend)
- **Goal:** scikit-learn API parity with zero interpreter tax

## Mandatory Rules

1. **Header-only.** No `.cpp` files under `Skigen/`. Every component is a `.h` header.
2. **Eigen-style layout.** Module headers at `Skigen/` root without extension (e.g. `Core`, `Preprocessing`). Internal headers in `Skigen/src/Module/Name.h`.
3. **Templatized.** All classes accept `typename Scalar = double`. No hardcoded numeric types.
4. **Zero-copy inputs.** Use `Eigen::Ref<const MatrixType>` for read-only parameters.
5. **No raw loops.** Use Eigen expression templates for all numerical operations.
6. **CRTP only.** Static polymorphism via Curiously Recurring Template Pattern. No `virtual`.
7. **Minimize allocations.** Favor in-place operations. Provide `_inplace` variants.
8. **Bit-level parity.** Output must match scikit-learn for identical inputs and defaults.
9. **Warning-free.** Must compile cleanly with `-Wall -Wextra -Wpedantic`.

## Type Conventions

```cpp
using ScalarType    = Scalar;
using MatrixType    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using VectorType    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using RowVectorType = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
using IndexType     = Eigen::Index;
```

## File Placement

| Component | Path |
|-----------|------|
| Module headers | `Skigen/Core`, `Skigen/Preprocessing`, `Skigen/Dense` |
| CRTP bases | `Skigen/src/Core/Base.h` |
| Concepts | `Skigen/src/Core/Concepts.h` |
| Type traits | `Skigen/src/Core/Traits.h` |
| Validation | `Skigen/src/Core/Validation.h` |
| Eigen helpers | `Skigen/src/Core/EigenHelpers.h` |
| Transformers | `Skigen/src/Preprocessing/` |
| Predictors | `Skigen/src/LinearModel/` |
| Metrics | `Skigen/src/Metrics/` |
| Tests | `tests/` |
| Benchmarks | `benchmarks/` |

## Include Pattern

Users include skigen the same way they include Eigen:
```cpp
#include <Skigen/Core>           // Base classes, traits, concepts
#include <Skigen/Preprocessing>  // StandardScaler, etc.
#include <Skigen/Dense>          // Everything bundled
```

## Namespace

All code lives under `namespace Skigen`. No anonymous namespaces in headers.
