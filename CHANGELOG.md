# Changelog

All notable changes to Skigen will be documented in this file.

## [1.0.0] - 2026-05-02

### Highlights

- **First stable release** — Skigen v1.0.0 delivers a complete, header-only C++ machine learning library built on Eigen, mirroring the scikit-learn API with zero interpreter tax.
- **Feature → Staging → Main branching** — mne-cpp-style branching model with separate CI pipelines for pull requests, staging, and releases.
- **CMake install support** — `cmake --install` deploys a clean, includeable header tree with `find_package(Skigen)` support (Eigen-style).

### Modules

- **Core** — CRTP base classes (`Estimator`, `Transformer`, `Predictor`), C++23 concepts (`TransformerLike`, `PredictorLike`), type traits, input validation, Eigen helpers
- **Preprocessing** — `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `LabelEncoder`, `PolynomialFeatures`
- **LinearModel** — `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`, `SGDClassifier`, `SGDRegressor`
- **Decomposition** — `PCA`, `TruncatedSVD`
- **Cluster** — `KMeans`, `MiniBatchKMeans`
- **Neighbors** — `KNeighborsClassifier`
- **Tree** — `DecisionTreeClassifier`
- **Metrics** — `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`
- **ModelSelection** — `train_test_split`, `cross_val_score`
- **Pipeline** — `make_pipeline` with compile-time variadic step composition

### Examples

- 21 per-module examples organized by submodule (`preprocessing/`, `linear_model/`, `decomposition/`, `cluster/`, `neighbors/`, `tree/`, `metrics/`, `model_selection/`, `pipeline/`)
- 3 integration workflow examples (`classification_workflow`, `regression_pipeline`, `pca_clustering_workflow`)

### Infrastructure

- CI pipelines: pull-request, staging, main/release (Linux GCC 13/14, macOS AppleClang, Windows MSVC)
- Release workflow: automatic GitHub Release with header tarball on `v*` tags
- Docusaurus documentation website with 21 API guide pages
- Python parity test infrastructure for cross-language verification
