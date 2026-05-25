# Changelog

All notable changes to Skigen will be documented in this file.

## [1.1.0] - Unreleased

### Highlights

- **32 new estimators** spanning Ensembles (RandomForest, GradientBoosting,
  HistGradientBoosting), full SVM (LinearSVC, LinearSVR, SVC, SVR; NuSVC,
  NuSVR, OneClassSVM as documented placeholders), Naive Bayes (Gaussian,
  Multinomial, Bernoulli), MLP (Classifier + Regressor), Feature Selection
  (VarianceThreshold, SelectKBest, SelectFromModel, RFE), Bayesian linear
  models (BayesianRidge, ARDRegression), Isotonic Regression, Manifold
  learning (Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding, t-SNE,
  UMAP), Calibration (CalibratedClassifierCV), and hyperparameter search
  (GridSearchCV, RandomizedSearchCV).
- **`partial_fit` infrastructure** — `IncrementalLike` C++23 concept; online
  Welford / Chan updates retrofitted onto `StandardScaler`, `MinMaxScaler`,
  `MaxAbsScaler`, `MiniBatchKMeans`, `SGDClassifier`, `SGDRegressor`.
- **`Eigen::SparseMatrix` support** — sparse fit/transform paths for
  `VarianceThreshold`, `Normalizer`, `TruncatedSVD`, `LinearRegression`,
  `Ridge`, `Lasso`, `ElasticNet`, `SelectKBest` (chi-squared), `KMeans`,
  `MiniBatchKMeans`, and the `DecisionTree*` family.
- **Strict sklearn parity** — parameter names, order, types, defaults, fitted
  attributes, and error semantics pinned to scikit-learn 1.7.x as a
  release-blocking contract.
- **One example per public estimator** — `examples/` now holds 57 programs
  organised by module.

### New modules

- **Ensemble** — `RandomForestClassifier`, `RandomForestRegressor`,
  `GradientBoostingClassifier`, `GradientBoostingRegressor`,
  `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`.
- **NaiveBayes** — `GaussianNB`, `MultinomialNB`, `BernoulliNB`
  (all support `partial_fit`).
- **SVM** — `LinearSVC`, `LinearSVR`, `SVC`, `SVR` (libsvm-style SMO);
  `NuSVC`, `NuSVR`, `OneClassSVM` ship as placeholders that throw from
  `fit` until the nu-SVM solver lands.
- **NeuralNetwork** — `MLPRegressor`, `MLPClassifier` with SGD solver.
- **FeatureSelection** — `VarianceThreshold`, `SelectKBest` (with `chi2`,
  `f_classif`, `f_regression`), `SelectFromModel`, `RFE`.
- **Manifold** — `Isomap`, `MDS` (SMACOF), `LocallyLinearEmbedding`,
  `SpectralEmbedding`, `TSNE` (exact), `UMAP`.
- **Calibration** — `CalibratedClassifierCV` (sigmoid + isotonic).
- **Isotonic** — public `IsotonicRegression` (PAVA).
- **LinearModel additions** — `BayesianRidge`, `ARDRegression`.
- **ModelSelection additions** — `GridSearchCV`, `RandomizedSearchCV`,
  `ParameterGrid`, `ParameterValue` reflection layer.

### Infrastructure

- New `IncrementalLike` concept and `partial_fit` contract documented in
  the developer guide.
- `SKIGEN_PARAMS(...)` reflection macro on every concrete estimator.
- 20 test executables; **376** unit tests, 100% passing.

### Breaking changes

All breaking changes are catalogued in
`doc/website/docs/development/migration-v1.1.0.mdx`. Summary:

- Regressor `coef_` is `MatrixType` (was `VectorType` for single-target).
  `coef_vec()` is the transitional helper.
- `Estimator::set_param(name, value)` / `get_params()` reflection surface
  added to the base class; concrete estimators register via `SKIGEN_PARAMS`.
- `LinearRegression` gains `positive` and `n_jobs` parameters.
- `Pipeline` accepts `(name, estimator)` named-step construction (positional
  form preserved); `"step__param"` nested syntax is now supported by
  `GridSearchCV`.
- Classification metrics that admit multi-output averaging return
  `MetricsResult` (variant of `Scalar` / `VectorType`).
- `KMeans::cluster_centers_` row/column convention pinned to
  `(n_clusters, n_features)`.

### Known gaps

- `NuSVC`, `NuSVR`, `OneClassSVM` are placeholder shells (throw from
  `fit`); the nu-SVM SMO solver is deferred.
- `PCA` and `BayesianRidge` sparse paths remain on the v1.1.x backlog;
  use `TruncatedSVD` for sparse decomposition in the meantime.
- Adam solver for `MLP` is on the v1.1.x backlog; SGD solver is the default
  and is fully functional.
- Parity-test CTest harness lands later in the v1.1 cycle.

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
