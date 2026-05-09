cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------
add_executable(skigen_tests
    test_standard_scaler.cpp
)

target_link_libraries(skigen_tests PRIVATE Skigen::Skigen)
target_compile_options(skigen_tests PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_tests COMMAND skigen_tests)

# ---------------------------------------------------------------------------
# Preprocessing tests (MinMaxScaler, MaxAbsScaler, RobustScaler, etc.)
# ---------------------------------------------------------------------------
add_executable(skigen_test_preprocessing
    test_preprocessing.cpp
)

target_link_libraries(skigen_test_preprocessing PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_preprocessing PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_preprocessing COMMAND skigen_test_preprocessing)

# ---------------------------------------------------------------------------
# Linear model tests (LinearRegression, Ridge)
# ---------------------------------------------------------------------------
add_executable(skigen_test_linear_model
    test_linear_model.cpp
)

target_link_libraries(skigen_test_linear_model PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_linear_model PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_linear_model COMMAND skigen_test_linear_model)

# ---------------------------------------------------------------------------
# Decomposition tests (PCA)
# ---------------------------------------------------------------------------
add_executable(skigen_test_decomposition
    test_decomposition.cpp
)

target_link_libraries(skigen_test_decomposition PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_decomposition PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_decomposition COMMAND skigen_test_decomposition)

# ---------------------------------------------------------------------------
# Cluster tests (KMeans)
# ---------------------------------------------------------------------------
add_executable(skigen_test_cluster
    test_cluster.cpp
)

target_link_libraries(skigen_test_cluster PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_cluster PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_cluster COMMAND skigen_test_cluster)

# ---------------------------------------------------------------------------
# Neighbors tests (KNeighborsClassifier, KNeighborsRegressor)
# ---------------------------------------------------------------------------
add_executable(skigen_test_neighbors
    test_neighbors.cpp
)

target_link_libraries(skigen_test_neighbors PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_neighbors PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_neighbors COMMAND skigen_test_neighbors)

# ---------------------------------------------------------------------------
# Tree tests (DecisionTreeClassifier, DecisionTreeRegressor)
# ---------------------------------------------------------------------------
add_executable(skigen_test_tree
    test_tree.cpp
)

target_link_libraries(skigen_test_tree PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_tree PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_tree COMMAND skigen_test_tree)

# ---------------------------------------------------------------------------
# Ensemble tests (RandomForestClassifier, RandomForestRegressor)
# ---------------------------------------------------------------------------
add_executable(skigen_test_ensemble
    test_ensemble.cpp
)

target_link_libraries(skigen_test_ensemble PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_ensemble PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_ensemble COMMAND skigen_test_ensemble)

# ---------------------------------------------------------------------------
# Model selection tests (TrainTestSplit)
# ---------------------------------------------------------------------------
add_executable(skigen_test_model_selection
    test_model_selection.cpp
)

target_link_libraries(skigen_test_model_selection PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_model_selection PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_model_selection COMMAND skigen_test_model_selection)

# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------
add_executable(skigen_test_metrics
    test_metrics.cpp
)

target_link_libraries(skigen_test_metrics PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_metrics PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_metrics COMMAND skigen_test_metrics)

# ---------------------------------------------------------------------------
# Remaining v1.0.0 tests (PolynomialFeatures, Lasso, ElasticNet,
# LogisticRegression, SGD, TruncatedSVD, MiniBatchKMeans, CrossVal, Pipeline)
# ---------------------------------------------------------------------------
add_executable(skigen_test_remaining
    test_remaining.cpp
)

target_link_libraries(skigen_test_remaining PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_remaining PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_remaining COMMAND skigen_test_remaining)

# ---------------------------------------------------------------------------
# Covariance tests (EmpiricalCovariance, LedoitWolf, OAS)
# ---------------------------------------------------------------------------
add_executable(skigen_test_covariance
    test_covariance.cpp
)

target_link_libraries(skigen_test_covariance PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_covariance PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_covariance COMMAND skigen_test_covariance)

# ---------------------------------------------------------------------------
# Naive Bayes tests (GaussianNB, MultinomialNB, BernoulliNB)
# ---------------------------------------------------------------------------
add_executable(skigen_test_naive_bayes
    test_naive_bayes.cpp
)

target_link_libraries(skigen_test_naive_bayes PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_naive_bayes PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_naive_bayes COMMAND skigen_test_naive_bayes)

# ---------------------------------------------------------------------------
# Feature selection tests (VarianceThreshold, SelectKBest, SelectFromModel, RFE)
# ---------------------------------------------------------------------------
add_executable(skigen_test_feature_selection
    test_feature_selection.cpp
)

target_link_libraries(skigen_test_feature_selection PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_feature_selection PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_feature_selection COMMAND skigen_test_feature_selection)

# ---------------------------------------------------------------------------
# Isotonic tests (IsotonicRegression — PAVA)
# ---------------------------------------------------------------------------
add_executable(skigen_test_isotonic
    test_isotonic.cpp
)

target_link_libraries(skigen_test_isotonic PRIVATE Skigen::Skigen)
target_compile_options(skigen_test_isotonic PRIVATE ${SKIGEN_WARNING_FLAGS})

add_test(NAME skigen_test_isotonic COMMAND skigen_test_isotonic)
