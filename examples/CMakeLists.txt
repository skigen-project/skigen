cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Helper: add an example target with standard flags
# ---------------------------------------------------------------------------
function(skigen_add_example name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE Skigen::Skigen)
    target_compile_options(${name} PRIVATE ${SKIGEN_WARNING_FLAGS})
endfunction()

# ---------------------------------------------------------------------------
# Preprocessing examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_standard_scaler     preprocessing/standard_scaler.cpp)
skigen_add_example(ex_minmax_scaler       preprocessing/minmax_scaler.cpp)
skigen_add_example(ex_maxabs_scaler       preprocessing/maxabs_scaler.cpp)
skigen_add_example(ex_robust_scaler       preprocessing/robust_scaler.cpp)
skigen_add_example(ex_normalizer          preprocessing/normalizer.cpp)
skigen_add_example(ex_label_encoder       preprocessing/label_encoder.cpp)
skigen_add_example(ex_polynomial_features preprocessing/polynomial_features.cpp)

# ---------------------------------------------------------------------------
# LinearModel examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_linear_regression   linear_model/linear_regression.cpp)
skigen_add_example(ex_ridge               linear_model/ridge.cpp)
skigen_add_example(ex_lasso               linear_model/lasso.cpp)
skigen_add_example(ex_elastic_net         linear_model/elastic_net.cpp)
skigen_add_example(ex_logistic_regression linear_model/logistic_regression.cpp)
skigen_add_example(ex_sgd                 linear_model/sgd.cpp)

# ---------------------------------------------------------------------------
# Decomposition examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_pca                 decomposition/pca.cpp)
skigen_add_example(ex_truncated_svd       decomposition/truncated_svd.cpp)

# ---------------------------------------------------------------------------
# Cluster examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_kmeans              cluster/kmeans.cpp)

# ---------------------------------------------------------------------------
# Neighbors examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_kneighbors          neighbors/kneighbors.cpp)

# ---------------------------------------------------------------------------
# Tree examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_decision_tree       tree/decision_tree.cpp)

# ---------------------------------------------------------------------------
# Metrics examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_metrics             metrics/metrics.cpp)

# ---------------------------------------------------------------------------
# ModelSelection examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_model_selection     model_selection/model_selection.cpp)

# ---------------------------------------------------------------------------
# Pipeline examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_pipeline            pipeline/pipeline.cpp)

# ---------------------------------------------------------------------------
# Integration / workflow examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_classification_workflow   classification_workflow.cpp)
skigen_add_example(ex_regression_pipeline       regression_pipeline.cpp)
skigen_add_example(ex_pca_clustering_workflow   pca_clustering_workflow.cpp)
