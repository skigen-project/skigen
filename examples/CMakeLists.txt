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
skigen_add_example(ex_bayesian_ridge      linear_model/bayesian_ridge.cpp)
skigen_add_example(ex_ard_regression      linear_model/ard_regression.cpp)

# ---------------------------------------------------------------------------
# Decomposition examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_pca                 decomposition/pca.cpp)
skigen_add_example(ex_truncated_svd       decomposition/truncated_svd.cpp)
skigen_add_example(ex_factor_analysis     decomposition/factor_analysis.cpp)

# ---------------------------------------------------------------------------
# Covariance examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_covariance          covariance/covariance.cpp)

# ---------------------------------------------------------------------------
# Cluster examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_kmeans              cluster/kmeans.cpp)

# ---------------------------------------------------------------------------
# Neighbors examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_kneighbors          neighbors/kneighbors.cpp)
skigen_add_example(ex_lof                 neighbors/local_outlier_factor.cpp)

# ---------------------------------------------------------------------------
# Tree examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_decision_tree       tree/decision_tree.cpp)

# ---------------------------------------------------------------------------
# Naive Bayes examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_gaussian_nb         naive_bayes/gaussian_nb.cpp)
skigen_add_example(ex_multinomial_nb      naive_bayes/multinomial_nb.cpp)
skigen_add_example(ex_bernoulli_nb        naive_bayes/bernoulli_nb.cpp)

# ---------------------------------------------------------------------------
# Feature selection examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_variance_threshold  feature_selection/variance_threshold.cpp)
skigen_add_example(ex_select_k_best       feature_selection/select_k_best.cpp)
skigen_add_example(ex_select_from_model   feature_selection/select_from_model.cpp)
skigen_add_example(ex_rfe                 feature_selection/rfe.cpp)

# ---------------------------------------------------------------------------
# Isotonic examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_isotonic_regression isotonic/isotonic_regression.cpp)

# ---------------------------------------------------------------------------
# Calibration examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_calibrated_classifier_cv calibration/calibrated_classifier_cv.cpp)

# ---------------------------------------------------------------------------
# SVM examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_linear_svc                svm/linear_svc.cpp)
skigen_add_example(ex_linear_svr                svm/linear_svr.cpp)
skigen_add_example(ex_svc                       svm/svc.cpp)

# ---------------------------------------------------------------------------
# Neural network examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_mlp_classifier            neural_network/mlp_classifier.cpp)
skigen_add_example(ex_mlp_regressor             neural_network/mlp_regressor.cpp)

# ---------------------------------------------------------------------------
# Ensemble examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_random_forest_classifier    ensemble/random_forest_classifier.cpp)
skigen_add_example(ex_random_forest_regressor     ensemble/random_forest_regressor.cpp)
skigen_add_example(ex_gradient_boosting_regressor  ensemble/gradient_boosting_regressor.cpp)
skigen_add_example(ex_gradient_boosting_classifier ensemble/gradient_boosting_classifier.cpp)
skigen_add_example(ex_hist_gradient_boosting_regressor  ensemble/hist_gradient_boosting_regressor.cpp)
skigen_add_example(ex_hist_gradient_boosting_classifier ensemble/hist_gradient_boosting_classifier.cpp)

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
