cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Helper: add an example target with standard flags
# ---------------------------------------------------------------------------
function(skigen_add_example name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE Skigen::Skigen)
    target_compile_options(${name} PRIVATE ${SKIGEN_WARNING_FLAGS})
endfunction()

# Augment an existing example target with SkigenPlot. No-op when
# SKIGEN_WITH_PLOT=OFF so the target stays dependency-free by default.
#
# Optional second argument is the image stem under
# doc/website/static/img/examples/. When given, the example is registered
# with the skigen_render_example_plots target, which runs it to emit
# <stem>_dark.png / <stem>_light.png for inclusion in the documentation.
function(skigen_enable_plot name)
    if(NOT SKIGEN_WITH_PLOT)
        return()
    endif()
    target_link_libraries(${name} PRIVATE SkigenPlot::SkigenPlot)
    target_compile_definitions(${name} PRIVATE SKIGEN_EXAMPLE_WITH_PLOT)
    if(ARGC GREATER 1)
        set_property(GLOBAL APPEND PROPERTY SKIGEN_PLOT_EXAMPLES "${name}|${ARGV1}")
    endif()
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
skigen_enable_plot(ex_kmeans              kmeans)

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
skigen_add_example(ex_svr                       svm/svr.cpp)
skigen_add_example(ex_nu_svc                    svm/nu_svc.cpp)
skigen_add_example(ex_nu_svr                    svm/nu_svr.cpp)
skigen_add_example(ex_one_class_svm             svm/one_class_svm.cpp)

# ---------------------------------------------------------------------------
# Manifold examples
# ---------------------------------------------------------------------------
skigen_add_example(ex_isomap                    manifold/isomap.cpp)
skigen_add_example(ex_mds                       manifold/mds.cpp)
skigen_add_example(ex_locally_linear_embedding  manifold/locally_linear_embedding.cpp)
skigen_add_example(ex_spectral_embedding        manifold/spectral_embedding.cpp)
skigen_add_example(ex_tsne                      manifold/tsne.cpp)
skigen_enable_plot(ex_tsne                      tsne)
skigen_add_example(ex_umap                      manifold/umap.cpp)

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
skigen_add_example(ex_grid_search         model_selection/grid_search.cpp)
skigen_add_example(ex_randomized_search   model_selection/randomized_search.cpp)

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
skigen_enable_plot(ex_pca_clustering_workflow   pca_clustering)

# ---------------------------------------------------------------------------
# Documentation plot rendering
#
# Runs every plot-enabled example registered with skigen_enable_plot(<tgt> <stem>)
# to emit <stem>_dark.png / <stem>_light.png into the Docusaurus static image
# tree, so the rendered figures can be referenced from the guide pages. Needs
# SKIGEN_WITH_PLOT=ON and a working SkigenPlot (Qt/RHI) runtime.
#
#   cmake -B build-plot -DSKIGEN_BUILD_EXAMPLES=ON -DSKIGEN_WITH_PLOT=ON
#   cmake --build build-plot --target skigen_render_example_plots
# ---------------------------------------------------------------------------
if(SKIGEN_WITH_PLOT)
    get_property(_skigen_plot_examples GLOBAL PROPERTY SKIGEN_PLOT_EXAMPLES)
    if(_skigen_plot_examples)
        set(_skigen_img_dir
            "${CMAKE_SOURCE_DIR}/doc/website/static/img/examples")
        set(_skigen_render_cmds
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_skigen_img_dir}")
        set(_skigen_render_deps "")
        foreach(_entry IN LISTS _skigen_plot_examples)
            string(REPLACE "|" ";" _parts "${_entry}")
            list(GET _parts 0 _tgt)
            list(GET _parts 1 _stem)
            list(APPEND _skigen_render_cmds
                COMMAND "$<TARGET_FILE:${_tgt}>" "${_skigen_img_dir}/${_stem}")
            list(APPEND _skigen_render_deps ${_tgt})
        endforeach()
        add_custom_target(skigen_render_example_plots
            ${_skigen_render_cmds}
            DEPENDS ${_skigen_render_deps}
            COMMENT "Rendering example plots into doc/website/static/img/examples/"
            VERBATIM)
    endif()
endif()
