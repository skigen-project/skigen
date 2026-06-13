#!/usr/bin/env python3
"""One-shot generator for the v1.1.0 per-estimator guide pages.

Emits docs/guide/<slug>.mdx from the structured table below. The generated
files are the maintained source of truth; this script documents how the
initial set was produced and keeps the format uniform.
"""

import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
GUIDE = os.path.join(HERE, "docs", "guide")

# Each entry: slug -> dict(title, header, api, blurb, algo, ctor, params,
# methods, attrs, example, parity)
# params: list of (name, default, desc); methods/attrs: list of (name, desc).
# parity: (generator_basename, data_dir) or ("placeholder", None) or None.

E = {}


def add(slug, **kw):
    E[slug] = kw


# --------------------------- Ensemble ------------------------------------
add("random-forest-classifier",
    title="RandomForestClassifier", header="Ensemble",
    api="api/ensemble/random-forest-classifier",
    blurb="An ensemble of decision trees, each grown on a bootstrap sample of "
          "the data and a random subspace of features. Class predictions are "
          "formed by majority (hard) or averaged-probability (soft) voting.",
    algo="Each tree is fit independently on a bootstrap resample; at every "
         "split a random subset of `max_features` candidate features is "
         "considered. Decorrelating the trees this way reduces the variance "
         "of the averaged ensemble without materially increasing bias. When "
         "`oob_score=true`, out-of-bag samples (those not drawn for a given "
         "tree) provide an unbiased generalisation estimate for free.",
    ctor="Skigen::RandomForestClassifier<Scalar> model(int n_estimators = 100, "
         "CriterionClf = Gini, std::optional<int> max_depth = nullopt, ...);",
    params=[("n_estimators", "100", "Number of trees in the forest."),
            ("criterion", "Gini", "Split quality measure (`Gini` or `Entropy`)."),
            ("max_depth", "nullopt", "Maximum tree depth; unbounded if unset."),
            ("max_features", "Sqrt", "Candidate features per split."),
            ("bootstrap", "true", "Sample with replacement per tree."),
            ("oob_score", "false", "Expose an out-of-bag accuracy estimate."),
            ("n_jobs", "1", "Trees fitted in parallel via `std::async`."),
            ("random_state", "nullopt", "Seed for reproducible forests.")],
    methods=[("fit(X, y)", "Grow the forest on labelled data."),
             ("predict(X)", "Majority-vote class labels."),
             ("predict_proba(X)", "Class probabilities (mean of tree votes)."),
             ("score(X, y)", "Mean accuracy.")],
    attrs=[("estimators()", "The fitted trees."),
           ("feature_importances()", "Mean impurity decrease per feature."),
           ("oob_score()", "Out-of-bag accuracy (when enabled).")],
    example="Skigen::RandomForestClassifier<double> rf(100);\n"
            "rf.fit(X, y);\nauto preds = rf.predict(X_test);",
    parity=("generate_ensemble_reference", "random_forest_classifier"))

add("random-forest-regressor",
    title="RandomForestRegressor", header="Ensemble",
    api="api/ensemble/random-forest-regressor",
    blurb="A bagged ensemble of regression trees whose predictions are "
          "averaged. Trades a small bias increase for a large variance "
          "reduction relative to a single deep tree.",
    algo="Identical bootstrap + feature-subspace scheme as the classifier, "
         "but each tree minimises squared error and the ensemble prediction "
         "is the mean of the per-tree outputs.",
    ctor="Skigen::RandomForestRegressor<Scalar> model(int n_estimators = 100, "
         "CriterionReg = MSE, std::optional<int> max_depth = nullopt, ...);",
    params=[("n_estimators", "100", "Number of trees."),
            ("max_features", "OneThird", "Candidate features per split."),
            ("bootstrap", "true", "Sample with replacement per tree."),
            ("oob_score", "false", "Expose an out-of-bag R² estimate."),
            ("n_jobs", "1", "Trees fitted in parallel."),
            ("random_state", "nullopt", "Seed.")],
    methods=[("fit(X, y)", "Grow the forest."),
             ("predict(X)", "Mean of per-tree predictions."),
             ("score(X, y)", "R² coefficient of determination.")],
    attrs=[("feature_importances()", "Mean impurity decrease per feature."),
           ("oob_score()", "Out-of-bag R² (when enabled).")],
    example="Skigen::RandomForestRegressor<double> rf(100);\n"
            "rf.fit(X, y);\nauto preds = rf.predict(X_test);",
    parity=("generate_ensemble_reference", "random_forest_regressor"))

add("gradient-boosting-classifier",
    title="GradientBoostingClassifier", header="Ensemble",
    api="api/ensemble/gradient-boosting-classifier",
    blurb="A stage-wise additive model that fits shallow regression trees to "
          "the negative gradient of the binary log-loss.",
    algo="Starting from the log-odds prior, each stage fits a regression tree "
         "to the pseudo-residuals (negative gradient) of the loss and adds it "
         "to the running prediction scaled by `learning_rate`. Binary "
         "log-loss is supported; multiclass and the exponential loss are "
         "rejected with a parity-gap message.",
    ctor="Skigen::GradientBoostingClassifier<Scalar> model(Loss = LogLoss, "
         "Scalar learning_rate = 0.1, int n_estimators = 100, ...);",
    params=[("learning_rate", "0.1", "Shrinkage applied to each stage."),
            ("n_estimators", "100", "Number of boosting stages."),
            ("max_depth", "3", "Depth of each regression tree."),
            ("subsample", "1.0", "Row fraction per stage (stochastic GB)."),
            ("random_state", "nullopt", "Seed.")],
    methods=[("fit(X, y)", "Boost over the stages."),
             ("predict(X)", "Thresholded class labels."),
             ("predict_proba(X)", "Sigmoid of the boosted score."),
             ("score(X, y)", "Mean accuracy.")],
    attrs=[("train_score()", "Per-stage training log-loss.")],
    example="Skigen::GradientBoostingClassifier<double> gb;\n"
            "gb.fit(X, y);\nauto p = gb.predict_proba(X_test);",
    parity=("generate_ensemble_reference", "gradient_boosting_classifier"))

add("gradient-boosting-regressor",
    title="GradientBoostingRegressor", header="Ensemble",
    api="api/ensemble/gradient-boosting-regressor",
    blurb="Stage-wise boosting of shallow regression trees against the "
          "squared-error gradient.",
    algo="Each stage fits a depth-bounded tree to the residuals of the "
         "current ensemble and adds it with a `learning_rate` shrinkage. "
         "Squared-error loss is supported in v1.1.0.",
    ctor="Skigen::GradientBoostingRegressor<Scalar> model(Loss = SquaredError, "
         "Scalar learning_rate = 0.1, int n_estimators = 100, ...);",
    params=[("learning_rate", "0.1", "Shrinkage per stage."),
            ("n_estimators", "100", "Number of stages."),
            ("max_depth", "3", "Tree depth."),
            ("subsample", "1.0", "Row fraction per stage."),
            ("random_state", "nullopt", "Seed.")],
    methods=[("fit(X, y)", "Boost over the stages."),
             ("predict(X)", "Sum of shrunk stage predictions."),
             ("score(X, y)", "R².")],
    attrs=[("train_score()", "Per-stage training MSE.")],
    example="Skigen::GradientBoostingRegressor<double> gb;\n"
            "gb.fit(X, y);\nauto preds = gb.predict(X_test);",
    parity=("generate_ensemble_reference", "gradient_boosting_regressor"))

add("hist-gradient-boosting-classifier",
    title="HistGradientBoostingClassifier", header="Ensemble",
    api="api/ensemble/hist-gradient-boosting-classifier",
    blurb="Histogram-based gradient boosting: features are binned up-front so "
          "split finding scans bin histograms rather than raw values, making "
          "training near-linear in the sample count.",
    algo="Each feature is quantile-binned into at most `max_bins` buckets. "
         "Split finding then operates on per-bin gradient/hessian histograms, "
         "which is dramatically faster on large datasets. Binary log-loss is "
         "supported.",
    ctor="Skigen::HistGradientBoostingClassifier<Scalar> model("
         "Scalar learning_rate = 0.1, int max_iter = 100, ...);",
    params=[("learning_rate", "0.1", "Shrinkage per iteration."),
            ("max_iter", "100", "Number of boosting iterations."),
            ("max_bins", "255", "Feature quantisation resolution."),
            ("max_leaf_nodes", "31", "Leaves per tree."),
            ("random_state", "nullopt", "Seed.")],
    methods=[("fit(X, y)", "Bin features, then boost."),
             ("predict(X)", "Class labels."),
             ("predict_proba(X)", "Calibrated-by-sigmoid scores."),
             ("score(X, y)", "Mean accuracy.")],
    attrs=[("bin_edges()", "Per-feature quantile bin edges."),
           ("train_score()", "Per-iteration training log-loss.")],
    example="Skigen::HistGradientBoostingClassifier<double> gb;\n"
            "gb.fit(X, y);\nauto preds = gb.predict(X_test);",
    parity=("generate_ensemble_reference", "hist_gradient_boosting_classifier"))

add("hist-gradient-boosting-regressor",
    title="HistGradientBoostingRegressor", header="Ensemble",
    api="api/ensemble/hist-gradient-boosting-regressor",
    blurb="Histogram-based gradient boosting for regression — the fast, "
          "large-data GB path.",
    algo="Features are quantile-binned, then squared-error boosting proceeds "
         "on the binned representation with histogram split finding.",
    ctor="Skigen::HistGradientBoostingRegressor<Scalar> model("
         "Scalar learning_rate = 0.1, int max_iter = 100, ...);",
    params=[("learning_rate", "0.1", "Shrinkage per iteration."),
            ("max_iter", "100", "Number of iterations."),
            ("max_bins", "255", "Bin resolution."),
            ("max_leaf_nodes", "31", "Leaves per tree."),
            ("random_state", "nullopt", "Seed.")],
    methods=[("fit(X, y)", "Bin features, then boost."),
             ("predict(X)", "Boosted prediction."),
             ("score(X, y)", "R².")],
    attrs=[("bin_edges()", "Per-feature quantile bin edges."),
           ("train_score()", "Per-iteration training MSE.")],
    example="Skigen::HistGradientBoostingRegressor<double> gb;\n"
            "gb.fit(X, y);\nauto preds = gb.predict(X_test);",
    parity=("generate_ensemble_reference", "hist_gradient_boosting_regressor"))

# --------------------------- Naive Bayes ---------------------------------
add("gaussian-nb",
    title="GaussianNB", header="Naive Bayes", api="api/naive-bayes/gaussian-nb",
    blurb="Gaussian Naive Bayes: each feature is modelled as class-conditional "
          "Gaussian, and predictions combine the per-feature likelihoods with "
          "the class priors under the naive independence assumption.",
    algo="For each class the per-feature mean and variance are estimated by "
         "maximum likelihood (a small `var_smoothing` floor is added for "
         "stability). Posterior class scores are the sum of Gaussian "
         "log-likelihoods plus the log prior. Supports streaming via "
         "`partial_fit` with Welford-style variance updates.",
    ctor="Skigen::GaussianNB<Scalar> model(VectorType priors = {}, "
         "Scalar var_smoothing = 1e-9);",
    params=[("priors", "empty", "Class priors; estimated from data if empty."),
            ("var_smoothing", "1e-9", "Variance floor as a fraction of the "
             "largest feature variance.")],
    methods=[("fit(X, y)", "Estimate per-class means and variances."),
             ("partial_fit(X, y, classes)", "Incremental update."),
             ("predict(X)", "MAP class labels."),
             ("predict_proba(X)", "Normalised class posteriors.")],
    attrs=[("theta()", "Per-class feature means."),
           ("var()", "Per-class feature variances."),
           ("class_prior()", "Class prior probabilities.")],
    example="Skigen::GaussianNB<double> nb;\nnb.fit(X, y);\n"
            "auto proba = nb.predict_proba(X_test);",
    parity=("generate_naive_bayes_reference", "gaussian_nb"))

add("multinomial-nb",
    title="MultinomialNB", header="Naive Bayes",
    api="api/naive-bayes/multinomial-nb",
    blurb="Multinomial Naive Bayes for count features — the classic "
          "bag-of-words text classifier. Sparse-aware.",
    algo="Per-class feature log-probabilities are the Laplace/Lidstone-"
         "smoothed relative frequencies of each feature within the class. "
         "Prediction sums `X · feature_log_prob + class_log_prior`. Accepts "
         "`Eigen::SparseMatrix` directly and supports `partial_fit`.",
    ctor="Skigen::MultinomialNB<Scalar> model(Scalar alpha = 1.0, "
         "bool fit_prior = true, VectorType class_prior = {});",
    params=[("alpha", "1.0", "Additive (Laplace/Lidstone) smoothing."),
            ("fit_prior", "true", "Learn class priors from data."),
            ("class_prior", "empty", "Optional fixed priors.")],
    methods=[("fit(X, y)", "Accumulate class/feature counts."),
             ("partial_fit(X, y, classes)", "Incremental update."),
             ("predict(X)", "MAP class labels."),
             ("predict_proba(X)", "Class posteriors.")],
    attrs=[("feature_log_prob()", "Smoothed log P(feature | class)."),
           ("class_log_prior()", "Log class priors."),
           ("feature_count()", "Accumulated per-class feature counts.")],
    example="Skigen::MultinomialNB<double> nb;\nnb.fit(X_counts, y);\n"
            "auto preds = nb.predict(X_test);",
    parity=("generate_naive_bayes_reference", "multinomial_nb"))

add("bernoulli-nb",
    title="BernoulliNB", header="Naive Bayes",
    api="api/naive-bayes/bernoulli-nb",
    blurb="Bernoulli Naive Bayes for binary/boolean features, with an explicit "
          "penalty for features that do *not* occur. Sparse-aware.",
    algo="Inputs are binarised at `binarize`. Each class stores the smoothed "
         "probability that each feature is 1; the decision function includes "
         "the log-probability of absence for unset features, distinguishing "
         "it from the multinomial model.",
    ctor="Skigen::BernoulliNB<Scalar> model(Scalar alpha = 1.0, "
         "bool fit_prior = true, std::optional<Scalar> binarize = 0.0);",
    params=[("alpha", "1.0", "Additive smoothing."),
            ("fit_prior", "true", "Learn class priors."),
            ("binarize", "0.0", "Threshold for binarising features; "
             "`nullopt` assumes pre-binarised input.")],
    methods=[("fit(X, y)", "Accumulate binary feature counts."),
             ("partial_fit(X, y, classes)", "Incremental update."),
             ("predict(X)", "MAP class labels."),
             ("predict_proba(X)", "Class posteriors.")],
    attrs=[("feature_log_prob()", "Log P(feature = 1 | class)."),
           ("class_log_prior()", "Log class priors.")],
    example="Skigen::BernoulliNB<double> nb;\nnb.fit(X_binary, y);\n"
            "auto preds = nb.predict(X_test);",
    parity=("generate_naive_bayes_reference", "bernoulli_nb"))

# --------------------------- SVM -----------------------------------------
add("linear-svc",
    title="LinearSVC", header="SVM", api="api/svm/linear-svc",
    blurb="Linear support-vector classifier trained with coordinate-descent / "
          "sub-gradient updates on the (squared) hinge loss. Scales to many "
          "samples and is sparse-aware.",
    algo="Minimises a regularised hinge objective directly in the primal. "
         "Multiclass uses one-vs-rest. Unlike kernel `SVC`, no support-vector "
         "set is materialised — only the weight vector and intercept.",
    ctor="Skigen::LinearSVC<Scalar> model(Scalar C = 1.0, Loss = SquaredHinge, "
         "Penalty = L2, ...);",
    params=[("C", "1.0", "Inverse regularisation strength."),
            ("loss", "SquaredHinge", "`Hinge` or `SquaredHinge`."),
            ("max_iter", "1000", "Optimiser iterations."),
            ("fit_intercept", "true", "Learn a bias term."),
            ("random_state", "nullopt", "Seed.")],
    methods=[("fit(X, y)", "Optimise the primal objective."),
             ("predict(X)", "Class labels."),
             ("decision_function(X)", "Signed margins.")],
    attrs=[("coef()", "Weight vector(s)."),
           ("intercept()", "Bias term(s).")],
    example="Skigen::LinearSVC<double> svc(1.0);\nsvc.fit(X, y);\n"
            "auto preds = svc.predict(X_test);",
    parity=("generate_svm_reference", "linear_svc"))

add("linear-svr",
    title="LinearSVR", header="SVM", api="api/svm/linear-svr",
    blurb="Linear support-vector regression with an epsilon-insensitive loss "
          "solved in the primal.",
    algo="Errors within `epsilon` of the target incur no penalty; outside the "
         "tube the loss is linear (or squared). The learning rate is scaled "
         "by a Lipschitz bound for stable convergence.",
    ctor="Skigen::LinearSVR<Scalar> model(Scalar C = 1.0, "
         "Scalar epsilon = 0.0, Loss = EpsilonInsensitive, ...);",
    params=[("C", "1.0", "Inverse regularisation strength."),
            ("epsilon", "0.0", "Width of the insensitive tube."),
            ("max_iter", "1000", "Optimiser iterations."),
            ("fit_intercept", "true", "Learn a bias term.")],
    methods=[("fit(X, y)", "Optimise the epsilon-insensitive objective."),
             ("predict(X)", "Real-valued predictions."),
             ("score(X, y)", "R².")],
    attrs=[("coef()", "Weight vector."), ("intercept()", "Bias term.")],
    example="Skigen::LinearSVR<double> svr(1.0);\nsvr.fit(X, y);\n"
            "auto preds = svr.predict(X_test);",
    parity=("generate_svm_reference", "linear_svr"))

add("svc",
    title="SVC", header="SVM", api="api/svm/svc",
    blurb="Kernel support-vector classifier built on a libsvm-style SMO "
          "solver, supporting linear, RBF, polynomial and sigmoid kernels.",
    algo="Sequential Minimal Optimization solves the dual quadratic program, "
         "selecting working pairs and updating their Lagrange multipliers "
         "until the KKT conditions hold. Multiclass uses one-vs-one voting. "
         "With `probability=true`, Platt scaling calibrates the margins into "
         "probabilities via internal cross-validation.",
    ctor="Skigen::SVC<Scalar> model(Scalar C = 1.0, Kernel = RBF, "
         "int degree = 3, Scalar gamma = scale, ...);",
    params=[("C", "1.0", "Penalty on margin violations."),
            ("kernel", "RBF", "`Linear`, `RBF`, `Poly` or `Sigmoid`."),
            ("degree", "3", "Polynomial-kernel degree."),
            ("gamma", "scale", "Kernel coefficient."),
            ("probability", "false", "Enable Platt-scaled probabilities.")],
    methods=[("fit(X, y)", "Solve the dual via SMO."),
             ("predict(X)", "Class labels."),
             ("decision_function(X)", "Margins (OvO for multiclass)."),
             ("predict_proba(X)", "Requires `probability=true`.")],
    attrs=[("n_support()", "Support vectors per class."),
           ("n_classes()", "Number of classes.")],
    example="Skigen::SVC<double> svc(1.0, Skigen::SVC<double>::Kernel::RBF);\n"
            "svc.fit(X, y);\nauto preds = svc.predict(X_test);",
    parity=("generate_svm_reference", "svc"))

add("svr",
    title="SVR", header="SVM", api="api/svm/svr",
    blurb="Kernel epsilon-support-vector regression using the same SMO solver "
          "and kernels as `SVC`.",
    algo="Fits the epsilon-insensitive dual with a kernel of choice; only "
         "samples outside the epsilon tube become support vectors.",
    ctor="Skigen::SVR<Scalar> model(Scalar C = 1.0, Kernel = RBF, "
         "Scalar epsilon = 0.1, ...);",
    params=[("C", "1.0", "Penalty strength."),
            ("kernel", "RBF", "Kernel policy."),
            ("epsilon", "0.1", "Insensitive-tube width."),
            ("gamma", "scale", "Kernel coefficient.")],
    methods=[("fit(X, y)", "Solve the epsilon-SVR dual."),
             ("predict(X)", "Real-valued predictions."),
             ("score(X, y)", "R².")],
    attrs=[("n_support()", "Number of support vectors.")],
    example="Skigen::SVR<double> svr;\nsvr.fit(X, y);\n"
            "auto preds = svr.predict(X_test);",
    parity=("generate_svm_reference", "svr"))

for slug, title, api, repl in [
    ("nu-svc", "NuSVC", "api/svm/nu-svc", "Skigen::SVC"),
    ("nu-svr", "NuSVR", "api/svm/nu-svr", "Skigen::SVR"),
    ("one-class-svm", "OneClassSVM", "api/svm/one-class-svm",
     "Skigen::SVC / novelty detection"),
]:
    add(slug, title=title, header="SVM", api=api,
        blurb=f"`{title}` ships as a documented **placeholder** in v1.1.0: the "
              "nu-parametrised formulation requires libsvm's specialised SMO "
              "variant, which is scheduled for a later release.",
        algo=f"The constructor and accessor surface are present so user code "
             f"can target the final API, but `fit()` throws `std::runtime_"
             f"error`. Use {repl} instead until the nu-solver lands.",
        ctor=f"Skigen::{title}<Scalar> model(Scalar nu = 0.5, ...);",
        params=[("nu", "0.5", "Upper bound on the fraction of margin errors "
                 "and lower bound on the fraction of support vectors.")],
        methods=[("fit(...)", "Throws — not yet implemented.")],
        attrs=[], example=f"// Placeholder: Skigen::{title} fit() throws in "
        "v1.1.0.", parity=("placeholder", None))

# --------------------------- Neural Network ------------------------------
add("mlp-classifier",
    title="MLPClassifier", header="Neural Network",
    api="api/neural-network/mlpclassifier",
    blurb="A dense feed-forward neural network for classification, trained "
          "with mini-batch Adam (or SGD) on the cross-entropy loss.",
    algo="Forward and backward passes are expressed entirely as Eigen "
         "matrix operations. The output layer is sigmoid (binary) or softmax "
         "(multiclass). Adam moment vectors persist across `partial_fit` "
         "calls for warm-started streaming.",
    ctor="Skigen::MLPClassifier<Scalar> model("
         "std::vector<int> hidden_layer_sizes = {100}, "
         "Activation = ReLU, Solver = Adam, ...);",
    params=[("hidden_layer_sizes", "{100}", "Units per hidden layer."),
            ("activation", "ReLU", "`Identity`, `Logistic`, `Tanh`, `ReLU`."),
            ("solver", "Adam", "`Adam` or `SGD` (`LBFGS` reserved)."),
            ("alpha", "1e-4", "L2 regularisation strength."),
            ("learning_rate_init", "1e-3", "Initial step size."),
            ("max_iter", "200", "Maximum epochs."),
            ("random_state", "nullopt", "Seed.")],
    methods=[("fit(X, y)", "Train the network."),
             ("predict(X)", "Class labels."),
             ("predict_proba(X)", "Softmax/sigmoid outputs."),
             ("score(X, y)", "Mean accuracy.")],
    attrs=[("coefs()", "Per-layer weight matrices."),
           ("intercepts()", "Per-layer bias vectors."),
           ("n_iter_run()", "Epochs actually run.")],
    example="Skigen::MLPClassifier<double> mlp({64, 32});\n"
            "mlp.fit(X, y);\nauto preds = mlp.predict(X_test);",
    parity=("generate_neural_network_reference", "mlp_classifier"))

add("mlp-regressor",
    title="MLPRegressor", header="Neural Network",
    api="api/neural-network/mlpregressor",
    blurb="A dense feed-forward neural network for regression with a linear "
          "output layer, trained with mini-batch Adam (or SGD).",
    algo="Minimises squared error with L2 weight decay. Multi-target "
         "regression is available through `fit_multi` / `predict_multi`.",
    ctor="Skigen::MLPRegressor<Scalar> model("
         "std::vector<int> hidden_layer_sizes = {100}, "
         "Activation = ReLU, Solver = Adam, ...);",
    params=[("hidden_layer_sizes", "{100}", "Units per hidden layer."),
            ("activation", "ReLU", "Hidden activation."),
            ("solver", "Adam", "`Adam` or `SGD`."),
            ("alpha", "1e-4", "L2 regularisation."),
            ("learning_rate_init", "1e-3", "Initial step size."),
            ("max_iter", "200", "Maximum epochs.")],
    methods=[("fit(X, y)", "Train on a single target."),
             ("fit_multi(X, Y)", "Train on multiple targets."),
             ("predict(X)", "Real-valued predictions."),
             ("score(X, y)", "R².")],
    attrs=[("coefs()", "Per-layer weights."),
           ("intercepts()", "Per-layer biases.")],
    example="Skigen::MLPRegressor<double> mlp({64});\n"
            "mlp.fit(X, y);\nauto preds = mlp.predict(X_test);",
    parity=("generate_neural_network_reference", "mlp_regressor"))

# --------------------------- Feature Selection ---------------------------
add("variance-threshold",
    title="VarianceThreshold", header="Feature Selection",
    api="api/feature-selection/variance-threshold",
    blurb="An unsupervised filter that drops features whose variance falls "
          "below a threshold — a cheap first pass for constant or "
          "near-constant columns. Sparse-aware.",
    algo="Computes each feature's variance in a single pass and keeps those "
         "above `threshold`. No labels required.",
    ctor="Skigen::VarianceThreshold<Scalar> model(Scalar threshold = 0.0);",
    params=[("threshold", "0.0", "Minimum variance to retain a feature.")],
    methods=[("fit(X)", "Compute per-feature variances."),
             ("transform(X)", "Drop low-variance columns."),
             ("get_support_mask()", "Boolean mask of retained features.")],
    attrs=[("variances()", "Per-feature variance.")],
    example="Skigen::VarianceThreshold<double> vt(0.0);\n"
            "auto X_reduced = vt.fit(X).transform(X);",
    parity=("generate_feature_selection_reference", "variance_threshold"))

add("select-k-best",
    title="SelectKBest", header="Feature Selection",
    api="api/feature-selection/select-kbest",
    blurb="Keeps the top-k features ranked by a univariate score function "
          "(`f_classif`, `f_regression`, or `chi2`).",
    algo="Each feature is scored independently against the target; the k "
         "highest-scoring features are retained. The chi-squared score is "
         "sparse-aware for text pipelines.",
    ctor="Skigen::SelectKBest<Scalar, ScoreFn> model(ScoreFn score, int k);",
    params=[("score_func", "—", "`FClassif`, `FRegression`, or `Chi2`."),
            ("k", "—", "Number of features to keep.")],
    methods=[("fit(X, y)", "Score and rank the features."),
             ("transform(X)", "Project onto the top-k features."),
             ("get_support_mask()", "Boolean mask of selected features.")],
    attrs=[("scores()", "Per-feature scores."),
           ("pvalues()", "Per-feature p-values.")],
    example="Skigen::SelectKBest<double, Skigen::feature_selection::"
            "FClassif<double>> sel({}, 5);\nsel.fit(X, y);\n"
            "auto X_top = sel.transform(X);",
    parity=("generate_feature_selection_reference", "f_classif"))

add("select-from-model",
    title="SelectFromModel", header="Feature Selection",
    api="api/feature-selection/select-from-model",
    blurb="A meta-transformer that selects features by thresholding the "
          "`coef_` or `feature_importances_` of a wrapped estimator.",
    algo="Fits the base estimator, then keeps features whose importance "
         "exceeds a threshold (`mean`, `median`, or a numeric value).",
    ctor="Skigen::SelectFromModel<Estimator> model(Estimator est, "
         "Threshold = Mean);",
    params=[("estimator", "—", "Estimator exposing coefficients/importances."),
            ("threshold", "Mean", "`Mean`, `Median`, or a numeric cutoff.")],
    methods=[("fit(X, y)", "Fit the base estimator and compute the mask."),
             ("transform(X)", "Keep features above the threshold."),
             ("get_support_mask()", "Boolean mask of selected features.")],
    attrs=[("estimator()", "The fitted base estimator.")],
    example="Skigen::SelectFromModel sel(Skigen::Ridge<double>(0.1));\n"
            "sel.fit(X, y);\nauto X_sel = sel.transform(X);",
    parity=None)

add("rfe",
    title="RFE", header="Feature Selection", api="api/feature-selection/rfe",
    blurb="Recursive Feature Elimination: repeatedly fit an estimator and "
          "discard the weakest feature until the target count remains.",
    algo="At each round the base estimator is refit on the surviving "
         "features and the one with the smallest absolute weight/importance "
         "is removed, yielding a ranking of all features.",
    ctor="Skigen::RFE<Estimator> model(Estimator est, "
         "int n_features_to_select);",
    params=[("estimator", "—", "Estimator exposing weights/importances."),
            ("n_features_to_select", "—", "Number of features to keep.")],
    methods=[("fit(X, y)", "Run the elimination loop."),
             ("transform(X)", "Project onto the selected features."),
             ("get_support_mask()", "Boolean mask of selected features.")],
    attrs=[("ranking()", "Elimination rank of each feature (1 = kept).")],
    example="Skigen::RFE rfe(Skigen::Ridge<double>(0.1), 3);\n"
            "rfe.fit(X, y);\nauto X_sel = rfe.transform(X);",
    parity=None)

# --------------------------- Bayesian Linear -----------------------------
add("bayesian-ridge",
    title="BayesianRidge", header="Linear Models",
    api="api/linear-model/bayesian-ridge",
    blurb="Bayesian ridge regression: a Gaussian linear model whose noise and "
          "weight precisions are estimated from the data by evidence "
          "maximisation, yielding predictive uncertainty for free.",
    algo="Gamma hyperpriors on the noise precision (alpha) and weight "
         "precision (lambda) are optimised by the iterative evidence-"
         "maximisation scheme of Tipping (2001). `predict(X, with_std)` "
         "returns the predictive mean and standard deviation.",
    ctor="Skigen::BayesianRidge<Scalar> model(int max_iter = 300, "
         "Scalar tol = 1e-3, Scalar alpha_1 = 1e-6, ...);",
    params=[("max_iter", "300", "Evidence-maximisation iterations."),
            ("tol", "1e-3", "Convergence tolerance."),
            ("alpha_1 / alpha_2", "1e-6", "Gamma prior on the noise precision."),
            ("lambda_1 / lambda_2", "1e-6", "Gamma prior on the weight precision."),
            ("compute_score", "false", "Track the log marginal likelihood.")],
    methods=[("fit(X, y)", "Optimise the evidence."),
             ("predict(X)", "Predictive mean."),
             ("predict(X, with_std)", "Mean and predictive std-dev.")],
    attrs=[("coef()", "Posterior mean weights."),
           ("alpha()", "Estimated noise precision."),
           ("lambda_()", "Estimated weight precision.")],
    example="Skigen::BayesianRidge<double> br;\nbr.fit(X, y);\n"
            "auto [mean, std] = br.predict(X_test, Skigen::with_std);",
    parity=("generate_bayesian_linear_reference", "bayesian_ridge"))

add("ard-regression",
    title="ARDRegression", header="Linear Models",
    api="api/linear-model/ardregression",
    blurb="Automatic Relevance Determination: Bayesian linear regression with "
          "a separate weight precision per feature, driving irrelevant "
          "coefficients to zero for sparse solutions.",
    algo="Identical evidence-maximisation machinery to `BayesianRidge` but "
         "with a per-feature lambda. Features whose precision exceeds "
         "`threshold_lambda` are pruned from the model.",
    ctor="Skigen::ARDRegression<Scalar> model(int max_iter = 300, "
         "Scalar tol = 1e-3, ..., Scalar threshold_lambda = 1e4);",
    params=[("max_iter", "300", "Iterations."),
            ("tol", "1e-3", "Convergence tolerance."),
            ("threshold_lambda", "1e4", "Prune features above this precision."),
            ("compute_score", "false", "Track the marginal likelihood.")],
    methods=[("fit(X, y)", "Optimise the per-feature evidence."),
             ("predict(X)", "Predictive mean."),
             ("predict(X, with_std)", "Mean and predictive std-dev.")],
    attrs=[("coef()", "Posterior mean weights (sparse)."),
           ("alpha()", "Noise precision."),
           ("lambda_()", "Per-feature weight precisions.")],
    example="Skigen::ARDRegression<double> ard;\nard.fit(X, y);\n"
            "auto preds = ard.predict(X_test);",
    parity=("generate_bayesian_linear_reference", "ard_regression"))

# --------------------------- Manifold ------------------------------------
add("isomap",
    title="Isomap", header="Manifold", api="api/manifold/isomap",
    blurb="Isometric mapping: a non-linear embedding that preserves geodesic "
          "(along-the-manifold) distances rather than straight-line ones.",
    algo="Builds a k-nearest-neighbour graph, approximates geodesic distances "
         "by shortest paths through it, then applies classical MDS to the "
         "geodesic distance matrix.",
    ctor="Skigen::Isomap<Scalar> model(int n_components = 2, "
         "int n_neighbors = 5);",
    params=[("n_components", "2", "Embedding dimensionality."),
            ("n_neighbors", "5", "Neighbours per point in the graph.")],
    methods=[("fit_transform(X)", "Compute and return the embedding.")],
    attrs=[("dist_matrix()", "Geodesic distance matrix."),
           ("reconstruction_error()", "Residual of the MDS fit.")],
    example="Skigen::Isomap<double> iso(2, 5);\n"
            "auto Y = iso.fit_transform(X);",
    parity=("generate_manifold_reference", "isomap"))

add("mds",
    title="MDS", header="Manifold", api="api/manifold/mds",
    blurb="Multidimensional scaling: places points in a low-dimensional space "
          "so that pairwise distances match the originals as closely as "
          "possible.",
    algo="Metric (and non-metric) MDS via the SMACOF majorisation "
         "iteration, which monotonically decreases the stress functional.",
    ctor="Skigen::MDS<Scalar> model(int n_components = 2, int max_iter = 300, "
         "Scalar eps = 1e-3, bool metric = true, uint64_t random_state = 0);",
    params=[("n_components", "2", "Embedding dimensionality."),
            ("max_iter", "300", "SMACOF iterations."),
            ("eps", "1e-3", "Stress-convergence tolerance."),
            ("metric", "true", "Metric vs non-metric MDS.")],
    methods=[("fit_transform(X)", "Return the embedding.")],
    attrs=[("stress()", "Final stress value.")],
    example="Skigen::MDS<double> mds(2);\nauto Y = mds.fit_transform(X);",
    parity=("generate_manifold_reference", "mds"))

add("locally-linear-embedding",
    title="LocallyLinearEmbedding", header="Manifold",
    api="api/manifold/locally-linear-embedding",
    blurb="LLE: embeds data by preserving the local linear reconstruction "
          "weights of each point from its neighbours.",
    algo="Each point is expressed as a weighted combination of its k nearest "
         "neighbours; the embedding minimises the same reconstruction error "
         "in low dimensions via a sparse eigenproblem (standard variant).",
    ctor="Skigen::LocallyLinearEmbedding<Scalar> model(int n_components = 2, "
         "int n_neighbors = 5, Scalar reg = 1e-3);",
    params=[("n_components", "2", "Embedding dimensionality."),
            ("n_neighbors", "5", "Neighbours per point."),
            ("reg", "1e-3", "Regularisation of the local Gram matrix.")],
    methods=[("fit_transform(X)", "Return the embedding.")],
    attrs=[("reconstruction_error()", "Embedding reconstruction error.")],
    example="Skigen::LocallyLinearEmbedding<double> lle(2, 5);\n"
            "auto Y = lle.fit_transform(X);",
    parity=("generate_manifold_reference", "locally_linear_embedding"))

add("spectral-embedding",
    title="SpectralEmbedding", header="Manifold",
    api="api/manifold/spectral-embedding",
    blurb="Laplacian eigenmaps: a spectral embedding from the eigenvectors of "
          "the graph Laplacian of a neighbourhood affinity graph.",
    algo="Constructs an affinity graph (nearest-neighbours by default), forms "
         "the normalised Laplacian, and embeds using its smallest non-trivial "
         "eigenvectors.",
    ctor="Skigen::SpectralEmbedding<Scalar> model(int n_components = 2, "
         "int n_neighbors = 5);",
    params=[("n_components", "2", "Embedding dimensionality."),
            ("n_neighbors", "5", "Neighbours in the affinity graph.")],
    methods=[("fit_transform(X)", "Return the embedding.")],
    attrs=[("affinity_matrix()", "The constructed affinity graph.")],
    example="Skigen::SpectralEmbedding<double> se(2, 5);\n"
            "auto Y = se.fit_transform(X);",
    parity=("generate_manifold_reference", "spectral_embedding"))

add("tsne",
    title="TSNE", header="Manifold", api="api/manifold/tsne",
    blurb="t-distributed Stochastic Neighbor Embedding: a popular non-linear "
          "method for visualising high-dimensional data in 2-D/3-D.",
    algo="Models pairwise similarities as conditional probabilities in both "
         "spaces and minimises their KL divergence by gradient descent, using "
         "a heavy-tailed Student-t kernel in the embedding to avoid crowding. "
         "v1.1.0 implements the exact (non-Barnes-Hut) gradient.",
    ctor="Skigen::TSNE<Scalar> model(int n_components = 2, "
         "Scalar perplexity = 30.0, Scalar learning_rate = 200.0, "
         "int max_iter = 1000, uint64_t random_state = 0);",
    params=[("n_components", "2", "Embedding dimensionality."),
            ("perplexity", "30.0", "Effective neighbourhood size."),
            ("learning_rate", "200.0", "Gradient-descent step size."),
            ("max_iter", "1000", "Optimisation iterations.")],
    methods=[("fit_transform(X)", "Return the embedding.")],
    attrs=[("kl_divergence()", "Final KL divergence."),
           ("n_iter()", "Iterations run.")],
    example="Skigen::TSNE<double> tsne(2, 30.0);\n"
            "auto Y = tsne.fit_transform(X);",
    parity=("generate_manifold_reference", "tsne"))

add("umap",
    title="UMAP", header="Manifold", api="api/manifold/umap",
    blurb="Uniform Manifold Approximation and Projection: a fast non-linear "
          "embedding that preserves both local and some global structure. An "
          "independent C++ port (BSD-3 attribution in the source header).",
    algo="Builds a fuzzy topological representation of the data via local "
         "fuzzy simplicial sets, then optimises a low-dimensional layout by "
         "stochastic gradient descent on the cross-entropy between the two "
         "fuzzy structures.",
    ctor="Skigen::UMAP<Scalar> model(int n_components = 2, "
         "int n_neighbors = 15, Scalar min_dist = 0.1, Scalar spread = 1.0, "
         "int n_epochs = 0, int negative_sample_rate = 5, "
         "uint64_t random_state = 0);",
    params=[("n_components", "2", "Embedding dimensionality."),
            ("n_neighbors", "15", "Local neighbourhood size."),
            ("min_dist", "0.1", "Minimum spacing in the embedding."),
            ("spread", "1.0", "Scale of embedded points.")],
    methods=[("fit_transform(X)", "Return the embedding.")],
    attrs=[("n_iter()", "Optimisation epochs run.")],
    example="Skigen::UMAP<double> umap(2, 15);\n"
            "auto Y = umap.fit_transform(X);",
    parity=("generate_manifold_reference", "umap"))

# --------------------------- Calibration / Isotonic ----------------------
add("calibrated-classifier-cv",
    title="CalibratedClassifierCV", header="Calibration",
    api="api/calibration/calibrated-classifier-cv",
    blurb="Post-hoc probability calibration that maps a base classifier's "
          "scores onto well-calibrated probabilities using sigmoid (Platt) or "
          "isotonic regression fit by cross-validation.",
    algo="For each CV fold the base estimator is fit on the training part and "
         "a calibrator (1-D logistic regression for sigmoid, PAVA for "
         "isotonic) is fit on the held-out scores. Predictions average the "
         "per-fold calibrators. v1.1.0 covers binary classification.",
    ctor="Skigen::CalibratedClassifierCV<Base, Scalar> model(Base est, "
         "Method = Sigmoid, int cv = 5, int n_jobs = 1, bool ensemble = true);",
    params=[("estimator", "—", "Base classifier to calibrate."),
            ("method", "Sigmoid", "`Sigmoid` (Platt) or `Isotonic`."),
            ("cv", "5", "Cross-validation folds."),
            ("ensemble", "true", "Average per-fold calibrators.")],
    methods=[("fit(X, y)", "Fit base estimators and calibrators."),
             ("predict(X)", "Calibrated class labels."),
             ("predict_proba(X)", "Calibrated probabilities.")],
    attrs=[("n_classes()", "Number of classes."),
           ("n_estimators_fitted()", "Calibrated base estimators held.")],
    example="Skigen::GaussianNB<double> nb;\n"
            "Skigen::CalibratedClassifierCV cc(nb, "
            "Skigen::CalibrationMethod::Sigmoid, 5);\ncc.fit(X, y);",
    parity=("generate_calibration_reference", "calibrated_classifier_cv"))

add("isotonic-regression",
    title="IsotonicRegression", header="Isotonic",
    api="api/isotonic/isotonic-regression",
    blurb="Non-parametric monotonic regression: fits the best non-decreasing "
          "(or non-increasing) step function to 1-D data via the "
          "Pool-Adjacent-Violators Algorithm.",
    algo="PAVA merges adjacent blocks that violate monotonicity, replacing "
         "them with their weighted mean, in O(n log n). Out-of-range queries "
         "follow the `out_of_bounds` policy (NaN, clip, or raise).",
    ctor="Skigen::IsotonicRegression<Scalar> model("
         "std::optional<Scalar> y_min = nullopt, "
         "std::optional<Scalar> y_max = nullopt, "
         "IsotonicIncreasing = True, OutOfBounds = Nan);",
    params=[("y_min / y_max", "nullopt", "Optional clamps on the output."),
            ("increasing", "True", "Direction (`True`, `False`, `Auto`)."),
            ("out_of_bounds", "Nan", "Behaviour outside the training range.")],
    methods=[("fit(X, y)", "Run PAVA."),
             ("predict(X)", "Interpolate the fitted step function."),
             ("transform(X)", "Alias of `predict`.")],
    attrs=[("increasing_resolved()", "Resolved monotonic direction.")],
    example="Skigen::IsotonicRegression<double> iso;\niso.fit(X, y);\n"
            "auto yhat = iso.predict(X_test);",
    parity=("generate_isotonic_reference", "isotonic"))

# --------------------------- Model Selection -----------------------------
add("grid-search-cv",
    title="GridSearchCV", header="Model Selection",
    api="api/model-selection/grid-search-cv",
    blurb="Exhaustive cross-validated search over a parameter grid, refitting "
          "the best configuration on the full training set.",
    algo="Every point in the Cartesian product of the grid is evaluated by "
         "K-fold cross-validation. Grid points are dispatched across threads "
         "with `n_jobs` (parallelism is over the grid, not the folds). Over a "
         "pipeline, parameters are addressed by step index, e.g. `1__alpha`.",
    ctor="Skigen::GridSearchCV<Estimator> model(Estimator est, "
         "ParameterGrid grid, int cv = 5, bool refit = true, int n_jobs = 1);",
    params=[("estimator", "—", "Estimator to tune."),
            ("param_grid", "—", "Grid of parameter values."),
            ("cv", "5", "Cross-validation folds."),
            ("n_jobs", "1", "Grid points evaluated in parallel (`-1` = all).")],
    methods=[("fit(X, y)", "Search and refit the best estimator."),
             ("predict(X)", "Predict with the best estimator."),
             ("best_params()", "Best parameter dictionary."),
             ("best_score()", "Best mean CV score.")],
    attrs=[("best_score()", "Best mean cross-validation score."),
           ("cv_results_mean_score()", "Mean score per grid point.")],
    example="Skigen::ParameterGrid grid({{\"alpha\", {0.1, 1.0, 10.0}}});\n"
            "Skigen::GridSearchCV<Skigen::Ridge<double>> gs("
            "Skigen::Ridge<double>(), grid, 5, true, -1);\ngs.fit(X, y);",
    parity=("generate_model_selection_reference", "grid_search_cv"))

add("randomized-search-cv",
    title="RandomizedSearchCV", header="Model Selection",
    api="api/model-selection/randomized-search-cv",
    blurb="Cross-validated search over random samples from parameter "
          "distributions — a budgeted alternative to exhaustive grid search.",
    algo="Draws `n_iter` parameter combinations from the supplied "
         "distributions and evaluates each by K-fold cross-validation, "
         "refitting the best on the full data.",
    ctor="Skigen::RandomizedSearchCV<Estimator> model(Estimator est, "
         "ParameterGrid distributions, int n_iter = 10, int cv = 5, "
         "bool refit = true, uint64_t random_state = nullopt);",
    params=[("estimator", "—", "Estimator to tune."),
            ("param_distributions", "—", "Distributions to sample."),
            ("n_iter", "10", "Number of sampled configurations."),
            ("cv", "5", "Cross-validation folds."),
            ("random_state", "nullopt", "Seed for sampling.")],
    methods=[("fit(X, y)", "Sample, search, and refit the best."),
             ("predict(X)", "Predict with the best estimator."),
             ("best_params()", "Best sampled parameters."),
             ("best_score()", "Best mean CV score.")],
    attrs=[("best_score()", "Best mean cross-validation score."),
           ("cv_results_params()", "Sampled parameter sets.")],
    example="Skigen::RandomizedSearchCV<Skigen::Ridge<double>> rs("
            "Skigen::Ridge<double>(), dist, 10, 5);\nrs.fit(X, y);",
    parity=("generate_model_selection_reference", "randomized_search_cv"))

# --------------------------- Covariance ----------------------------------
add("empirical-covariance",
    title="EmpiricalCovariance", header="Covariance",
    api="api/covariance/empirical-covariance",
    blurb="The maximum-likelihood (sample) covariance estimator.",
    algo="Centres the data (unless `assume_centered`) and forms the 1/n "
         "scatter matrix. The maximum-likelihood normalisation matches "
         "scikit-learn.",
    ctor="Skigen::EmpiricalCovariance<Scalar> model("
         "bool assume_centered = false);",
    params=[("assume_centered", "false", "Skip mean subtraction if data is "
             "already centred.")],
    methods=[("fit(X)", "Estimate the covariance."),
             ("score(X)", "Gaussian log-likelihood of new data.")],
    attrs=[("covariance()", "Estimated p×p covariance."),
           ("location()", "Estimated mean.")],
    example="Skigen::EmpiricalCovariance<double> cov;\ncov.fit(X);\n"
            "auto C = cov.covariance();",
    parity=("generate_covariance_reference", "empirical_covariance"))

add("ledoit-wolf",
    title="LedoitWolf", header="Covariance",
    api="api/covariance/ledoit-wolf",
    blurb="Shrinkage covariance with the Ledoit-Wolf data-driven shrinkage "
          "intensity — better conditioned than the sample covariance, "
          "especially when p is comparable to n.",
    algo="Shrinks the sample covariance toward a scaled identity by the "
         "closed-form Ledoit-Wolf coefficient that minimises expected "
         "Frobenius error.",
    ctor="Skigen::LedoitWolf<Scalar> model(bool assume_centered = false);",
    params=[("assume_centered", "false", "Skip mean subtraction.")],
    methods=[("fit(X)", "Estimate the shrunk covariance."),
             ("score(X)", "Gaussian log-likelihood.")],
    attrs=[("covariance()", "Shrunk covariance."),
           ("shrinkage()", "Estimated shrinkage intensity.")],
    example="Skigen::LedoitWolf<double> lw;\nlw.fit(X);\n"
            "auto C = lw.covariance();",
    parity=("generate_covariance_reference", "ledoit_wolf"))

add("oas",
    title="OAS", header="Covariance", api="api/covariance/oas",
    blurb="Oracle Approximating Shrinkage covariance — a shrinkage estimator "
          "whose intensity targets the oracle under Gaussian assumptions, "
          "often improving on Ledoit-Wolf for Gaussian data.",
    algo="Uses the OAS closed-form shrinkage coefficient (the scikit-learn-"
         "corrected formula) to shrink the sample covariance toward a scaled "
         "identity.",
    ctor="Skigen::OAS<Scalar> model(bool assume_centered = false);",
    params=[("assume_centered", "false", "Skip mean subtraction.")],
    methods=[("fit(X)", "Estimate the shrunk covariance."),
             ("score(X)", "Gaussian log-likelihood.")],
    attrs=[("covariance()", "Shrunk covariance."),
           ("shrinkage()", "OAS shrinkage intensity.")],
    example="Skigen::OAS<double> oas;\noas.fit(X);\n"
            "auto C = oas.covariance();",
    parity=("generate_covariance_reference", "oas"))

# --------------------------- Decomposition / Neighbors -------------------
add("factor-analysis",
    title="FactorAnalysis", header="Decomposition",
    api="api/decomposition/factor-analysis",
    blurb="A latent linear-Gaussian model that explains observed features as "
          "a few common factors plus per-feature (heteroscedastic) noise — "
          "unlike PCA, which assumes isotropic noise.",
    algo="Loadings and noise variances are fit by expectation-maximisation. "
         "Loadings are identifiable only up to rotation, so downstream use "
         "should rely on the implied covariance or the transformed factors.",
    ctor="Skigen::FactorAnalysis<Scalar> model(int n_components = 0, "
         "int max_iter = 1000, Scalar tol = 1e-3);",
    params=[("n_components", "0", "Number of latent factors."),
            ("max_iter", "1000", "EM iterations."),
            ("tol", "1e-3", "Log-likelihood convergence tolerance.")],
    methods=[("fit(X)", "Run EM."),
             ("transform(X)", "Project onto the latent factors."),
             ("get_covariance()", "Model-implied covariance.")],
    attrs=[("components()", "Loading matrix W."),
           ("noise_variance()", "Per-feature noise variance."),
           ("log_likelihood()", "Final log-likelihood.")],
    example="Skigen::FactorAnalysis<double> fa(2);\nfa.fit(X);\n"
            "auto Z = fa.transform(X);",
    parity=("generate_decomposition_reference", "factor_analysis"))

add("local-outlier-factor",
    title="LocalOutlierFactor", header="Neighbors",
    api="api/neighbors/local-outlier-factor",
    blurb="A density-based anomaly score: each point is compared to the local "
          "density of its neighbours, flagging those in relatively sparse "
          "regions as outliers.",
    algo="Computes the local reachability density of each point and the ratio "
         "to its neighbours' densities (the local outlier factor). Scores far "
         "below 1 indicate inliers; large scores indicate outliers. Exposed "
         "as the negated factor to match scikit-learn's convention.",
    ctor="Skigen::LocalOutlierFactor<Scalar> model(int n_neighbors = 20, "
         "Scalar contamination = -1);",
    params=[("n_neighbors", "20", "Neighbours used for the density estimate."),
            ("contamination", "-1", "Expected outlier fraction; `-1` disables "
             "automatic thresholding.")],
    methods=[("fit(X)", "Score every training point."),
             ("fit_predict_labels(X)", "Inlier/outlier labels.")],
    attrs=[("negative_outlier_factor()", "Negated LOF scores (lower = more "
            "anomalous)."),
           ("n_neighbors_used()", "Effective neighbour count.")],
    example="Skigen::LocalOutlierFactor<double> lof(20);\nlof.fit(X);\n"
            "auto scores = lof.negative_outlier_factor();",
    parity=("generate_neighbors_reference", "local_outlier_factor"))


# --------------------------- Emit ----------------------------------------
def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def render(slug, spec, position):
    lines = [f"---", f"sidebar_position: {position}", f"title: {spec['title']}",
             f"---", "", f"# {spec['title']}", "", spec["blurb"], "",
             "## Algorithm", "", spec["algo"], "", "## Constructor", "",
             "```cpp", spec["ctor"], "```", ""]
    if spec["params"]:
        lines += ["## Parameters", "",
                  table(["Parameter", "Default", "Description"],
                        [[f"`{n}`", f"`{d}`", t] for n, d, t in spec["params"]]),
                  ""]
    if spec["methods"]:
        lines += ["## Methods", "",
                  table(["Method", "Description"],
                        [[f"`{n}`", t] for n, t in spec["methods"]]), ""]
    if spec["attrs"]:
        lines += ["## Fitted Attributes", "",
                  table(["Accessor", "Description"],
                        [[f"`{n}`", t] for n, t in spec["attrs"]]), ""]
    lines += ["## Example", "", "```cpp", spec["example"], "```", ""]

    parity = spec["parity"]
    if parity and parity[0] == "placeholder":
        lines += [":::caution Placeholder",
                  f"`{spec['title']}` is an API placeholder in v1.1.0 — `fit()` "
                  "throws until the nu-SVM solver lands. It is intentionally "
                  "excluded from behavioural parity.", ":::", ""]
    elif parity:
        gen, data = parity
        parity_file = {"Linear Models": "bayesian_linear",
                       "Naive Bayes": "naive_bayes",
                       "Feature Selection": "feature_selection",
                       "Neural Network": "neural_network",
                       "Model Selection": "model_selection",
                       }.get(spec["header"],
                             spec["header"].lower().replace(" ", "_"))
        lines += [":::note Verified against scikit-learn",
                  f"This estimator is checked by the parity suite. See the "
                  f"generator `tests/parity/{gen}.py` and the reference "
                  f"fixtures in `tests/parity/data/{data}/`, exercised by "
                  f"`tests/parity/parity_{parity_file}.cpp`.", ":::", ""]
    else:
        lines += [":::note Verified by unit tests",
                  "Covered by the module's CTest suite under `tests/`.", ":::",
                  ""]

    lines += [":::tip API Reference",
              f"For full signatures see the [{spec['title']} API Reference]"
              f"(/docs/{spec['api']}).", ":::", ""]
    return "\n".join(lines)


def main():
    for i, (slug, spec) in enumerate(E.items(), start=100):
        path = os.path.join(GUIDE, slug + ".mdx")
        with open(path, "w") as fh:
            fh.write(render(slug, spec, i))
    print(f"Wrote {len(E)} guide pages to {GUIDE}")


if __name__ == "__main__":
    main()
