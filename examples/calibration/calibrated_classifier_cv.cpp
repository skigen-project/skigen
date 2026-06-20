// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// calibrated_classifier_cv.cpp — wrap a GaussianNB binary classifier with
// CalibratedClassifierCV (sigmoid Platt scaling).
//
// Equivalent scikit-learn snippet:
//
//   from sklearn.calibration import CalibratedClassifierCV
//   from sklearn.naive_bayes import GaussianNB
//   import numpy as np
//   rng = np.random.default_rng(42)
//   X = np.vstack([rng.normal(loc=-1.5, size=(100, 2)),
//                  rng.normal(loc= 1.5, size=(100, 2))])
//   y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
//   cc = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5)
//   cc.fit(X, y)
//   print("training accuracy =", cc.score(X, y))
//   print("first row probas  =", cc.predict_proba(X[:1]))

#include <Skigen/Calibration>
#include <Skigen/NaiveBayes>

#include <Eigen/Core>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>
#include <random>
#include <vector>

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
#include <skigen/plot/figure.h>

namespace {

auto calibrationCurve(const Eigen::VectorXi& labels,
                      const Eigen::VectorXd& positive_probabilities,
                      int positive_label,
                      int bin_count)
    -> std::pair<Eigen::VectorXd, Eigen::VectorXd> {
    std::vector<double> predicted_sum(static_cast<std::size_t>(bin_count), 0.0);
    std::vector<double> positive_sum(static_cast<std::size_t>(bin_count), 0.0);
    std::vector<int> sample_count(static_cast<std::size_t>(bin_count), 0);

    for (Eigen::Index sample = 0; sample < positive_probabilities.size(); ++sample) {
        const double probability = std::clamp(positive_probabilities(sample), 0.0, 1.0);
        const int bin = std::min(bin_count - 1, static_cast<int>(probability * bin_count));
        predicted_sum[static_cast<std::size_t>(bin)] += probability;
        positive_sum[static_cast<std::size_t>(bin)] += labels(sample) == positive_label ? 1.0 : 0.0;
        ++sample_count[static_cast<std::size_t>(bin)];
    }

    int non_empty_bins = 0;
    for (const int count : sample_count) {
        if (count > 0) ++non_empty_bins;
    }

    Eigen::VectorXd mean_predicted(non_empty_bins);
    Eigen::VectorXd positive_fraction(non_empty_bins);
    int output_index = 0;
    for (int bin = 0; bin < bin_count; ++bin) {
        const int count = sample_count[static_cast<std::size_t>(bin)];
        if (count == 0) continue;
        mean_predicted(output_index) = predicted_sum[static_cast<std::size_t>(bin)] / count;
        positive_fraction(output_index) = positive_sum[static_cast<std::size_t>(bin)] / count;
        ++output_index;
    }

    return {mean_predicted, positive_fraction};
}

} // namespace
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    constexpr int n = 200;
    std::mt19937_64 rng(42);
    std::normal_distribution<double> nz(0.0, 1.0);

    Eigen::MatrixXd X(n, 2);
    Eigen::VectorXi y(n);
    for (int i = 0; i < n; ++i) {
        const double cls = (i < n / 2) ? -1.5 : 1.5;
        X(i, 0) = cls + nz(rng);
        X(i, 1) = cls + nz(rng);
        y(i)    = (cls > 0) ? 1 : 0;
    }

    Skigen::GaussianNB<double> nb;
    Skigen::CalibratedClassifierCV<Skigen::GaussianNB<double>, double> cc(
        nb, Skigen::CalibrationMethod::Sigmoid, /*cv=*/5,
        /*n_jobs=*/1, /*ensemble=*/true,
        std::optional<uint64_t>(42));
    cc.fit(X, y);

    auto pred = cc.predict(X);
    int correct = 0;
    for (int i = 0; i < y.size(); ++i) if (pred(i) == y(i)) ++correct;
    const double acc = static_cast<double>(correct) / y.size();

    const Eigen::MatrixXd probabilities = cc.predict_proba(X);
    Eigen::MatrixXd P_first = probabilities.topRows(3);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== CalibratedClassifierCV (sigmoid, 5-fold) ===\n";
    std::cout << "  base                = GaussianNB\n";
    std::cout << "  n_estimators_fitted = " << cc.n_estimators_fitted() << "\n";
    std::cout << "  n_classes           = " << cc.n_classes() << "\n";
    std::cout << "  classes             = ["
              << cc.classes()(0) << ", " << cc.classes()(1) << "]\n";
    std::cout << "  training accuracy   = " << acc << "\n";
    std::cout << "  first 3 rows predict_proba:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "    row " << i << ": ["
                  << P_first(i, 0) << ", " << P_first(i, 1) << "]\n";
    }

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
    //! [example_calibrated_classifier_cv_reliability_plot]
    const auto [mean_predicted, positive_fraction] =
        calibrationCurve(y, probabilities.col(1), cc.classes()(1), 10);
    Eigen::Vector2d diagonal;
    diagonal << 0.0, 1.0;

    Skigen::Plot::Figure fig;
    fig.title("CalibratedClassifierCV Reliability")
       .caption("Predicted positive-class probabilities compared with observed positive fractions")
       .xlabel("mean predicted probability")
       .ylabel("fraction of positives")
       .plot(diagonal, diagonal, {.lineWidth = 1.2f, .opacity = 0.55f})
       .plot(mean_predicted, positive_fraction, {.lineWidth = 2.6f})
       .scatter(mean_predicted, positive_fraction, {.pointSize = 8.0f, .hollow = true});

    return argc > 1 ? (fig.saveThemed(argv[1]) ? 0 : 1) : fig.show();
    //! [example_calibrated_classifier_cv_reliability_plot]
#else
    return 0;
#endif
}
