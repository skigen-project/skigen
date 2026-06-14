// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// kmeans.cpp — KMeans and MiniBatchKMeans clustering.
//
// Build modes
//   ex_kmeans (default)            — minimal headless run; prints results.
//   ex_kmeans (with SKIGEN_WITH_PLOT=ON)
//     • run with no args           — opens an interactive plot window
//                                    (theme toggle in the overlay).
//     • run with an output stem    — renders <stem>_dark.png and
//                                    <stem>_light.png and exits.

#include <Skigen/Cluster>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
#include <skigen/plot/figure.h>
#endif

namespace {

auto makeClusters() -> Eigen::MatrixXd {
    constexpr int n_per = 60;
    constexpr int n = n_per * 3;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);
    const double cx[3] = {-4.0, 4.0, 0.0};
    const double cy[3] = { 0.0, 0.0, 5.0};
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < n_per; ++i) {
            X(c * n_per + i, 0) = cx[c] + noise(rng);
            X(c * n_per + i, 1) = cy[c] + noise(rng);
        }
    }
    return X;
}

} // namespace

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    const Eigen::MatrixXd X = makeClusters();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Data: " << X.rows() << " samples, 2 features, 3 clusters\n\n";

    //! [example_kmeans]
    // KMeans
    Skigen::KMeans<double> km(3, /*max_iter=*/300, /*n_init=*/10, /*random_state=*/42);
    km.fit(X);

    std::cout << "=== KMeans (k=3) ===\n";
    std::cout << "Inertia:    " << km.inertia() << "\n";
    std::cout << "Iterations: " << km.n_iter() << "\n";
    std::cout << "Centers:\n" << km.cluster_centers() << "\n\n";

    // Predict on new points
    Eigen::MatrixXd X_new(3, 2);
    X_new << -4.0, 0.0,
              4.0, 0.0,
              0.0, 5.0;
    auto labels = km.predict(X_new);
    std::cout << "New point labels: " << labels.transpose() << "\n\n";
    //! [example_kmeans]

    //! [example_mini_batch_kmeans]
    // MiniBatchKMeans — faster for large datasets
    Skigen::MiniBatchKMeans<double> mbk(3, /*batch_size=*/30, /*max_iter=*/100, /*random_state=*/42);
    mbk.fit(X);

    std::cout << "=== MiniBatchKMeans (k=3, batch=30) ===\n";
    std::cout << "Inertia:    " << mbk.inertia() << "\n";
    std::cout << "Centers:\n" << mbk.cluster_centers() << "\n";
    //! [example_mini_batch_kmeans]

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
    //! [example_kmeans_plot]
    Skigen::Plot::Figure fig;
    fig.title("KMeans Clustering")
        .caption("Three Gaussian clusters grouped by Skigen::KMeans, visualized with SkigenPlot")
       .xlabel("feature 1")
       .ylabel("feature 2")
       .scatter(X, km.predict(X))
       .scatter(km.cluster_centers(), km.predict(km.cluster_centers()),
                {.pointSize = 18.0f, .hollow = true});

    return argc > 1 ? (fig.saveThemed(argv[1]) ? 0 : 1) : fig.show();
    //! [example_kmeans_plot]
#else
    return 0;
#endif
}
