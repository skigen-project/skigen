// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// kmeans.cpp — KMeans and MiniBatchKMeans clustering.
//
// Build modes
//   ex_kmeans       — minimal headless run; prints inertia/centers/labels.
//   ex_kmeans_plot  — same algorithmic code plus a SkigenPlot scatter
//                     rendering (dark + light PNG). Built only when
//                     SKIGEN_WITH_PLOT=ON; gated below with
//                     SKIGEN_EXAMPLE_WITH_PLOT.
//
//   Usage (plot variant): ex_kmeans_plot [output_directory]

#include <Skigen/Cluster>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <random>

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
#include <skigen/plot/plotview.h>
#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QTimer>
#include <functional>
#include <memory>
#include <vector>
#endif

namespace {

struct Dataset {
    Eigen::MatrixXd X;
    int n_per;
};

auto makeClusters() -> Dataset {
    constexpr int n_per = 60;
    constexpr int n = n_per * 3;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    Eigen::MatrixXd X(n, 2);
    // Cluster A: centered at (-4, 0)
    for (int i = 0; i < n_per; ++i) {
        X(i, 0) = -4.0 + noise(rng);
        X(i, 1) = 0.0 + noise(rng);
    }
    // Cluster B: centered at (4, 0)
    for (int i = 0; i < n_per; ++i) {
        X(n_per + i, 0) = 4.0 + noise(rng);
        X(n_per + i, 1) = 0.0 + noise(rng);
    }
    // Cluster C: centered at (0, 5)
    for (int i = 0; i < n_per; ++i) {
        X(2 * n_per + i, 0) = 0.0 + noise(rng);
        X(2 * n_per + i, 1) = 5.0 + noise(rng);
    }
    return {std::move(X), n_per};
}

} // namespace

#ifdef SKIGEN_EXAMPLE_WITH_PLOT
namespace {

void plotClusters(Skigen::Plot::PlotView& view,
                  const Eigen::MatrixXd& X,
                  const Eigen::VectorXi& labels,
                  const Eigen::MatrixXd& centers,
                  const Skigen::Plot::Theme& theme) {
    view.clear();
    view.setTheme(theme);
    view.setTitle(QStringLiteral("KMeans Clustering"));
    view.setCaption(QStringLiteral(
        "Three Gaussian clusters recovered by Skigen::KMeans, visualized with SkigenPlot"));
    view.setAxisLabels(QStringLiteral("feature 1"), QStringLiteral("feature 2"));

    // One scatter series per cluster — PlotView assigns palette colors per
    // series so the recovered clusters come out in distinct theme colors.
    const int k = static_cast<int>(centers.rows());
    for (int c = 0; c < k; ++c) {
        std::vector<float> xs, ys;
        xs.reserve(X.rows());
        ys.reserve(X.rows());
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            if (labels(i) == c) {
                xs.push_back(static_cast<float>(X(i, 0)));
                ys.push_back(static_cast<float>(X(i, 1)));
            }
        }
        Eigen::Map<const Eigen::VectorXf> x(xs.data(),
            static_cast<Eigen::Index>(xs.size()));
        Eigen::Map<const Eigen::VectorXf> y(ys.data(),
            static_cast<Eigen::Index>(ys.size()));
        view.scatter(x, y, {
            .pointSize = 7.0f,
            .label = QStringLiteral("cluster %1").arg(c + 1),
        });
    }

    // Cluster centers as large markers in the theme text color.
    Eigen::VectorXf cx = centers.col(0).cast<float>();
    Eigen::VectorXf cy = centers.col(1).cast<float>();
    view.scatter(cx, cy, {
        .color = theme.textColor,
        .pointSize = 16.0f,
        .label = QStringLiteral("centers"),
    });
}

int renderPlots(int argc, char* argv[],
                const Eigen::MatrixXd& X,
                const Eigen::VectorXi& labels,
                const Eigen::MatrixXd& centers) {
    QApplication app(argc, argv);

    // When an output directory is given render both themes to PNG and exit.
    // When run interactively (no argument) keep the window open so the user
    // can inspect the plot; press Ctrl+C or close the window to quit.
    const bool interactive = (argc < 2);
    const QString outDir = interactive ? QString() : QString::fromLocal8Bit(argv[1]);
    if (!outDir.isEmpty())
        QDir().mkpath(outDir);

    struct Frame {
        QString filename;
        Skigen::Plot::Theme theme;
    };
    const std::vector<Frame> frames = {
        {QStringLiteral("kmeans_dark.png"),  Skigen::Plot::Theme::dark()},
        {QStringLiteral("kmeans_light.png"), Skigen::Plot::Theme::light()},
    };

    Skigen::Plot::PlotView view;
    view.setOverlayVisible(interactive);
    view.resize(1200, 800);
    view.setWindowTitle(QStringLiteral("KMeans Clustering — SkigenPlot"));
    plotClusters(view, X, labels, centers, frames.front().theme);
    view.show();

    if (interactive)
        return app.exec();

    auto index = std::make_shared<std::size_t>(0);
    auto runNext = std::make_shared<std::function<void()>>();
    *runNext = [&]() {
        if (*index >= frames.size()) {
            app.quit();
            return;
        }
        const Frame& frame = frames[*index];
        plotClusters(view, X, labels, centers, frame.theme);
        view.update();

        QTimer::singleShot(200, &view, [&, frame]() {
            const QString path = outDir + QDir::separator() + frame.filename;
            if (!view.savePng(path, 1200, 800))
                qWarning() << "Failed to render" << path;
            ++(*index);
            (*runNext)();
        });
    };

    QTimer::singleShot(200, &view, [&]() { (*runNext)(); });
    return app.exec();
}

} // namespace
#endif // SKIGEN_EXAMPLE_WITH_PLOT

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    const auto data = makeClusters();
    const Eigen::MatrixXd& X = data.X;

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
    const auto train_labels = km.predict(X);
    return renderPlots(argc, argv, X, train_labels, km.cluster_centers());
#else
    return 0;
#endif
}
