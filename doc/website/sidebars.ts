import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
    docsSidebar: [
        'overview',
        'getting-started',
        {
            type: 'category',
            label: 'Preprocessing',
            items: [
                'guide/standard-scaler',
                'guide/minmax-scaler',
                'guide/robust-scaler',
                'guide/maxabs-scaler',
                'guide/normalizer',
                'guide/polynomial-features',
                'guide/label-encoder',
            ],
        },
        {
            type: 'category',
            label: 'Linear Models',
            items: [
                'guide/linear-regression',
                'guide/logistic-regression',
                'guide/ridge',
                'guide/lasso',
                'guide/elastic-net',
                'guide/sgd',
                'guide/bayesian-ridge',
                'guide/ard-regression',
            ],
        },
        {
            type: 'category',
            label: 'Decomposition',
            items: [
                'guide/pca',
                'guide/truncated-svd',
                'guide/factor-analysis',
            ],
        },
        {
            type: 'category',
            label: 'Clustering',
            items: [
                'guide/kmeans',
            ],
        },
        {
            type: 'category',
            label: 'Tree & Neighbors',
            items: [
                'guide/decision-tree',
                'guide/kneighbors',
                'guide/local-outlier-factor',
            ],
        },
        {
            type: 'category',
            label: 'Ensembles',
            items: [
                'guide/random-forest-classifier',
                'guide/random-forest-regressor',
                'guide/gradient-boosting-classifier',
                'guide/gradient-boosting-regressor',
                'guide/hist-gradient-boosting-classifier',
                'guide/hist-gradient-boosting-regressor',
            ],
        },
        {
            type: 'category',
            label: 'Naive Bayes',
            items: [
                'guide/gaussian-nb',
                'guide/multinomial-nb',
                'guide/bernoulli-nb',
            ],
        },
        {
            type: 'category',
            label: 'Support Vector Machines',
            items: [
                'guide/linear-svc',
                'guide/linear-svr',
                'guide/svc',
                'guide/svr',
                'guide/nu-svc',
                'guide/nu-svr',
                'guide/one-class-svm',
            ],
        },
        {
            type: 'category',
            label: 'Neural Networks',
            items: [
                'guide/mlp-classifier',
                'guide/mlp-regressor',
            ],
        },
        {
            type: 'category',
            label: 'Feature Selection',
            items: [
                'guide/variance-threshold',
                'guide/select-k-best',
                'guide/select-from-model',
                'guide/rfe',
            ],
        },
        {
            type: 'category',
            label: 'Calibration & Isotonic',
            items: [
                'guide/calibrated-classifier-cv',
                'guide/isotonic-regression',
            ],
        },
        {
            type: 'category',
            label: 'Covariance',
            items: [
                'guide/empirical-covariance',
                'guide/ledoit-wolf',
                'guide/oas',
            ],
        },
        {
            type: 'category',
            label: 'Manifold Learning',
            items: [
                'guide/isomap',
                'guide/mds',
                'guide/locally-linear-embedding',
                'guide/spectral-embedding',
                'guide/tsne',
                'guide/umap',
            ],
        },
        {
            type: 'category',
            label: 'Pipeline & Utilities',
            items: [
                'guide/pipeline',
                'guide/model-selection',
                'guide/grid-search-cv',
                'guide/randomized-search-cv',
                'guide/metrics',
            ],
        },
        'cite',
    ],
    apiSidebar: [
        {
            type: 'category',
            label: 'Skigen::LinearModel',
            items: [
                'api/linear-model/elastic-net',
                'api/linear-model/ridge',
                'api/linear-model/lasso',
                'api/linear-model/linear-regression',
                'api/linear-model/logistic-regression',
                'api/linear-model/sgdclassifier',
                'api/linear-model/sgdregressor',
            ],
        },
        {
            type: 'category',
            label: 'Skigen::Preprocessing',
            items: [
                'api/preprocessing/standard-scaler',
                'api/preprocessing/min-max-scaler',
                'api/preprocessing/max-abs-scaler',
                'api/preprocessing/robust-scaler',
                'api/preprocessing/normalizer',
                'api/preprocessing/polynomial-features',
                'api/preprocessing/label-encoder',
            ],
        },
        {
            type: 'category',
            label: 'Skigen::Decomposition',
            items: [
                'api/decomposition/pca',
                'api/decomposition/truncated-svd',
            ],
        },
        {
            type: 'category',
            label: 'Skigen::Cluster',
            items: [
                'api/cluster/kmeans',
                'api/cluster/mini-batch-kmeans',
            ],
        },
        {
            type: 'category',
            label: 'Skigen::Tree',
            items: [
                'api/tree/decision-tree-classifier',
                'api/tree/decision-tree-regressor',
            ],
        },
        {
            type: 'category',
            label: 'Skigen::Neighbors',
            items: [
                'api/neighbors/kneighbors-classifier',
                'api/neighbors/kneighbors-regressor',
            ],
        },
        {
            type: 'category',
            label: 'Skigen::Pipeline',
            items: [
                'api/pipeline/pipeline',
            ],
        },
    ],
    developmentSidebar: [
        'development/building',
        'development/architecture',
        'development/contributing',
        'development/ees-standard',
    ],
};

export default sidebars;
