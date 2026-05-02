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
            ],
        },
        {
            type: 'category',
            label: 'Decomposition',
            items: [
                'guide/pca',
                'guide/truncated-svd',
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
            ],
        },
        {
            type: 'category',
            label: 'Pipeline & Utilities',
            items: [
                'guide/pipeline',
                'guide/model-selection',
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
