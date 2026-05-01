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
    ],
    developmentSidebar: [
        'development/building',
        'development/architecture',
        'development/contributing',
        'development/ees-standard',
    ],
};

export default sidebars;
