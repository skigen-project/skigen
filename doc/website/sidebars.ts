import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
    docsSidebar: [
        'overview',
        'getting-started',
        {
            type: 'category',
            label: 'Guide',
            items: [
                'guide/standard-scaler',
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
