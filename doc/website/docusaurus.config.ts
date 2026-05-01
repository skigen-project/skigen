import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
    title: 'Skigen',
    tagline: 'Energy-efficient machine learning, native to C++ and Eigen.',
    favicon: 'img/favicon.ico',

    url: 'https://skigen-project.github.io',
    baseUrl: '/',

    organizationName: 'skigen-project',
    projectName: 'skigen-project.github.io',

    onBrokenLinks: 'warn',
    onBrokenMarkdownLinks: 'warn',

    i18n: {
        defaultLocale: 'en',
        locales: ['en'],
    },

    presets: [
        [
            'classic',
            {
                docs: {
                    sidebarPath: './sidebars.ts',
                    remarkPlugins: [remarkMath],
                    rehypePlugins: [rehypeKatex],
                },
                blog: false,
                theme: {
                    customCss: './src/css/custom.css',
                },
            } satisfies Preset.Options,
        ],
    ],

    themes: [
        '@docusaurus/theme-mermaid',
        [
            '@easyops-cn/docusaurus-search-local',
            {
                hashed: true,
                indexDocs: true,
                indexBlog: false,
                docsRouteBasePath: '/docs',
                searchBarShortcutHint: true,
                searchBarPosition: 'right',
            },
        ],
    ],

    markdown: {
        format: 'mdx',
        mermaid: true,
    },

    stylesheets: [
        {
            href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
            type: 'text/css',
            integrity: 'sha384-nB0miv6/jRmo5OUTIL0uadyRNZQKPH0YISY3TDe5Ck5nNBsi5gOF4CgkNzBMR2M',
            crossorigin: 'anonymous',
        },
    ],

    themeConfig: {
        colorMode: {
            defaultMode: 'dark',
            disableSwitch: false,
            respectPrefersColorScheme: true,
        },
        navbar: {
            title: 'Skigen',
            logo: {
                alt: 'Skigen Logo',
                src: 'img/skigen-logo.svg',
            },
            items: [
                {
                    type: 'docSidebar',
                    sidebarId: 'docsSidebar',
                    position: 'left',
                    label: 'Documentation',
                },
                {
                    type: 'docSidebar',
                    sidebarId: 'developmentSidebar',
                    position: 'left',
                    label: 'Development',
                },
                {
                    href: 'https://github.com/skigen-project/skigen',
                    position: 'right',
                    className: 'header-github-link',
                    'aria-label': 'GitHub repository',
                },
            ],
        },
        footer: {
            style: 'dark',
            links: [
                {
                    title: 'Documentation',
                    items: [
                        { label: 'Getting Started', to: '/docs/getting-started' },
                        { label: 'API Guide', to: '/docs/guide/standard-scaler' },
                    ],
                },
                {
                    title: 'Development',
                    items: [
                        { label: 'Build from Source', to: '/docs/development/building' },
                        { label: 'Contributing', to: '/docs/development/contributing' },
                    ],
                },
                {
                    title: 'Community',
                    items: [
                        { label: 'GitHub', href: 'https://github.com/skigen-project/skigen' },
                    ],
                },
            ],
            copyright: `Copyright © ${new Date().getFullYear()} Skigen Contributors.`,
        },
        prism: {
            theme: prismThemes.github,
            darkTheme: prismThemes.dracula,
            additionalLanguages: ['cpp', 'cmake', 'bash'],
        },
    } satisfies Preset.ThemeConfig,
};

export default config;
