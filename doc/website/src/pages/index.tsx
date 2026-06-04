import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import ThemedImage from '@theme/ThemedImage';
import { useColorMode } from '@docusaurus/theme-common';
import { useState } from 'react';
import styles from './index.module.css';

const features = [
    {
        title: 'Versatile',
        description:
            'Preprocessing, linear models, decomposition, clustering, trees, neighbors, pipelines, and metrics — all with a consistent fit / transform / predict API.',
    },
    {
        title: 'Fast',
        description:
            "Eigen's expression templates, explicit SIMD vectorization, and compile-time polymorphism. No interpreter, no GC, no runtime dispatch.",
    },
    {
        title: 'Elegant',
        description:
            'Header-only — just drop Skigen/ next to Eigen/ and #include. The same API you know from scikit-learn, native to C++.',
    },
];

const codeExample = `#include <Skigen/Dense>

int main() {
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 4);
    Eigen::VectorXd y = X.col(0) * 2.0 + X.col(1) * 0.5;

    // scikit-learn-style pipeline
    Skigen::StandardScaler scaler;
    scaler.fit(X);
    auto X_scaled = scaler.transform(X);

    Skigen::LinearRegression model;
    model.fit(X_scaled, y);
    auto predictions = model.predict(X_scaled);
}`;

function HexGrid() {
    return (
        <div className={styles.hexGrid} aria-hidden="true">
            <svg className={styles.hexSvg} viewBox="0 0 1440 500" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="gridGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.08" />
                        <stop offset="100%" stopColor="#a855f7" stopOpacity="0.08" />
                    </linearGradient>
                </defs>
                {/* Subtle hexagonal grid pattern */}
                <pattern id="hexPattern" x="0" y="0" width="60" height="52" patternUnits="userSpaceOnUse" patternTransform="rotate(0)">
                    <path d="M30,0 L60,15 L60,37 L30,52 L0,37 L0,15 Z" fill="none" stroke="url(#gridGrad)" strokeWidth="0.5" />
                </pattern>
                <rect width="100%" height="100%" fill="url(#hexPattern)" className={styles.hexPatternRect} />
                {/* Floating glow orbs */}
                <circle cx="200" cy="150" r="120" fill="url(#gridGrad)" opacity="0.3" className={styles.orb1} />
                <circle cx="1200" cy="350" r="160" fill="url(#gridGrad)" opacity="0.2" className={styles.orb2} />
            </svg>
        </div>
    );
}

function HeroLogo() {
    const { colorMode } = useColorMode();
    const logoSrc = colorMode === 'dark' ? 'img/skigen-logo-dark.svg' : 'img/skigen-logo.svg';
    return <img src={logoSrc} alt="Skigen" className={styles.heroLogo} />;
}

// Plot of the KMeans example (examples/cluster/kmeans.cpp built with
// SKIGEN_WITH_PLOT=ON). The PNGs are rendered fresh by CI on every docs
// deploy and are never committed; if rendering was skipped the image
// silently hides itself so the page still renders cleanly.
function ShowcasePlot() {
    const dark = useBaseUrl('img/plots/kmeans_dark.png');
    const light = useBaseUrl('img/plots/kmeans_light.png');
    const [missing, setMissing] = useState(false);
    if (missing) return null;
    return (
        <ThemedImage
            className={styles.showcaseImage}
            alt="KMeans clusters visualized with SkigenPlot"
            sources={{ light, dark }}
            onError={() => setMissing(true)}
        />
    );
}

export default function Home(): React.JSX.Element {
    const { siteConfig } = useDocusaurusContext();
    return (
        <Layout title="Home" description={siteConfig.tagline}>
            <header className={styles.hero}>
                <HexGrid />
                <div className={styles.heroInner}>
                    <HeroLogo />
                    <p className={styles.heroTagline}>{siteConfig.tagline}</p>
                    <div className={styles.heroCta}>
                        <Link className="button button--primary button--lg" to="/docs/getting-started">
                            Get Started
                        </Link>
                        <Link className={clsx('button button--lg', styles.btnOutline)} to="https://github.com/skigen-project/skigen">
                            View on GitHub
                        </Link>
                    </div>
                </div>
            </header>

            <main>
                {/* Features */}
                <section className={styles.features}>
                    <div className="container">
                        <div className="row">
                            {features.map((f, idx) => (
                                <div key={idx} className="col col--4">
                                    <div className={styles.featureCard}>
                                        <Heading as="h3" className={styles.featureTitle}>{f.title}</Heading>
                                        <p className={styles.featureDesc}>{f.description}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Showcase: KMeans rendered with SkigenPlot */}
                <section className={styles.showcase}>
                    <div className="container">
                        <div className={styles.showcaseInner}>
                            <Heading as="h2" className={styles.showcaseTitle}>
                                See your results, instantly
                            </Heading>
                            <p className={styles.showcaseDesc}>
                                Pair Skigen with the optional <Link to="https://github.com/skigen-project/skigen-plot">SkigenPlot</Link>{' '}
                                library for hardware-accelerated visualization — straight from{' '}
                                <code>Eigen</code> matrices, no Python round-trip. See the{' '}
                                <Link to="https://skigen-project.github.io/skigen-plot/">SkigenPlot docs</Link>{' '}
                                for details. The image below is rendered on every docs deploy.
                            </p>
                            <ShowcasePlot />
                        </div>
                    </div>
                </section>

                {/* Code example */}
                <section className={styles.codeSection}>
                    <div className="container">
                        <div className="row">
                            <div className="col col--5">
                                <Heading as="h2" className={styles.codeSectionTitle}>
                                    scikit-learn for C++
                                </Heading>
                                <p className={styles.codeSectionDesc}>
                                    The same <code>fit</code> / <code>transform</code> / <code>predict</code> workflow
                                    you know from Python — compiled directly to vectorized machine code via Eigen.
                                </p>
                                <Link className="button button--primary" to="/docs/guide/standard-scaler">
                                    Explore the API →
                                </Link>
                            </div>
                            <div className="col col--7">
                                <div className={styles.codeBlock}>
                                    <div className={styles.codeHeader}>
                                        <span className={styles.codeDot} style={{ background: '#ff5f57' }} />
                                        <span className={styles.codeDot} style={{ background: '#febc2e' }} />
                                        <span className={styles.codeDot} style={{ background: '#28c840' }} />
                                        <span className={styles.codeFilename}>pipeline.cpp</span>
                                    </div>
                                    <pre className={styles.codePre}><code>{codeExample}</code></pre>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Stats */}
                <section className={styles.stats}>
                    <div className="container">
                        <div className={styles.statsGrid}>
                            <div className={styles.statItem}>
                                <span className={styles.statNumber}>0</span>
                                <span className={styles.statLabel}>Runtime dependencies</span>
                            </div>
                            <div className={styles.statItem}>
                                <span className={styles.statNumber}>C++23</span>
                                <span className={styles.statLabel}>Modern standard</span>
                            </div>
                            <div className={styles.statItem}>
                                <span className={styles.statNumber}>Header-only</span>
                                <span className={styles.statLabel}>Just #include</span>
                            </div>
                            <div className={styles.statItem}>
                                <span className={styles.statNumber}>scikit-learn</span>
                                <span className={styles.statLabel}>API compatible</span>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
        </Layout>
    );
}
