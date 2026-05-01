import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const features = [
    {
        title: 'Eigen-Native',
        icon: '⚡',
        description:
            "Built entirely on Eigen's expression templates. No wrapper overhead — just vectorized math from start to finish.",
    },
    {
        title: 'scikit-learn API',
        icon: '🔬',
        description:
            'The fit / transform / predict paradigm you already know. Porting logic from Python to C++ is straightforward.',
    },
    {
        title: 'Energy-Efficient',
        icon: '🌱',
        description:
            'Header-only, zero-copy, zero interpreter tax. Every cycle goes to computation, not infrastructure.',
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

export default function Home(): React.JSX.Element {
    const { siteConfig } = useDocusaurusContext();
    return (
        <Layout title="Home" description={siteConfig.tagline}>
            <header className={styles.hero}>
                <HexGrid />
                <div className={styles.heroInner}>
                    <img src="img/skigen-logo.png" alt="Skigen" className={styles.heroLogo} />
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
                                        <span className={styles.featureIcon}>{f.icon}</span>
                                        <Heading as="h3" className={styles.featureTitle}>{f.title}</Heading>
                                        <p className={styles.featureDesc}>{f.description}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Code example */}
                <section className={styles.codeSection}>
                    <div className="container">
                        <div className="row">
                            <div className="col col--5">
                                <Heading as="h2" className={styles.codeSectionTitle}>
                                    Familiar API, native performance
                                </Heading>
                                <p className={styles.codeSectionDesc}>
                                    If you know scikit-learn, you already know Skigen.
                                    The same <code>fit</code> / <code>transform</code> / <code>predict</code> workflow,
                                    compiled to optimized machine code with zero runtime overhead.
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
                                <span className={styles.statNumber}>3</span>
                                <span className={styles.statLabel}>Platform targets</span>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
        </Layout>
    );
}
