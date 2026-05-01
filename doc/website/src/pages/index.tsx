import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const features = [
    {
        title: 'Eigen-Native',
        description:
            'Built entirely on Eigen\'s expression templates. No wrapper overhead — just vectorized math from start to finish.',
    },
    {
        title: 'scikit-learn API',
        description:
            'The fit / transform / predict paradigm you already know. Porting logic from Python to C++ is straightforward.',
    },
    {
        title: 'Energy-Efficient',
        description:
            'Header-only, zero-copy, zero interpreter tax. Every cycle goes to computation, not infrastructure.',
    },
];

function Feature({ title, description }: { title: string; description: string }) {
    return (
        <div className={clsx('col col--4')}>
            <div className="text--center padding-horiz--md padding-vert--lg">
                <Heading as="h3">{title}</Heading>
                <p>{description}</p>
            </div>
        </div>
    );
}

function HomepageHeader() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <header className={clsx('hero hero--primary', styles.heroBanner)}>
            <div className="container">
                <Heading as="h1" className="hero__title">
                    {siteConfig.title}
                </Heading>
                <p className="hero__subtitle">{siteConfig.tagline}</p>
                <div className={styles.buttons}>
                    <Link
                        className="button button--secondary button--lg"
                        to="/docs/getting-started">
                        Get Started
                    </Link>
                </div>
            </div>
        </header>
    );
}

export default function Home(): React.JSX.Element {
    const { siteConfig } = useDocusaurusContext();
    return (
        <Layout title={siteConfig.title} description={siteConfig.tagline}>
            <HomepageHeader />
            <main>
                <section className={styles.features}>
                    <div className="container">
                        <div className="row">
                            {features.map((props, idx) => (
                                <Feature key={idx} {...props} />
                            ))}
                        </div>
                    </div>
                </section>
            </main>
        </Layout>
    );
}
