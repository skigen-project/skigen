import ThemedImage from '@theme/ThemedImage';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { useState } from 'react';

type ExamplePlotProps = {
    alt: string;
    className?: string;
    stem: string;
};

export default function ExamplePlot({ alt, className, stem }: ExamplePlotProps): React.JSX.Element {
    const [missing, setMissing] = useState(false);
    if (missing) return null;

    return (
        <ThemedImage
            alt={alt}
            className={className}
            sources={{
                light: useBaseUrl(`/img/examples/${stem}_light.png`),
                dark: useBaseUrl(`/img/examples/${stem}_dark.png`),
            }}
            onError={() => setMissing(true)}
        />
    );
}