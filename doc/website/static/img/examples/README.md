# Rendered example plots

The `*_dark.png` / `*_light.png` files in this directory are **generated** from
the plot-enabled example programs (SkigenPlot / Qt-RHI), not hand-authored.
Do not edit them by hand.

## Regenerating

On a machine with a working SkigenPlot (Qt 6 + RHI) runtime:

```bash
cmake -B build-plot -DSKIGEN_BUILD_EXAMPLES=ON -DSKIGEN_WITH_PLOT=ON
cmake --build build-plot --target skigen_render_example_plots
```

Each example registered via `skigen_enable_plot(<target> <stem>)` in
`examples/CMakeLists.txt` writes `<stem>_dark.png` and `<stem>_light.png`
here. Currently registered: `kmeans` (more examples gain plots over time).

## Referencing from a guide page

Use Docusaurus `ThemedImage` so the figure follows the site's light/dark mode:

```mdx
import ThemedImage from '@theme/ThemedImage';

<ThemedImage
  alt="KMeans clusters rendered with SkigenPlot"
  sources={{
    light: '/img/examples/kmeans_light.png',
    dark: '/img/examples/kmeans_dark.png',
  }}
/>
```

Add the reference to a guide page only once the corresponding PNGs have been
generated and committed, so the published site never links a missing image.
