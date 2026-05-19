#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Skigen Contributors
#
# build-docs.sh — Generate API reference Markdown for Docusaurus.
# Usage: cd doc && bash build-docs.sh
#
# Requires:
#   - doxygen  (https://www.doxygen.nl/)
#   - python3

set -euo pipefail
cd "$(dirname "$0")"

SKIGENPLOT_DIR="${SKIGENPLOT_DIR:-$(pwd)/../../skigen-plot}"

echo "=== [1/3] Running Doxygen (Skigen) ==="
doxygen Doxyfile

echo "=== [2/3] Running doxygen2mdx.py (Skigen) ==="
python3 doxygen2mdx.py \
    --xml-dir doxygen_output/xml \
    --out-dir website/docs/api \
    --generate-sidebars

if [ -d "$SKIGENPLOT_DIR/doc" ]; then
    echo "=== [3/3] Processing SkigenPlot docs ==="

    # Run Doxygen on skigen-plot headers
    (cd "$SKIGENPLOT_DIR/doc" && doxygen Doxyfile)

    # Generate API reference MDX from skigen-plot Doxygen XML
    python3 doxygen2mdx.py \
        --xml-dir "$SKIGENPLOT_DIR/doc/doxygen_output/xml" \
        --out-dir website/docs/plot/api \
        --registry "$SKIGENPLOT_DIR/doc/api_registry.json"

    # Copy hand-written guide pages into the website
    mkdir -p website/docs/plot/guide
    cp -r "$SKIGENPLOT_DIR/doc/guide/"*.mdx website/docs/plot/guide/ 2>/dev/null || true

    echo "  Plot API:    website/docs/plot/api/"
    echo "  Plot guides: website/docs/plot/guide/"
else
    echo "=== [3/3] SkigenPlot not found at $SKIGENPLOT_DIR — skipping ==="
fi

echo "=== Done ==="
echo "Run 'cd website && npm run build' to build the full site."
