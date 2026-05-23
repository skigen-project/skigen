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

echo "=== [1/2] Running Doxygen (Skigen) ==="
doxygen Doxyfile

echo "=== [2/2] Running doxygen2mdx.py (Skigen) ==="
python3 doxygen2mdx.py \
    --xml-dir doxygen_output/xml \
    --out-dir website/docs/api \
    --generate-sidebars

echo "=== Done ==="
echo "Run 'cd website && npm run build' to build the full site."
