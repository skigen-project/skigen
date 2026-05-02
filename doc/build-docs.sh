#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Skigen Contributors
#
# build-docs.sh — Generate API reference Markdown for Docusaurus.
# Usage: cd doc && bash build-docs.sh
#
# Requires:
#   - doxygen  (https://www.doxygen.nl/)
#   - doxybook2 (https://github.com/matusnovak/doxybook2)

set -euo pipefail
cd "$(dirname "$0")"

echo "=== [1/3] Running Doxygen ==="
doxygen Doxyfile

echo "=== [2/3] Running Doxybook2 ==="
# Clean previous generated API docs to avoid stale pages
rm -rf website/docs/api
mkdir -p website/docs/api

doxybook2 \
    --input doxygen_output/xml \
    --output website/docs/api \
    --config doxybook2.json

echo "=== [3/3] Fixing MDX compatibility ==="
# Doxybook2 emits HTML tags and C++ syntax that conflict with Docusaurus MDX.
# 1) Self-close void HTML elements for JSX compatibility.
# 2) Add 'format: md' to frontmatter to force CommonMark parsing, preventing
#    C++ angle brackets and braces from being interpreted as JSX/expressions.
find website/docs/api -name '*.md' -exec sed -i.bak \
    -e 's/<br>/<br \/>/g' \
    -e 's/<hr>/<hr \/>/g' \
    -e '1s/^---$/---\nformat: md/' \
    {} +
find website/docs/api -name '*.md.bak' -delete

echo "=== Done ==="
echo "Output: website/docs/api/"
echo "Run 'cd website && npm run build' to build the full site."
