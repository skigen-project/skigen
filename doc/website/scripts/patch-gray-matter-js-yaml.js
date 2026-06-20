// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Christoph Dinh <christoph.dinh@mne-cpp.org>

const fs = require('node:fs');
const path = require('node:path');

const enginePath = path.join(__dirname, '..', 'node_modules', 'gray-matter', 'lib', 'engines.js');

if (!fs.existsSync(enginePath)) {
  process.exit(0);
}

let source = fs.readFileSync(enginePath, 'utf8');
source = source.replace('parse: yaml.safeLoad.bind(yaml),', 'parse: yaml.load.bind(yaml),');
source = source.replace('stringify: yaml.safeDump.bind(yaml)', 'stringify: yaml.dump.bind(yaml)');
fs.writeFileSync(enginePath, source);
