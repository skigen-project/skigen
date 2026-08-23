#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_FILE="${SCRIPT_DIR}/../../config/external_deps.env"
QT_VERSION="$(sed -n 's/^QT_VERSION=//p' "${DEPS_FILE}" | tr -d '\r')"

if [[ -z "${QT_VERSION}" ]]; then
    echo "QT_VERSION is not defined in ${DEPS_FILE}" >&2
    exit 1
fi

emit()
{
    local name="$1"
    local value="$2"
    echo "${name}=${value}"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        printf '%s=%s\n' "${name}" "${value}" >> "${GITHUB_ENV}"
    fi
}

emit QT_VERSION "${QT_VERSION}"
emit QT_VERSION_TOKEN "${QT_VERSION//./}"
