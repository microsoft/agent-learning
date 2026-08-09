#!/usr/bin/env bash
# scripts/publish.sh
#
# Build and publish agent-learning to PyPI / TestPyPI.
#
# Usage:
#   ./scripts/publish.sh                       # build only (no upload)
#   ./scripts/publish.sh --target testpypi     # upload to TestPyPI
#   ./scripts/publish.sh --target pypi         # upload to real PyPI
#   ./scripts/publish.sh --target testpypi --skip-tests --no-clean
#
# Credentials (use ONE of):
#   - Env var TWINE_PASSWORD set to your API token (TWINE_USERNAME defaults to __token__).
#   - ~/.pypirc with [pypi] / [testpypi] sections.
#   - For CI, prefer PyPI Trusted Publishers (see .github/workflows/publish.yaml).
#
# IMPORTANT: PyPI may reserve the "azure-*" namespace for Microsoft. Verify
# you can register the project name (try TestPyPI first) before pushing to
# production PyPI.

set -euo pipefail

TARGET="none"
SKIP_TESTS=0
NO_CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)        TARGET="$2"; shift 2 ;;
        --target=*)      TARGET="${1#*=}"; shift ;;
        --skip-tests)    SKIP_TESTS=1; shift ;;
        --no-clean)      NO_CLEAN=1; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$TARGET" in
    none|testpypi|pypi) ;;
    *) echo "Invalid --target '$TARGET' (use: none|testpypi|pypi)" >&2; exit 2 ;;
esac

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SDK_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

echo "=== agent-learning publish ==="
echo "SDK root:   ${SDK_ROOT}"
echo "Target:     ${TARGET}"
echo "Skip tests: ${SKIP_TESTS}"
echo "No clean:   ${NO_CLEAN}"
echo ""

cd "${SDK_ROOT}"

# Pick a python interpreter
if command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "❌ No python interpreter found on PATH." >&2
    exit 1
fi
echo "Using python: ${PYTHON}"

# 1. Clean
if [[ "${NO_CLEAN}" -eq 0 ]]; then
    echo ""
    echo "[1/5] Cleaning dist/, build/, *.egg-info ..."
    rm -rf dist build
    find src -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
fi

# 2. Tooling
echo ""
echo "[2/5] Ensuring build + twine are installed ..."
"${PYTHON}" -m pip install --upgrade pip build twine

# 3. Tests
if [[ "${SKIP_TESTS}" -eq 0 ]]; then
    echo ""
    echo "[3/5] Running pytest ..."
    "${PYTHON}" -m pip install --quiet -e ".[dev]"
    "${PYTHON}" -m pytest tests
else
    echo ""
    echo "[3/5] Skipping tests (--skip-tests)."
fi

# 4. Build
echo ""
echo "[4/5] Building distributions ..."
"${PYTHON}" -m build

echo ""
echo "Artifacts:"
ls -lh dist/ | awk 'NR>1 {print "  " $9 "  (" $5 ")"}'

echo ""
echo "Validating metadata with twine check ..."
"${PYTHON}" -m twine check dist/*

# 5. Upload
echo ""
case "${TARGET}" in
    none)
        echo "[5/5] Build complete. No upload requested."
        echo "    Re-run with --target testpypi or --target pypi to publish."
        ;;
    testpypi)
        echo "[5/5] Uploading to TestPyPI ..."
        export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"
        if [[ -z "${TWINE_PASSWORD:-}" ]]; then
            echo "TWINE_PASSWORD not set; twine will fall back to ~/.pypirc."
        fi
        "${PYTHON}" -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
        echo ""
        echo "Published. View at:"
        echo "  https://test.pypi.org/project/agents-learning-sdk/"
        ;;
    pypi)
        echo "[5/5] Uploading to PyPI (production) ..."
        export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"
        if [[ -z "${TWINE_PASSWORD:-}" ]]; then
            echo "TWINE_PASSWORD not set; twine will fall back to ~/.pypirc."
        fi
        "${PYTHON}" -m twine upload dist/*
        echo ""
        echo "Published. View at:"
        echo "  https://pypi.org/project/agents-learning-sdk/"
        ;;
esac
