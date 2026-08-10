#!/usr/bin/env bash
set -euo pipefail

APP_VERSION="dev"
SKIP_TESTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app-version)
            APP_VERSION="$2"
            shift 2
            ;;
        --app-version=*)
            APP_VERSION="${1#*=}"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=1
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./scripts/build_linux_installer.sh [--app-version VERSION] [--skip-tests]

Build a standalone Linux agent-learn binary and archive it for release.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== agent-learning Linux installer build ==="
echo "SDK root:    ${SDK_ROOT}"
echo "App version: ${APP_VERSION}"
echo "Skip tests:  ${SKIP_TESTS}"

cd "${SDK_ROOT}"

python -m pip install --upgrade pip

if [[ "${SKIP_TESTS}" -eq 0 ]]; then
    python -m pip install -e ".[dev]" pyinstaller
    python -m pytest -q
else
    python -m pip install pyinstaller
fi

rm -rf build dist dist-installer
mkdir -p dist-installer

python -m PyInstaller \
    --clean \
    --noconfirm \
    --onefile \
    --name "agent-learn" \
    "packaging/linux/agent_learning_entry.py"

chmod +x dist/agent-learn

tar -czf "dist-installer/agent-learning-cli-${APP_VERSION}-linux-x64.tar.gz" -C dist agent-learn

echo "Built installer artifacts in dist-installer/"
