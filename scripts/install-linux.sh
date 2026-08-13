#!/usr/bin/env bash
set -euo pipefail

VERSION=""
INSTALL_DIR="/usr/local/bin"
DOWNLOAD_URL=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --version=*)
            VERSION="${1#*=}"
            shift
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --install-dir=*)
            INSTALL_DIR="${1#*=}"
            shift
            ;;
        --url)
            DOWNLOAD_URL="$2"
            shift 2
            ;;
        --url=*)
            DOWNLOAD_URL="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./scripts/install-linux.sh --version <version> [--install-dir <dir>] [--url <url>] [--dry-run]

Install the standalone agent-learn binary from a Linux release artifact.
Examples:
    ./scripts/install-linux.sh --version 0.8.2 --install-dir /usr/local/bin
    ./scripts/install-linux.sh --version 0.8.2 --install-dir "$HOME/.local/bin"
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${VERSION}" && -z "${DOWNLOAD_URL}" ]]; then
    echo "Either --version or --url is required." >&2
    exit 2
fi

if [[ -n "${VERSION}" ]]; then
    if [[ "${VERSION}" == v* ]]; then
        RELEASE_TAG="${VERSION}"
        RELEASE_VERSION="${VERSION#v}"
    else
        RELEASE_TAG="v${VERSION}"
        RELEASE_VERSION="${VERSION}"
    fi
    DOWNLOAD_URL="https://github.com/microsoft/agent-learning/releases/download/${RELEASE_TAG}/agent-learning-cli-${RELEASE_VERSION}-linux-x64.tar.gz"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry run: would install agent-learn from ${DOWNLOAD_URL}"
    echo "Install directory: ${INSTALL_DIR}"
    exit 0
fi

mkdir -p "${INSTALL_DIR}"

if [[ ! -w "${INSTALL_DIR}" ]]; then
    echo "Install directory is not writable: ${INSTALL_DIR}" >&2
    exit 1
fi

curl -fsSL "${DOWNLOAD_URL}" -o "${TMP_DIR}/agent-learning.tar.gz"
tar -xzf "${TMP_DIR}/agent-learning.tar.gz" -C "${TMP_DIR}"
install -m 0755 "${TMP_DIR}/agent-learn" "${INSTALL_DIR}/agent-learn"

echo "Installed agent-learn to ${INSTALL_DIR}/agent-learn"
echo "Run: agent-learn --help"
