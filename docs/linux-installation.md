# Linux installation

Agent Learning ships a standalone Linux CLI archive that is suitable for Debian/Ubuntu, RHEL-compatible distributions, VMs, containers, and CI runners.

## Install from a release artifact

Use the automated installer script to download the published Linux archive and place `agent-learn` on your `PATH`:

```bash
curl -fsSL https://raw.githubusercontent.com/microsoft/agent-learning/main/scripts/install-linux.sh -o /tmp/install-linux.sh
bash /tmp/install-linux.sh --version 0.8.0 --install-dir /usr/local/bin
```

The installer expects a release asset named `agent-learning-cli-<version>-linux-x64.tar.gz`. For a user-local install, point `--install-dir` at `~/.local/bin`.

## Debian/Ubuntu and RHEL-compatible usage

The same archive works on Debian/Ubuntu and RHEL-compatible systems. For example:

```bash
sudo bash ./scripts/install-linux.sh --version 0.8.0 --install-dir /usr/local/bin
```

The installer only needs the release archive and a writable destination directory, so the same command is suitable for VMs, containers, and CI runners on either family of Linux distributions.

After installation, verify the CLI:

```bash
agent-learn --help
```

## Container deployment

The release artifact can also be used in containers and CI jobs without requiring Python or `pip`.

```bash
docker run --rm --entrypoint /bin/bash ubuntu:24.04 -lc '
  set -e
  apt-get update
  apt-get install -y curl tar ca-certificates
  curl -fsSL -o /tmp/agent-learning.tar.gz https://github.com/microsoft/agent-learning/releases/download/v0.8.0/agent-learning-cli-0.8.0-linux-x64.tar.gz
  mkdir -p /opt/agent-learning/bin
  tar -xzf /tmp/agent-learning.tar.gz -C /opt/agent-learning/bin
  ln -sf /opt/agent-learning/bin/agent-learn /usr/local/bin/agent-learn
  agent-learn --help
'
```

For GitHub Actions or other CI systems, download the release archive directly and add the extracted binary to `PATH`.
