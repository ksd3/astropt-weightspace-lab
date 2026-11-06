#!/usr/bin/env bash
# Build everything the sandbox needs. Requires Docker and uv (https://docs.astral.sh/uv/).
#   ./setup_sandbox.sh            the full set of packages the experiments use (PyTorch with CUDA: about 7 GB)
#   ./setup_sandbox.sh --minimal  numpy only: enough for sandbox_example.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.sandbox-python"
PYTHON_VERSION=3.13.5

# The image holds only empty mount points. /usr and /etc come from the host, read-only, at run time.
ctx="$(mktemp -d)"; trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx"/rootfs/{etc,home,opt,root,run,sbin,tmp,usr,var/tmp,work}
ln -s usr/bin "$ctx/rootfs/bin"; ln -s usr/lib "$ctx/rootfs/lib"; ln -s usr/lib64 "$ctx/rootfs/lib64"
printf 'FROM scratch\nCOPY rootfs/ /\n' > "$ctx/Dockerfile"
docker build -q -t sandbox-host:1 "$ctx"

# A clean Python that lives wholly inside one folder, so it can be mounted read-only into the containers.
mkdir -p "$PY"
UV_PYTHON_INSTALL_DIR="$PY/cpython" uv python install "$PYTHON_VERSION"
uv venv --clear --python "$(echo "$PY"/cpython/cpython-"$PYTHON_VERSION"-*/bin/python3)" "$PY/env"
if [ "${1:-}" = "--minimal" ]; then
  uv pip install --python "$PY/env/bin/python" numpy==2.3.4
else
  uv pip install --python "$PY/env/bin/python" -r "$HERE/requirements-sandbox.txt"
fi
mkdir -p "$HOME/.sandbox-runs"
echo "Sandbox ready. Next: python3 sandbox_example.py"
