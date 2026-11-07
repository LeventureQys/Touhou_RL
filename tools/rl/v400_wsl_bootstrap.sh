#!/usr/bin/env bash
# v4.0.0 P0-WP0: bootstrap the TH08 training ground inside WSL2 (Ubuntu).
#
# Run *inside* the distro.  Everything here is idempotent, never prompts for a
# password, and never writes one to disk.
#
#   bash tools/rl/v400_wsl_bootstrap.sh --dry-run     # print the plan, run nothing
#   bash tools/rl/v400_wsl_bootstrap.sh               # apt + venv + pip
#   bash tools/rl/v400_wsl_bootstrap.sh --assets-from /mnt/d/WorkShop/Touhou_RL/third_party/th08/build/play
#   bash tools/rl/v400_wsl_bootstrap.sh --clone-from  /mnt/d/WorkShop/Touhou_RL \
#        --pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple
#
# What it does, in order (see Document/Update/Alpha/v4.0.0/10-P0-训练场.md WP0):
#   1. sanity-check that this really is an Ubuntu WSL2 distro;
#   2. apt-get install the build/runtime dependencies;
#   3. create a Python venv at $V400_VENV (default ~/v400-venv) and pip install;
#   4. [opt-in] clone the parent repository onto ext4, submodule included;
#   5. [opt-in] copy th08.dat / thbgm.dat into $V400_DATA/game and verify their hashes;
#   6. remind you to run tools/rl/v400_env_check.py, which is the actual gate.
#
# Step 6 is the gate: this script only makes the tools available, it does not
# prove the environment works.  Exit code 0 from v400_env_check.py does.
#
# Network notes: on a network where pypi.org or github.com is unreachable, pass
# --pip-index-url <mirror>; the submodule is then cloned from --clone-from.

set -euo pipefail

DRY_RUN=0
V400_VENV="${V400_VENV:-$HOME/v400-venv}"
V400_DATA="${V400_DATA:-$HOME/v400-data}"
CLONE_FROM=""
ASSETS_FROM=""
TORCH_SPEC="${TORCH_SPEC:-torch}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
# Where the general pip packages come from.  Empty means the default PyPI index.
# On networks where pypi.org is unreachable, point it at a mirror:
#   --pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
# The th08 submodule is fetched from github.com, which is not reachable from every
# network.  When --clone-from names a checkout that already contains the submodule,
# the URL is rewritten so the objects come from the local copy instead.
SUBMODULE_URL="https://github.com/N0zoM1z0/th08.git"

# Packages from 10-P0 WP0, kept in one place so the script and the doc cannot drift.
APT_PACKAGES=(
  build-essential g++ cmake ninja-build pkg-config git python3 python3-venv
  libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libfontconfig1-dev libfreetype-dev libgl-dev
  mesa-utils xvfb xauth gdb zstd fonts-vlgothic
)
PIP_PACKAGES=(numpy pytest gymnasium zstandard numba)

usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --clone-from) CLONE_FROM="${2:-}"; shift 2 ;;
    --clone-from=*) CLONE_FROM="${1#*=}"; shift ;;
    --assets-from) ASSETS_FROM="${2:-}"; shift 2 ;;
    --assets-from=*) ASSETS_FROM="${1#*=}"; shift ;;
    --pip-index-url) PIP_INDEX_URL="${2:-}"; shift 2 ;;
    --pip-index-url=*) PIP_INDEX_URL="${1#*=}"; shift ;;
    --torch-index-url) TORCH_INDEX_URL="${2:-}"; shift 2 ;;
    --torch-index-url=*) TORCH_INDEX_URL="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [dry-run] $*"
  else
    "$@"
  fi
}

step() { echo; echo "== $* =="; }

# --------------------------------------------------------------------------
step "1/6 sanity"
if [ "$(uname -s)" != "Linux" ]; then
  echo "FATAL: this must run inside the WSL2 Linux distro, not on Windows." >&2
  exit 3
fi
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  echo "  distro   : ${PRETTY_NAME:-unknown} (${VERSION_ID:-?})"
  if [ "${VERSION_ID:-}" != "24.04" ]; then
    echo "  WARNING  : v4.0.0 assumes Ubuntu 24.04; continuing on ${VERSION_ID:-unknown}." >&2
  fi
else
  echo "  WARNING  : /etc/os-release missing, cannot confirm the distro." >&2
fi
if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  echo "  kernel   : $(uname -r) (WSL)"
else
  echo "  kernel   : $(uname -r) (WARNING: this does not look like WSL)" >&2
fi
echo "  cores    : $(nproc)"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
    echo "  note     : not root, apt steps will use sudo (may prompt for a password)"
  else
    echo "FATAL: not root and sudo is missing; rerun as root." >&2
    exit 3
  fi
fi

# --------------------------------------------------------------------------
step "2/6 apt dependencies"
run $SUDO apt-get update
run env DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y "${APT_PACKAGES[@]}"

# --------------------------------------------------------------------------
step "3/6 python venv"
if [ -x "$V400_VENV/bin/python" ]; then
  echo "  reusing $V400_VENV"
else
  run python3 -m venv "$V400_VENV"
fi
PIP="$V400_VENV/bin/pip"
PIP_ARGS=()
[ -n "$PIP_INDEX_URL" ] && PIP_ARGS=(--index-url "$PIP_INDEX_URL")
run "$PIP" install "${PIP_ARGS[@]}" --upgrade pip
if [ -n "$TORCH_INDEX_URL" ]; then
  run "$PIP" install --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
else
  run "$PIP" install "${PIP_ARGS[@]}" "$TORCH_SPEC"
fi
run "$PIP" install "${PIP_ARGS[@]}" "${PIP_PACKAGES[@]}"

# --------------------------------------------------------------------------
step "4/6 repository (ext4, never /mnt/*)"
if [ -n "$CLONE_FROM" ]; then
  TARGET="${V400_REPO:-$HOME/Touhou_RL}"
  if [ -d "$TARGET/.git" ]; then
    echo "  reusing $TARGET"
  else
    run git clone "$CLONE_FROM" "$TARGET"
  fi
  # github.com may be unreachable; prefer the local copy of the submodule when the
  # source checkout already has it.  Written to the user config, not the repository.
  if [ -d "$CLONE_FROM/third_party/th08/.git" ] || [ -f "$CLONE_FROM/third_party/th08/.git" ]; then
    run git config --global "url.$CLONE_FROM/third_party/th08.insteadOf" "$SUBMODULE_URL"
    echo "  submodule url rewritten to $CLONE_FROM/third_party/th08"
  fi
  run git -C "$TARGET" submodule update --init
else
  echo "  skipped (pass --clone-from <path> to clone; keep the work tree on ext4)"
fi

# --------------------------------------------------------------------------
step "5/6 game assets"
if [ -n "$ASSETS_FROM" ]; then
  run mkdir -p "$V400_DATA/game"
  for name in th08.dat thbgm.dat; do
    if [ -f "$V400_DATA/game/$name" ]; then
      echo "  present: $V400_DATA/game/$name"
    else
      run cp "$ASSETS_FROM/$name" "$V400_DATA/game/$name"
    fi
  done
  if [ "$DRY_RUN" -eq 0 ]; then
    for name in th08.dat thbgm.dat; do
      printf '  %-11s %s\n' "$name" "$(sha256sum "$V400_DATA/game/$name" | cut -d' ' -f1)"
    done
  fi
else
  echo "  skipped (pass --assets-from <dir with th08.dat/thbgm.dat> to copy)"
fi

# --------------------------------------------------------------------------
step "6/6 verify"
echo "  V400_DATA=$V400_DATA"
echo "  V400_VENV=$V400_VENV"
echo
echo "  The gate is the env check, not this script:"
echo "    source $V400_VENV/bin/activate"
echo "    python tools/rl/v400_env_check.py"
echo
if [ "$DRY_RUN" -eq 0 ]; then
  echo "  torch: $("$V400_VENV/bin/python" -c 'import torch;print(torch.__version__, "cuda=" + str(torch.cuda.is_available()))' 2>/dev/null || echo 'not importable yet')"
fi
echo "  Record the torch version in EXP-P0-001; a mismatch against Windows must be noted."
