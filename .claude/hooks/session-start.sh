#!/bin/bash
set -euo pipefail

# Only touch the environment on Claude Code on the web; leave local machines
# alone.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# harmonic_plots.py's custom_rcparams (CLAUDE.md Section 6) requests a serif
# font stack ("Times New Roman", "Times", "Nimbus Roman") to match the
# project's paper figures. Only "Nimbus Roman" (URW base-35) is available via
# apt; without it matplotlib silently falls back to DejaVu Sans and prints a
# "findfont" warning on every figure.
apt-get update -qq
apt-get install -y -qq fonts-urw-base35

# The base image's system Python is externally managed (PEP 668) and its
# apt-installed "packaging" has no pip RECORD, so a plain "pip install"
# aborts trying to upgrade it in place; --ignore-installed sidesteps that
# without needing a separate venv.
python3 -m pip install --break-system-packages --ignore-installed --quiet \
  -r "$CLAUDE_PROJECT_DIR/requirements.txt"
