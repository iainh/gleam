#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DEFAULT_FACIL_DIR="$ROOT_DIR/vendor/facil.io"

if [ -z "${FACIL_DIR:-}" ]; then
  FACIL_DIR="$DEFAULT_FACIL_DIR"
fi

make -C "$ROOT_DIR" FACIL_DIR="$FACIL_DIR"
