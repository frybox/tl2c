#!/usr/bin/env bash
# Build the tl2c standalone executable (PyInstaller onedir).
#
# Requires a python with tilelang + PyInstaller installed (default: the base
# miniconda python). Produces dist/tl2c/tl2c (a directory bundle).
#
# Usage:
#   ./build.sh                 # build into ./dist/tl2c
#   PY=/path/to/python ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-/home/jzjian/miniconda3/bin/python}"

# A failed build must not leave a half-assembled dist/tl2c (EXE assembly treats
# an existing directory at the output path as an invalid file).
trap 'echo "[build] FAILED; removing partial dist/tl2c"; rm -rf dist/tl2c' ERR

echo "[build] using python: $PY"
"$PY" - <<'PY' | sed 's/^/[build]   /'
import tilelang, PyInstaller
print("tilelang", tilelang.__version__, "| PyInstaller", PyInstaller.__version__)
PY

rm -rf dist/tl2c build/tl2c
"$PY" -m PyInstaller tl2c.spec --clean --noconfirm --distpath dist --workpath build

echo
echo "[build] done. Output: $(pwd)/dist/tl2c/"
echo "[build] try: $(pwd)/dist/tl2c/tl2c --help"
# quick self-check if an example is present
if [ -f smoketest/op.py ]; then
  echo "[build] self-test..."
  OUT="$(mktemp -d)/add.cu"
  ./dist/tl2c/tl2c smoketest/op.py -o "$OUT" \
    --args M=1024 N=1024 block_M=32 block_N=32 threads=128 in_dtype=float32 out_dtype=float32 >/dev/null
  if [ -s "$OUT" ]; then
    echo "[build] self-test OK: $(wc -c < "$OUT") bytes -> $OUT (temp)"
  else
    echo "[build] self-test FAILED" >&2
    exit 1
  fi
fi
