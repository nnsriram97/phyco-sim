#!/usr/bin/env bash
# Sets up a no-Docker environment for VidGenSim using Blender's bundled Python.
# Downloads Blender 3.4 to the parent of the repo, then pip-installs all deps
# into Blender's Python interpreter so kubric/PyBullet can run headlessly with bpy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_ROOT")"
BLENDER_DIR="$PARENT_DIR/blender-3.4.0-linux-x64"
BLENDER_PY="$BLENDER_DIR/3.4/python/bin/python3.10"

cd "$PARENT_DIR"

if [ ! -d "$BLENDER_DIR" ]; then
  wget https://download.blender.org/release/Blender3.4/blender-3.4.0-linux-x64.tar.xz
  tar -xvf blender-3.4.0-linux-x64.tar.xz
fi

"$BLENDER_PY" -m ensurepip --upgrade
"$BLENDER_PY" -m pip install --upgrade pip

"$BLENDER_PY" -m pip install -r "$REPO_ROOT/kubric/requirements.txt"
"$BLENDER_PY" -m pip install -r "$REPO_ROOT/src/requirements.txt"

"$BLENDER_PY" -m pip install https://download.blender.org/pypi/bpy/bpy-3.4.0-cp310-cp310-manylinux_2_17_x86_64.whl
"$BLENDER_PY" -m pip install pybullet importlib_resources OpenEXR loguru scikit-image matplotlib
# opencv-python build compatible with numpy 1.26.4
"$BLENDER_PY" -m pip install opencv-python==4.10.0.84

echo
echo "Done. Use this Python for all VidGenSim launches:"
echo "  $BLENDER_PY"
