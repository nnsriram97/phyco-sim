# Installation

VidGenSim runs without Docker by using Blender 3.4's bundled CPython 3.10 as the interpreter — `bpy` (Blender-as-a-Python-module) ships in that interpreter, so headless rendering works without a separate Blender install.

## One-shot setup

From the repo root:

```bash
bash setup_scripts/setup_kubric_no_docker.sh
```

This downloads Blender 3.4 into the **parent directory** of the repo (so it lives next to the source tree, not inside it) and pip-installs the runtime deps into Blender's Python:

- `kubric/requirements.txt` and `src/requirements.txt`
- `bpy==3.4.0` wheel
- pybullet, OpenEXR, loguru, scikit-image, matplotlib, opencv-python 4.10

When it finishes, it prints the absolute path of the Python interpreter to use. Save that path — every launch script takes it as `--python_path`:

```bash
PYTHON_PATH=../blender-3.4.0-linux-x64/3.4/python/bin/python3.10
```

## Verifying the environment

```bash
"$PYTHON_PATH" - <<'PY'
import bpy, numpy, pybullet
print("bpy:", bpy.app.version_string)
print("numpy:", numpy.__version__)
print("pybullet ok")
PY
```

## Smoke test (one video)

```bash
bash scripts/launch/ball_drop_v2_parallel.sh \
  --python_path "$PYTHON_PATH" \
  --num_workers 1 --num_videos 1 \
  --output_dir ./output/smoke_test
```

## GPU rendering

Set `KUBRIC_USE_GPU=true` and ensure NVIDIA drivers + CUDA are available. The launch scripts already export this. The Cycles kernel cache lives at `~/.cache/cycles/kernels` — persist it between runs to avoid recompilation. See [EFFICIENT_RENDERING.md](EFFICIENT_RENDERING.md) for further tuning.

## Texture / HDRI assets

The scenarios sample ground textures, wood/concrete materials, and HDRI lighting from a `SIM_ASSETS_DIR` tree (default `./sim_assets`). See the asset bundle section of the top-level [README](../README.md) for the download URL.
