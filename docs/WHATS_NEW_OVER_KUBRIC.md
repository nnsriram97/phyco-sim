# What's New Over Kubric

[Kubric](https://github.com/google-research/kubric) is a great framework for generating synthetic video data, but it was designed primarily for rigid body scenes. VidGenSim extends it with several capabilities needed for generating physically diverse training data.

---

## Soft Body Simulation

Kubric only supports rigid bodies. We add a full soft body pipeline:

- New `kb.SoftBody` class for deformable objects (jelly cubes, soft balls, rubber objects)
- Two material models: **Mass-Spring** (fast, intuitive) and **Neo-Hookean** (physically accurate, rubber-like)
- Tetrahedral mesh support (`.vtk` format) for volumetric simulation via PyBullet's `loadSoftBody`
- Tri-to-tet vertex mapping for reconstructing surface meshes from volumetric deformations
- Blender shape-key animation for rendering deformed meshes frame-by-frame
- Soft body anchoring — pin nodes to world positions or rigid bodies

See [SOFT_BODY_GUIDE.md](SOFT_BODY_GUIDE.md) for usage details, parameter tuning, and mesh preparation.

## Force Application

Kubric has no mechanism to apply external forces mid-simulation. We add:

- `apply_force(obj, force, point, frame)` — instantaneous force at a specific frame
- `add_persistent_force(obj, force, ...)` — force applied every simulation substep
- `apply_torque(obj, torque, frame)` — rotational forces
- `apply_soft_body_force(obj, node_index, force)` — forces on individual soft body nodes
- `add_persistent_velocity(obj, velocity)` — enforce constant velocity each substep

These are exposed through the `PyBullet` simulator wrapper in `kubric/kubric/simulator/pybullet.py`.

## Fine-Grained Physics Control

Every scenario script exposes physics parameters as command-line arguments, making it easy to systematically vary properties across large dataset runs:

| Property | Example Arguments | Used In |
|---|---|---|
| **Friction** | `--platform_friction`, `--obj_friction` | Friction slide scenarios |
| **Restitution** | `--ball_restitution`, `--platform_restitution` | Ball drop, ball-wall collision |
| **Deformation** | `--neo_hookean_mu`, `--spring_elastic_stiffness` | Cube deform, soft ball drop |
| **Force** | `--force_magnitude`, `--force_direction` | Brick toss, pool table, Jenga |

## Texture Randomization

Scenarios randomly sample PBR textures for surfaces and objects to increase visual diversity. Supported texture categories:

- Wood, concrete, brick, ground, sponge, rubber

Textures are loaded from a configurable directory set via the `SIM_ASSETS_DIR` environment variable. See the main [README](../README.md#downloading-assets) for download instructions.

## HDRI Environment Lighting

Support for loading HDRI environment maps for realistic ambient lighting and backgrounds, with configurable rotation and intensity. Implemented in the Blender renderer wrapper (`kubric/kubric/renderer/blender.py`).

## Rich Metadata Export

Every simulation outputs a JSON file with:

- Per-object physics properties (mass, friction, restitution, soft body parameters)
- Camera intrinsics and extrinsics
- Applied forces (magnitude, direction, application point) in both world and camera coordinates
- Per-frame object positions and rotations
- Segmentation IDs for each object

This is handled by the utilities in `src/kubric_utils.py`.

## Scalable Execution

- `ThreadPoolExecutor`-based parallel runners (`src/run_parallel_*.py`) with cyclic GPU assignment
- SLURM orchestrator script for HPC clusters (`scripts/launch/generic_slurm_orchestrator_v2.sh`)

See [SLURM.md](SLURM.md) for HPC details.
