# SLURM Orchestration

`scripts/launch/generic_slurm_orchestrator_v2.sh` submits any of the per-scenario `*_parallel.sh` launchers as a SLURM job array. Each job in the array runs one batch of videos with fixed resources.

## Fixed per-job resources

- 8 CPUs
- 48 GB RAM
- 1 GPU
- 2 workers
- 40 videos per batch (default)

## Usage

```bash
bash scripts/launch/generic_slurm_orchestrator_v2.sh \
  --script scripts/launch/ball_drop_v2_parallel.sh \
  --total_num_videos 15000 \
  --videos_per_batch 40 \
  --partition gpu \
  --time_limit "12:00:00" \
  --script_args "--python_path /path/to/blender/python --output_dir /path/to/output"
```

The orchestrator computes `ceil(total_num_videos / videos_per_batch)` array tasks and submits them as a single `sbatch --array=...` job. Pass any per-scenario flags through `--script_args` (in quotes).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--script` | _(required)_ | Path to a `*_parallel.sh` launcher |
| `--total_num_videos` | 2000 | Total videos to generate across the array |
| `--videos_per_batch` | 40 | Videos per array task |
| `--partition` | _(none)_ | SLURM partition |
| `--qos` | _(none)_ | SLURM QOS |
| `--time_limit` | `24:00:00` | Per-job wall time |
| `--dry_run` | false | Print the `sbatch` command without submitting |
| `--script_args` | _(empty)_ | Quoted string of args forwarded to `--script` |

## Monitoring

```bash
squeue -u $USER                # active jobs
scontrol show job <job_id>     # job details
tail -f <job_name>_<job_id>.out
```

Each task writes one `.out` and one `.err` log next to the output directory.
