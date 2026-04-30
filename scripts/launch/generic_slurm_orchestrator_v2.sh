#!/bin/bash

# Simple SLURM Job Array Orchestrator for Video Generation
# Submits a job array with fixed resources per job

# Default values
SCRIPT_TO_RUN=""
TOTAL_NUM_VIDEOS=2000
VIDEOS_PER_BATCH=40
PARTITION=""
QOS=""
TIME_LIMIT="24:00:00"
DRY_RUN=false
SCRIPT_ARGS=""

# Script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --script)
      SCRIPT_TO_RUN="$2"
      shift 2
      ;;
    --total_num_videos)
      TOTAL_NUM_VIDEOS="$2"
      shift 2
      ;;
    --videos_per_batch)
      VIDEOS_PER_BATCH="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --time_limit)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --dry_run)
      DRY_RUN=true
      shift
      ;;
    --script_args)
      SCRIPT_ARGS="$2"
      shift 2
      ;;
    --help)
      echo "Simple SLURM Job Array Video Generation Orchestrator"
      echo ""
      echo "Usage: $0 --script SCRIPT_PATH [OPTIONS] --script_args \"ARGS_FOR_SCRIPT\""
      echo ""
      echo "Required:"
      echo "  --script PATH              Path to the script to run on each node"
      echo ""
      echo "Options:"
      echo "  --total_num_videos N       Total number of videos to generate (default: 2000)"
      echo "  --videos_per_batch N       Videos per batch/job (default: 40)"
      echo "  --partition NAME           SLURM partition to use"
      echo "  --qos NAME                 SLURM QOS to use"
      echo "  --time_limit TIME          Time limit per job (default: 24:00:00)"
      echo "  --dry_run                  Show what would be done without executing"
      echo "  --script_args \"ARGS\"       Arguments to pass to the script (in quotes)"
      echo "  --help                     Show this help message"
      echo ""
      echo "Fixed Resources per Job:"
      echo "  - 8 CPUs"
      echo "  - 48GB RAM"
      echo "  - 1 GPU"
      echo "  - 2 workers"
      echo "  - 40 videos per job"
      echo ""
      echo "Example:"
      echo "  $0 --script scripts/launch/ball_drop_v2_parallel.sh \\"
      echo "     --total_num_videos 2000 \\"
      echo "     --script_args \"--python_path /path/to/python --output_dir /path/to/output\""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$SCRIPT_TO_RUN" ]; then
    echo "ERROR: --script is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_TO_RUN" ]; then
    echo "ERROR: Script does not exist: $SCRIPT_TO_RUN"
    exit 1
fi

# Calculate number of jobs needed
total_jobs=$(( (TOTAL_NUM_VIDEOS + VIDEOS_PER_BATCH - 1) / VIDEOS_PER_BATCH ))

echo "================================================"
echo "Simple SLURM Job Array Video Generation"
echo "================================================"
echo "Script to run: $SCRIPT_TO_RUN"
echo "Total videos: $TOTAL_NUM_VIDEOS"
echo "Videos per job: $VIDEOS_PER_BATCH"
echo "Total jobs: $total_jobs"
if [ -n "$PARTITION" ]; then
    echo "Partition: $PARTITION"
fi
if [ -n "$QOS" ]; then
    echo "QOS: $QOS"
fi
echo "Time limit: $TIME_LIMIT"
echo "Script args: $SCRIPT_ARGS"
echo "Dry run: $DRY_RUN"
echo ""
echo "Fixed resources per job:"
echo "  - 8 CPUs"
echo "  - 48GB RAM" 
echo "  - 1 GPU"
echo "  - 2 workers"
echo "================================================"

# Check SLURM availability
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: SLURM commands not available. Are you on a SLURM cluster?"
    exit 1
fi

# Create job array script
job_script="/tmp/slurm_video_generation_array_$$.sh"

cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=video_gen_array
#SBATCH --output=video_gen_array_%A_%a.out
#SBATCH --error=video_gen_array_%A_%a.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=1-${total_jobs}
EOF

if [ -n "$PARTITION" ]; then
    echo "#SBATCH --partition=${PARTITION}" >> "$job_script"
fi

if [ -n "$QOS" ]; then
    echo "#SBATCH --qos=${QOS}" >> "$job_script"
fi

cat >> "$job_script" << EOF

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export KUBRIC_USE_GPU=true
export NVIDIA_DRIVER_CAPABILITIES=all
export NVIDIA_VISIBLE_DEVICES=0

# Calculate videos for this job
TOTAL_VIDEOS=${TOTAL_NUM_VIDEOS}
VIDEOS_PER_JOB=${VIDEOS_PER_BATCH}
JOB_INDEX=\$SLURM_ARRAY_TASK_ID

# Calculate how many videos this specific job should generate
START_VIDEO=\$(( (JOB_INDEX - 1) * VIDEOS_PER_JOB + 1 ))
END_VIDEO=\$(( JOB_INDEX * VIDEOS_PER_JOB ))

# For the last job, don't exceed total videos
if [ \$END_VIDEO -gt \$TOTAL_VIDEOS ]; then
    END_VIDEO=\$TOTAL_VIDEOS
fi

VIDEOS_THIS_JOB=\$(( END_VIDEO - START_VIDEO + 1 ))

# Print job information
echo "================================================"
echo "SLURM Job Array: video_gen_array"
echo "Array Job ID: \$SLURM_ARRAY_JOB_ID"
echo "Task ID: \$SLURM_ARRAY_TASK_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Job ID: \$SLURM_JOB_ID"
echo "CPUs: 8"
echo "Memory: 48GB"
echo "GPUs: 1"
echo "Workers: 2"
echo "Videos for this job: \$VIDEOS_THIS_JOB"
echo "Video range: \$START_VIDEO to \$END_VIDEO"
echo "Script: ${SCRIPT_TO_RUN}"
echo "================================================"

# Run the script with fixed parameters
bash "${SCRIPT_TO_RUN}" --num_workers 2 --num_videos \$VIDEOS_THIS_JOB ${SCRIPT_ARGS}

EXIT_CODE=\$?

echo "================================================"
echo "Job \$SLURM_ARRAY_TASK_ID completed with exit code: \$EXIT_CODE"
echo "================================================"

exit \$EXIT_CODE
EOF

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN: Would submit job array with script:"
    echo "---"
    cat "$job_script"
    echo "---"
    echo "Command: sbatch $job_script"
    rm "$job_script"
    exit 0
fi

# Submit the job array
echo "Submitting job array..."
job_output=$(sbatch "$job_script" 2>&1)
job_id=$(echo "$job_output" | awk '{print $4}')
rm "$job_script"

if [ -n "$job_id" ] && [[ "$job_output" == *"Submitted batch job"* ]]; then
    echo "Successfully submitted job array with ID: $job_id"
    echo "Job array will run $total_jobs jobs, each generating $VIDEOS_PER_BATCH videos"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $job_id"
    echo "  squeue -u \$USER | grep video_gen"
    echo ""
    echo "Cancel with:"
    echo "  scancel $job_id"
    echo "================================================"
else
    echo "Failed to submit job array: $job_output"
    exit 1
fi

echo "Job array submission complete!"
exit 0