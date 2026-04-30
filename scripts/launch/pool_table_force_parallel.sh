#!/bin/bash

# Non-Docker Parallel Pool Table Force Simulation Launch Script
# This script runs parallel pool table force simulations using a specific Python interpreter across multiple GPUs

# Set default values
PYTHON_PATH="/workspace/blender-3.4.0-linux-x64/3.4/python/bin/python3.10"
OUTPUT_DIR="/net/acadia2a/data/sriram/vidgen/datasets/kubric_generated/pool_table_force"
NUM_WORKERS=12
NUM_VIDEOS=1000
RESOLUTION="768x432"
FRAME_END=96
FRAME_RATE=24
MIN_FORCE=80.0
MAX_FORCE=450.0
COMPOSITION_STYLE="overhead"
FFMPEG_PATH="/home/sriram/research/helper_packages/ffmpeg-git-20240629-amd64-static/"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --python_path)
      PYTHON_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --num_videos)
      NUM_VIDEOS="$2"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --frame_end)
      FRAME_END="$2"
      shift 2
      ;;
    --frame_rate)
      FRAME_RATE="$2"
      shift 2
      ;;
    --min_force)
      MIN_FORCE="$2"
      shift 2
      ;;
    --max_force)
      MAX_FORCE="$2"
      shift 2
      ;;
    --composition_style)
      COMPOSITION_STYLE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 --python_path PATH [OPTIONS]"
      echo ""
      echo "Required:"
      echo "  --python_path PATH         Path to Python interpreter (e.g., Blender Python)"
      echo ""
      echo "Options:"
      echo "  --output_dir DIR           Output directory"
      echo "  --num_workers N            Number of parallel workers (default: 12)"
      echo "  --num_videos N             Total number of videos to generate (default: 1000)"
      echo "  --resolution WxH           Video resolution (default: 768x432)"
      echo "  --frame_end N              Number of frames (default: 15)"
      echo "  --frame_rate N             Frame rate (default: 10)"
      echo "  --min_force N              Minimum force magnitude in Newtons (default: 80.0)"
      echo "  --max_force N              Maximum force magnitude in Newtons (default: 450.0)"
      echo "  --composition_style STYLE  Camera composition style (overhead, corner, side, cue_line) (default: overhead)"
      echo "  --help                     Show this help message"
      echo ""
      echo "Note: This script will run parallel pool table force simulations using the specified Python interpreter."
      echo "      Each video will randomly vary ball properties, table properties, force magnitude,"
      echo "      and composition style for realistic pool table physics."
      echo ""
      echo "Example:"
      echo "  $0 --python_path /path/to/blender/python --num_workers 8 --num_videos 100"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Add ffmpeg path to PATH
export PATH="$FFMPEG_PATH:$PATH"

# Check if Python path is provided
if [ -z "$PYTHON_PATH" ]; then
    echo "ERROR: --python_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if Python path exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "ERROR: Python path does not exist: $PYTHON_PATH"
    exit 1
fi

echo "================================================"
echo "Parallel Pool Table Force Simulation Launch (Non-Docker)"
echo "================================================"
echo "Python path: $PYTHON_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of workers: $NUM_WORKERS"
echo "Number of videos: $NUM_VIDEOS"
echo "Resolution: $RESOLUTION"
echo "Frames: $FRAME_END"
echo "Frame rate: $FRAME_RATE"
echo "Force parameters:"
echo "  Min force: ${MIN_FORCE}N"
echo "  Max force: ${MAX_FORCE}N"
echo "Composition style: $COMPOSITION_STYLE"
echo "Variable parameters:"
echo "  Ball friction: 0.2-0.3"
echo "  Ball restitution: 0.8-0.95"
echo "  Table friction: 0.2-0.3"
echo "  Table restitution: 0.5-0.6"
echo "  Force magnitude: ${MIN_FORCE}N-${MAX_FORCE}N (uniform sampling)"
echo "Available composition styles:"
echo "  overhead, corner, side, cue_line"
echo "================================================"

# Detect available GPUs
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ $GPU_COUNT -eq 0 ]; then
    echo "WARNING: No GPUs detected. This may impact performance."
    GPU_COUNT=1
else
    echo "Detected GPUs: $GPU_COUNT"
fi

# Recommend optimal worker count
if [ $NUM_WORKERS -gt $GPU_COUNT ]; then
    echo "INFO: Using $NUM_WORKERS workers with $GPU_COUNT GPUs. Workers will share GPUs."
elif [ $NUM_WORKERS -lt $GPU_COUNT ]; then
    echo "INFO: Using $NUM_WORKERS workers with $GPU_COUNT GPUs. Some GPUs will be unused."
    echo "      Consider increasing --num_workers to $GPU_COUNT for better utilization."
fi

# Create local output directory
mkdir -p "$OUTPUT_DIR"

echo "Local output directory: $OUTPUT_DIR"
echo "================================================"

# Set up environment variables
export KUBRIC_USE_GPU=true
export NVIDIA_DRIVER_CAPABILITIES=all
export NVIDIA_VISIBLE_DEVICES=all

echo "================================================"
echo "Starting parallel pool table force simulation..."
echo "================================================"

# Run the Python script
"$PYTHON_PATH" src/run_parallel_pool_table_force.py \
  --output_dir="$OUTPUT_DIR" \
  --num_workers="$NUM_WORKERS" \
  --num_videos="$NUM_VIDEOS" \
  --resolution="$RESOLUTION" \
  --frame_end="$FRAME_END" \
  --frame_rate="$FRAME_RATE" \
  --min_force="$MIN_FORCE" \
  --max_force="$MAX_FORCE" \
  --composition_style="$COMPOSITION_STYLE" \
  --efficient_rendering \
  --save_mp4 \
  --layers=image,segmentation,depth

EXIT_CODE=$?

echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Parallel pool table force simulation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    
    # Count generated videos
    if [ -d "$OUTPUT_DIR" ]; then
        VIDEO_COUNT=$(find "$OUTPUT_DIR" -name "*.mp4" | wc -l)
        echo "Generated videos: $VIDEO_COUNT"
        
        # Show directory structure
        echo ""
        echo "Output directory structure:"
        ls -la "$OUTPUT_DIR"
        
        # Show sample video directories if they exist
        echo ""
        echo "Sample video directories:"
        find "$OUTPUT_DIR" -maxdepth 2 -type d -name "*" | head -5
        
        # Show sample metadata for first few videos
        echo ""
        echo "Sample video metadata:"
        find "$OUTPUT_DIR" -name "metadata.json" | head -3 | while read metadata_file; do
            video_dir=$(dirname "$metadata_file")
            echo "Video: $(basename "$video_dir")"
            if command -v jq >/dev/null 2>&1; then
                echo "  Ball friction: $(jq -r '.ball_friction // "N/A"' "$metadata_file")"
                echo "  Ball restitution: $(jq -r '.ball_restitution // "N/A"' "$metadata_file")"
                echo "  Table friction: $(jq -r '.table_friction // "N/A"' "$metadata_file")"
                echo "  Table restitution: $(jq -r '.table_restitution // "N/A"' "$metadata_file")"
                echo "  Force magnitude: $(jq -r '.force_magnitude // "N/A"' "$metadata_file")N"
                echo "  Composition style: $(jq -r '.composition_style // "N/A"' "$metadata_file")"
            else
                echo "  (Install jq to see detailed metadata)"
            fi
        done
    fi
else
    echo "❌ Parallel pool table force simulation failed with exit code: $EXIT_CODE"
fi
echo "================================================"

exit $EXIT_CODE
