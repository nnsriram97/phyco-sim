#!/bin/bash

# Non-Docker Parallel Ball-Wall Collision Simulation Launch Script
# This script runs parallel ball-wall collision simulations using a specific Python interpreter across multiple GPUs

# Set default values
PYTHON_PATH="/workspace/blender-3.4.0-linux-x64/3.4/python/bin/python3.10"
OUTPUT_DIR="/net/acadia1a/data/sriram/vidgen/datasets/kubric_generated/ball_wall_collision_v2_ball_only"
NUM_WORKERS=16
NUM_VIDEOS=2000
RESOLUTION="768x432"
FRAME_END=97
FRAME_RATE=24
PLATFORM_FRICTION=0.2
BALL_FRICTION=0.2
WALL_FRICTION=0.2
SIM_ASSETS_DIR="/net/acadia1a/data/sriram/sim_assets"
KUBRIC_CACHE_DIR="./kubric_cache/cycles_kernels"
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
    --platform_friction)
      PLATFORM_FRICTION="$2"
      shift 2
      ;;
    --ball_friction)
      BALL_FRICTION="$2"
      shift 2
      ;;
    --wall_friction)
      WALL_FRICTION="$2"
      shift 2
      ;;
    --sim_assets_dir)
      SIM_ASSETS_DIR="$2"
      shift 2
      ;;
    --kubric_cache_dir)
      KUBRIC_CACHE_DIR="$2"
      shift 2
      ;;
    --ffmpeg_path)
      FFMPEG_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 --python_path PATH [OPTIONS]"
      echo ""
      echo "Required:"
      echo "  --python_path PATH         Path to Python interpreter (e.g., Blender Python)"
      echo ""
      echo "Options:"
      echo "  --output_dir DIR           Output directory (default: /net/acadia1a/data/sriram/vidgen/datasets/kubric_generated/ball_wall_collision)"
      echo "  --num_workers N            Number of parallel workers (default: 16)"
      echo "  --num_videos N             Total number of videos to generate (default: 2000)"
      echo "  --resolution WxH           Video resolution (default: 768x432)"
      echo "  --frame_end N              Number of frames (default: 97)"
      echo "  --frame_rate N             Frame rate (default: 24)"
      echo "  --platform_friction F      Platform friction coefficient (default: 0.2)"
      echo "  --ball_friction F          Ball friction coefficient (default: 0.2)"
      echo "  --wall_friction F          Wall friction coefficient (default: 0.2)"
      echo "  --sim_assets_dir DIR       Simulation assets directory (default: /net/acadia1a/data/sriram/sim_assets)"
      echo "  --kubric_cache_dir DIR     Kubric cache directory (default: ./kubric_cache/cycles_kernels)"
      echo "  --ffmpeg_path PATH         Path to ffmpeg binaries (default: /home/sriram/research/helper_packages/ffmpeg-git-20240629-amd64-static/)"
      echo "  --help                     Show this help message"
      echo ""
      echo "Note: This script will run parallel simulations using the specified Python interpreter."
      echo "      Each video will randomly vary ball and wall restitution coefficients"
      echo "      while keeping friction values fixed for realistic collision dynamics."
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

# Add ffmpeg path to PATH
export PATH="$FFMPEG_PATH:$PATH"

echo "================================================"
echo "Parallel Ball-Wall Collision Simulation Launch (Non-Docker)"
echo "================================================"
echo "Python path: $PYTHON_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of workers: $NUM_WORKERS"
echo "Number of videos: $NUM_VIDEOS"
echo "Resolution: $RESOLUTION"
echo "Frames: $FRAME_END"
echo "Frame rate: $FRAME_RATE"
echo "Fixed friction values:"
echo "  Platform friction: $PLATFORM_FRICTION"
echo "  Ball friction: $BALL_FRICTION"
echo "  Wall friction: $WALL_FRICTION"
echo "Restitution variation: Random ball and wall coefficients"
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
echo "Starting parallel ball-wall collision simulation..."
echo "================================================"

# Run the Python script
"$PYTHON_PATH" src/run_parallel_ball_wall_collide.py \
  --output_dir="$OUTPUT_DIR" \
  --num_workers="$NUM_WORKERS" \
  --num_videos="$NUM_VIDEOS" \
  --resolution="$RESOLUTION" \
  --frame_end="$FRAME_END" \
  --frame_rate="$FRAME_RATE" \
  --platform_friction="$PLATFORM_FRICTION" \
  --ball_friction="$BALL_FRICTION" \
  --wall_friction="$WALL_FRICTION" \
  --efficient_rendering \
  --save_mp4 \
  --vary_restitution_only \
  --layers=image,segmentation,depth

EXIT_CODE=$?

echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Parallel ball-wall collision simulation completed successfully!"
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
        find "$OUTPUT_DIR" -maxdepth 2 -type d -name "*-*-*" | head -5
        
        # Show sample metadata for first few videos
        echo ""
        echo "Sample video metadata:"
        find "$OUTPUT_DIR" -name "metadata.json" | head -3 | while read metadata_file; do
            video_dir=$(dirname "$metadata_file")
            echo "Video: $(basename "$video_dir")"
            if command -v jq >/dev/null 2>&1; then
                echo "  Ball restitution: $(jq -r '.ball_restitution // "N/A"' "$metadata_file")"
                echo "  Wall restitution: $(jq -r '.wall_restitution // "N/A"' "$metadata_file")"
                echo "  Ball friction: $(jq -r '.ball_friction // "N/A"' "$metadata_file")"
                echo "  Wall friction: $(jq -r '.wall_friction // "N/A"' "$metadata_file")"
            else
                echo "  (Install jq to see detailed metadata)"
            fi
        done
    fi
else
    echo "❌ Parallel ball-wall collision simulation failed with exit code: $EXIT_CODE"
fi
echo "================================================"

exit $EXIT_CODE

