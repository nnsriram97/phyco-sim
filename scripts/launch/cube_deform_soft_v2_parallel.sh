#!/bin/bash

# Non-Docker Parallel Cube Deform Simulation Launch Script
# This script runs parallel cube deformation simulations using a specific Python interpreter across multiple GPUs

# Set default values
PYTHON_PATH="/workspace/blender-3.4.0-linux-x64/3.4/python/bin/python3.10"
OUTPUT_DIR="/net/acadia1a/data/sriram/vidgen/datasets/kubric_generated/cube_deform_soft_v2_noeff"
NUM_WORKERS=12
NUM_VIDEOS=1000
RESOLUTION="768x432"
FRAME_END=97
FRAME_RATE=24
SIM_ASSETS_DIR="/net/acadia1a/data/sriram/sim_assets"
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
    --sim_assets_dir)
      SIM_ASSETS_DIR="$2"
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
      echo "  --frame_end N              Number of frames (default: 97)"
      echo "  --frame_rate N             Frame rate (default: 24)"
      echo "  --sim_assets_dir DIR       Simulation assets directory (default: /net/acadia1a/data/sriram/sim_assets)"
      echo "  --help                     Show this help message"
      echo ""
      echo "Note: This script will run parallel simulations using the specified Python interpreter."
      echo "      Each video will randomly vary cube neo-hookean mu and damping"
      echo "      while keeping cube mass, friction, and restitution fixed."
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
echo "Parallel Cube Deform Simulation Launch (Non-Docker)"
echo "================================================"
echo "Python path: $PYTHON_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of workers: $NUM_WORKERS"
echo "Number of videos: $NUM_VIDEOS"
echo "Resolution: $RESOLUTION"
echo "Frames: $FRAME_END"
echo "Frame rate: $FRAME_RATE"
echo "Fixed parameters:"
echo "  Cube mass: 1.0 kg"
echo "  Cube restitution: 0.05"
echo "  Weight type: ball (fixed)"
echo "  Weight mass: 1.0 kg (fixed)"
echo "Variable parameters:"
echo "  Cube neo-hookean mu: 60-600"
echo "  Cube neo-hookean damping: 0.01-1.0"
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
echo "Starting parallel cube deform simulation..."
echo "================================================"

# Run the Python script
"$PYTHON_PATH" src/run_parallel_cube_deform_soft_v2.py \
  --output_dir="$OUTPUT_DIR" \
  --num_workers="$NUM_WORKERS" \
  --num_videos="$NUM_VIDEOS" \
  --resolution="$RESOLUTION" \
  --frame_end="$FRAME_END" \
  --frame_rate="$FRAME_RATE" \
  --save_mp4 \
  --layers=image,segmentation,depth

EXIT_CODE=$?

echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Parallel cube deform simulation completed successfully!"
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
                echo "  Cube mass: $(jq -r '.object_data.mass[0] // "N/A"' "$metadata_file")"
                echo "  Cube friction: $(jq -r '.object_data.friction[0] // "N/A"' "$metadata_file")"
                echo "  Cube restitution: $(jq -r '.object_data.restitution[0] // "N/A"' "$metadata_file")"
                echo "  Weight type: ball"
                echo "  Weight mass: 1.0 kg"
                echo "  Cube neo-hookean mu: $(jq -r '.object_data._neo_hookean_mu[0] // "N/A"' "$metadata_file")"
                echo "  Cube neo-hookean damping: $(jq -r '.object_data._neo_hookean_damping[0] // "N/A"' "$metadata_file")"
            else
                echo "  (Install jq to see detailed metadata)"
            fi
        done
    fi
else
    echo "❌ Parallel cube deform simulation failed with exit code: $EXIT_CODE"
fi
echo "================================================"

exit $EXIT_CODE
