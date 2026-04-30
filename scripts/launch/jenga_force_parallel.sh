#!/bin/bash

# Non-Docker Parallel Jenga Force Simulation Launch Script
# This script runs parallel Jenga force simulations using a specific Python interpreter across multiple GPUs

# Set default values
PYTHON_PATH="/workspace/blender-3.4.0-linux-x64/3.4/python/bin/python3.10"
OUTPUT_DIR="/net/acadia2a/data/sriram/vidgen/datasets/kubric_generated/jenga_force"
NUM_WORKERS=12
NUM_VIDEOS=1000
RESOLUTION="768x432"
FRAME_END=96
FRAME_RATE=24
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
      echo "  --help                     Show this help message"
      echo ""
      echo "Note: This script will run parallel Jenga force simulations using the specified Python interpreter."
      echo "      Each video will randomly vary composition styles with fixed Jenga tower parameters:"
      echo "      - Layers: 12-20 blocks"
      echo "      - Missing block probability: 40%"
      echo "      - Force layer bias: 0.8 (middle layer preference)"
      echo "      - Velocity magnitude: 1.0 m/s"
      echo "      - Camera positioned within 60° of velocity direction"
      echo ""
      echo "Available composition styles:"
      echo "  center, left_third, right_third, upper_left, upper_right, lower_center, lower_left, lower_right"
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
echo "Parallel Jenga Force Simulation Launch (Non-Docker)"
echo "================================================"
echo "Python path: $PYTHON_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of workers: $NUM_WORKERS"
echo "Number of videos: $NUM_VIDEOS"
echo "Resolution: $RESOLUTION"
echo "Frames: $FRAME_END"
echo "Frame rate: $FRAME_RATE"
echo "Fixed Jenga parameters:"
echo "  Tower layers: 12-20"
echo "  Missing block probability: 40%"
echo "  Force layer bias: 0.8 (middle layer preference)"
echo "  Velocity magnitude: 1.0 m/s"
echo "  Tower stability attempts: 10"
echo "Variable parameters:"
echo "  Composition styles: center, left_third, right_third, upper_left, upper_right, lower_center, lower_left, lower_right"
echo "Camera features:"
echo "  - Positioned within 60° of velocity direction"
echo "  - Varied composition styles for visual diversity"
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
echo "Starting parallel Jenga force simulation..."
echo "================================================"

# Run the Python script
"$PYTHON_PATH" src/run_parallel_jenga_force.py \
  --output_dir="$OUTPUT_DIR" \
  --num_workers="$NUM_WORKERS" \
  --num_videos="$NUM_VIDEOS" \
  --resolution="$RESOLUTION" \
  --frame_end="$FRAME_END" \
  --frame_rate="$FRAME_RATE" \
  --efficient_rendering \
  --save_mp4 \
  --layers=image,segmentation,depth

EXIT_CODE=$?

echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Parallel Jenga force simulation completed successfully!"
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
                echo "  Jenga layers: $(jq -r '.jenga_layer_count // "N/A"' "$metadata_file")"
                echo "  Blocks total: $(jq -r '.jenga_blocks_total // "N/A"' "$metadata_file")"
                echo "  Missing blocks: $(jq -r '.jenga_missing_blocks // "N/A"' "$metadata_file")"
                echo "  Velocity magnitude: $(jq -r '.velocity_magnitude // "N/A"' "$metadata_file") m/s"
                echo "  Composition style: $(jq -r '.camera_composition_style // "N/A"' "$metadata_file")"
                echo "  Camera-velocity angle: $(jq -r '.camera_velocity_angle_degrees // "N/A"' "$metadata_file")°"
            else
                echo "  (Install jq to see detailed metadata)"
            fi
        done
    fi
else
    echo "❌ Parallel Jenga force simulation failed with exit code: $EXIT_CODE"
fi
echo "================================================"

exit $EXIT_CODE
