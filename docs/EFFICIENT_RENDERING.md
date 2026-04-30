# Efficient Rendering Feature

## Overview

The efficient rendering feature automatically detects when objects in your simulation have stopped moving and avoids re-rendering identical frames. This can provide significant performance improvements for simulations where dynamic motion settles before the total frame count is reached.

## How It Works

1. **Motion Detection**: After running the physics simulation, the system analyzes the velocity and angular velocity of all dynamic objects across all frames.

2. **Settlement Detection**: When all dynamic objects have velocities below configurable thresholds for a specified number of consecutive frames, the scene is considered "settled."

3. **Efficient Rendering**: Only frames up to the settlement point are actually rendered. The remaining frames reuse the last rendered frame, maintaining visual consistency while dramatically reducing computation time.

## Usage

### Enable Efficient Rendering

```bash
python src/run_ball_drop_v2.py --efficient_rendering --frame_end=40
```

### Configure Thresholds

```bash
python src/run_ball_drop_v2.py \
    --efficient_rendering \
    --velocity_threshold=0.001 \
    --angular_velocity_threshold=0.001 \
    --settle_frames=3 \
    --frame_end=40
```

### Disable for Testing

```bash
python src/run_ball_drop_v2.py --disable_efficient_rendering --frame_end=40
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--efficient_rendering` | False | Enable efficient rendering |
| `--velocity_threshold` | 0.001 | Linear velocity threshold (m/s) below which object is considered stationary |
| `--angular_velocity_threshold` | 0.001 | Angular velocity threshold (rad/s) below which object is considered stationary |
| `--settle_frames` | 3 | Number of consecutive frames objects must be stationary to consider scene settled |
| `--disable_efficient_rendering` | False | Explicitly disable efficient rendering (for testing) |

## Performance Benefits

### Example Scenario
- **Total frames**: 40
- **Motion settles at**: frame 15
- **Frames actually rendered**: 16 (0-15)
- **Frames reused**: 24 (16-39)
- **Efficiency gain**: 60% fewer frames rendered

### Expected Time Savings
The time savings depend on when motion settles:
- Motion settles at 25% of total frames: ~75% time reduction
- Motion settles at 50% of total frames: ~50% time reduction  
- Motion settles at 75% of total frames: ~25% time reduction

## Metadata Tracking

The system automatically tracks efficiency metrics in the metadata:

```json
{
  "rendering_efficiency": {
    "settle_frame": 15,
    "frames_rendered": 16,
    "frames_reused": 24,
    "total_frames": 40,
    "efficiency_percent": 60.0,
    "mode": "efficient"
  }
}
```

### Modes
- `"efficient"`: Motion settled, frames were reused
- `"efficient_no_settle"`: Efficient rendering enabled but motion never settled
- `"traditional"`: Traditional rendering used

## Testing

Run the test script to compare efficient vs traditional rendering:

```bash
python test_efficient_rendering.py
```

This will:
1. Run a simulation with traditional rendering
2. Run the same simulation with efficient rendering
3. Compare performance and show efficiency gains
4. Clean up test files

## When to Use

### Best For:
- Object dropping/falling simulations
- Physics simulations that reach equilibrium
- Longer frame sequences (>20 frames)
- Scenes where objects settle quickly

### Not Recommended For:
- Continuous motion scenarios (e.g., rolling objects)
- Short frame sequences (<10 frames)
- Scenes with ongoing collisions
- Camera movement simulations

## Technical Details

### Detection Algorithm
1. Extract velocity and angular velocity data for all dynamic objects from simulation
2. For each frame, check if all dynamic objects have velocities below thresholds
3. Count consecutive frames where all objects are below thresholds
4. When count reaches `settle_frames`, mark scene as settled

### Frame Replication
- Last rendered frame is duplicated using `np.tile()`
- Both RGBA and segmentation data are replicated
- Final data stack maintains original frame count and format

### Safety Features
- Automatic fallback to traditional rendering if motion never settles
- Comprehensive logging of efficiency gains
- Metadata tracking for analysis and debugging

## Troubleshooting

### Motion Never Settles
- Increase `velocity_threshold` and `angular_velocity_threshold`
- Reduce `settle_frames` requirement
- Check for continuous small motions (e.g., vibrations)

### False Early Settlement
- Decrease thresholds for more sensitive detection
- Increase `settle_frames` to require longer stability period

### Performance Issues
- For very long sequences, consider chunked rendering
- Monitor memory usage with frame replication

## Implementation Notes

- Compatible with existing rendering pipeline
- No changes to simulation accuracy
- Maintains visual consistency
- Thread-safe implementation
- Supports all existing output formats (MP4, GIF, images) 