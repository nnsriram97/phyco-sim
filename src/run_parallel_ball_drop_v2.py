import os
import sys
import sys; sys.path = ["kubric"] + sys.path
import multiprocessing as mp
import subprocess
import random
import string
import time
import argparse
from typing import List, Dict, Any
import logging
from loguru import logger
import kubric as kb
import signal
import uuid
import copy
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

def generate_video_id() -> str:
    """Generate a unique 6-letter alphanumeric video ID."""
    return str(uuid.uuid4())[:6]

def get_available_gpus():
    """Get list of available GPU device IDs."""
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        return list(range(gpu_count))
    except:
        # Fallback: try to detect from nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, check=True)
            gpu_count = len([line for line in result.stdout.strip().split('\n') if line.startswith('GPU')])
            return list(range(gpu_count))
        except:
            logger.warning("Could not detect GPUs, assuming 4 GPUs available")
            return [0, 1, 2, 3]  # Default fallback

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def time_limit(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Reset the alarm
            return result
        return wrapper
    return decorator

def setup_worker_logging(worker_id: int):
    """Setup logging for worker process."""
    logger.remove()  # Remove default logger
    logger.add(
        sys.stderr,
        format=f"<green>{{time}}</green> | <level>{{level}}</level> | Worker-{worker_id} | {{message}}",
        level="INFO"
    )

class ParallelBallDropGenerator:
    def __init__(self, base_args):
        self.base_args = base_args
        self.generated_videos = 0
        self.failed_videos = 0
        self.lock = threading.Lock()
        self.active_processes = set()  # Track active subprocesses
        self.executor = None  # Store ThreadPoolExecutor reference
        
        # Get available GPUs
        self.available_gpus = get_available_gpus()
        logger.info(f"Detected {len(self.available_gpus)} GPUs: {self.available_gpus}")
        
        # Create GPU assignment queue
        self.gpu_queue = []
        for gpu_id in self.available_gpus:
            self.gpu_queue.append(gpu_id)
        
        # If we have more workers than GPUs, cycle through GPUs
        while len(self.gpu_queue) < base_args.num_workers:
            self.gpu_queue.extend(self.available_gpus)
        
        logger.info(f"GPU assignment queue: {self.gpu_queue[:base_args.num_workers]}")
        
    def cleanup_processes(self):
        """Clean up all child processes."""
        logger.info("Cleaning up processes...")
        if self.executor:
            self.executor.shutdown(wait=False)
        
        # Kill all tracked processes
        with self.lock:
            for process in self.active_processes:
                try:
                    if process.poll() is None:  # Check if process is still running
                        process.terminate()
                except Exception:
                    pass
            
            # Wait a bit for processes to terminate
            time.sleep(1)
            
            # Force kill any remaining processes
            for process in self.active_processes:
                try:
                    if process.poll() is None:  # Check if process is still running
                        process.kill()
                except Exception:
                    pass
            
            self.active_processes.clear()

    def run_single_video(self, video_id: str, gpu_id: int, ball_restitution: float, platform_restitution: float, composition_style: str) -> bool:
        """Run a single ball drop video generation using subprocess with specific GPU."""
        try:
            logger.info(f"Generating ball drop video {video_id} on GPU {gpu_id}")
            
            # Build command arguments
            cmd = [
                sys.executable,
                "src/run_ball_drop_v2.py",
                f"--video_id={video_id}",
                f"--output_dir={self.base_args.output_dir}",
                f"--scenario={self.base_args.scenario}",
                f"--resolution={self.base_args.resolution}",
                f"--frame_end={self.base_args.frame_end}",
                f"--frame_rate={self.base_args.frame_rate}",
                f"--ball_restitution={ball_restitution}",
                f"--platform_restitution={platform_restitution}",
                f"--platform_friction={self.base_args.platform_friction}",
                f"--ball_friction={self.base_args.ball_friction}",
                f"--velocity_threshold={self.base_args.velocity_threshold}",
                f"--angular_velocity_threshold={self.base_args.angular_velocity_threshold}",
                f"--settle_frames={self.base_args.settle_frames}",
                f"--not_visible_stop_threshold={self.base_args.not_visible_stop_threshold}",
                f"--focal_length={self.base_args.focal_length}",
                f"--sensor_width={self.base_args.sensor_width}",
                f"--max_motion_blur={self.base_args.max_motion_blur}",
                f"--layers={self.base_args.layers}",
                f"--composition_style={composition_style}",
                f"--kubasic_assets={self.base_args.kubasic_assets}",
                f"--hdri_assets={self.base_args.hdri_assets}",
            ]
            
            # Add optional arguments
            if self.base_args.save_mp4:
                cmd.append("--save_mp4")
            if self.base_args.save_gif:
                cmd.append("--save_gif")
            if self.base_args.tar:
                cmd.append("--tar")
            if self.base_args.efficient_rendering:
                cmd.append("--efficient_rendering")
            if self.base_args.force_focal_length:
                cmd.append("--force_focal_length")
            if self.base_args.vary_restitution_only:
                cmd.append("--vary_restitution_only")
            if self.base_args.debug_gui:
                cmd.append("--debug_gui")
            if self.base_args.debug_frustum:
                cmd.append("--debug_frustum")
            
            # Add camera angles if specified
            if hasattr(self.base_args, 'camera_elevation_angle') and self.base_args.camera_elevation_angle is not None:
                cmd.append(f"--camera_elevation_angle={self.base_args.camera_elevation_angle}")
            if hasattr(self.base_args, 'camera_azimuth_angle') and self.base_args.camera_azimuth_angle is not None:
                cmd.append(f"--camera_azimuth_angle={self.base_args.camera_azimuth_angle}")
            
            # Add scene save path if specified
            if hasattr(self.base_args, 'scene_save_path') and self.base_args.scene_save_path:
                cmd.append(f"--scene_save_path={self.base_args.scene_save_path}")
            
            # Set up environment with GPU restriction
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["KUBRIC_USE_GPU"] = "true"
            
            # Run the subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                env=env  # Pass the modified environment
            )
            
            with self.lock:
                self.active_processes.add(process)
            
            try:
                stdout, stderr = process.communicate(timeout=3000)  # 50 minute timeout (as in ball drop script)
                
                if process.returncode == 0:
                    logger.info(f"Successfully generated ball drop video {video_id} on GPU {gpu_id}")
                    with self.lock:
                        self.generated_videos += 1
                    return True
                else:
                    logger.error(f"Failed to generate ball drop video {video_id} on GPU {gpu_id}")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    with self.lock:
                        self.failed_videos += 1
                    return False
                    
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"Ball drop video {video_id} generation timed out on GPU {gpu_id}")
                with self.lock:
                    self.failed_videos += 1
                return False
            finally:
                with self.lock:
                    self.active_processes.discard(process)
                
        except Exception as e:
            logger.error(f"Exception while generating ball drop video {video_id} on GPU {gpu_id}: {e}")
            with self.lock:
                self.failed_videos += 1
            return False
    
    def run_parallel_generation(self, num_workers: int, num_videos: int):
        """Run parallel ball drop video generation using subprocesses with GPU assignment."""
        logger.info(f"Starting parallel ball drop generation: {num_workers} workers, {num_videos} videos")
        logger.info(f"Available GPUs: {self.available_gpus}")
        
        # Update output dir with output_dir/YYYY-MM-DD
        self.base_args.output_dir = os.path.join(self.base_args.output_dir, datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(self.base_args.output_dir, exist_ok=True)
        
        # Save the args to a json file
        with open(os.path.join(self.base_args.output_dir, "args.json"), "w") as f:
            json.dump(self.base_args.__dict__, f, indent=4)

        # Generate unique video IDs
        video_ids = set()
        while len(video_ids) < num_videos:
            tmp_id = generate_video_id()
            if tmp_id not in video_ids and tmp_id not in os.listdir(self.base_args.output_dir):
                video_ids.add(tmp_id)
        
        video_id_list = list(video_ids)

        # Sample ball and platform restitution coefficients
        # Create varied combinations for interesting drop dynamics
        # Distribution: 30% both low, 30% both high, 40% mixed combinations
        restitution_range = np.random.choice([0, 1, 2], size=num_videos, p=[0.35, 0.35, 0.3])
        ball_restitution_list = []
        platform_restitution_list = []
        
        for i, video_id in enumerate(video_id_list):
            if restitution_range[i] == 0:
                # Both low restitution (inelastic bounces)
                ball_restitution_list.append(np.random.uniform(0.01, 0.3))
                platform_restitution_list.append(1.0)
                # platform_restitution_list.append(np.random.uniform(0.01, 0.3))
            elif restitution_range[i] == 1:
                # Both high restitution (elastic bounces)
                ball_restitution_list.append(np.random.uniform(0.8, 0.95))
                platform_restitution_list.append(1.0)
                # platform_restitution_list.append(np.random.uniform(0.8, 0.95))
            else:
                # Mixed restitution values for varied dynamics
                ball_restitution_list.append(np.random.uniform(0.01, 0.95))
                platform_restitution_list.append(1.0)
                # platform_restitution_list.append(np.random.uniform(0.01, 0.95))
        composition_styles = [
            'center',           # Ball in center (original behavior)
            'left_third',       # Ball on left third of frame  
            'right_third',      # Ball on right third of frame
            'upper_left',       # Ball in upper left area
            'upper_right',      # Ball in upper right area
            'lower_center',     # Ball in lower center (more platform visible)
            'lower_left',       # Ball in lower left area
            'lower_right',      # Ball in lower right area
        ]        
        composition_style_list = np.random.choice(composition_styles, size=num_videos)
        # Use ThreadPoolExecutor to manage subprocess execution
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                self.executor = executor
                
                # Submit all tasks with GPU assignments
                future_to_video = {}
                for i, video_id in enumerate(video_id_list):
                    gpu_id = self.gpu_queue[i % len(self.gpu_queue)]  # Cycle through available GPUs
                    future = executor.submit(self.run_single_video, video_id, gpu_id, ball_restitution_list[i], platform_restitution_list[i], composition_style_list[i])
                    future_to_video[future] = (video_id, gpu_id)
                
                # Monitor progress
                completed_count = 0
                for future in as_completed(future_to_video):
                    completed_count += 1
                    video_id, gpu_id = future_to_video[future]
                    
                    try:
                        success = future.result()
                        status = "success" if success else "failed"
                        logger.info(f"Ball drop video {video_id} completed on GPU {gpu_id}: {status}")
                    except Exception as e:
                        logger.error(f"Ball drop video {video_id} failed on GPU {gpu_id} with exception: {e}")
                        with self.lock:
                            self.failed_videos += 1
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = num_videos - completed_count
                    eta = remaining / rate if rate > 0 else float('inf')
                    
                    logger.info(f"Progress: {completed_count}/{num_videos} ball drop videos completed "
                               f"({self.generated_videos} success, {self.failed_videos} failed), "
                               f"Rate: {rate:.2f} videos/sec, ETA: {eta/60:.1f} min")
        finally:
            self.cleanup_processes()
        
        logger.info(f"Parallel ball drop generation completed: {self.generated_videos} successful, "
                   f"{self.failed_videos} failed")

def create_parser():
    """Create argument parser for parallel ball drop video generation."""
    parser = kb.ArgumentParser()
    
    # Arguments forwarded to run_ball_drop_v2.py
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--scenario", type=str, default="ball_drop")
    parser.add_argument("--scene_save_path", type=str, default="")
    parser.add_argument("--hdri_assets", type=str, default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    parser.add_argument("--kubasic_assets", type=str, default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--max_motion_blur", type=float, default=0.0)
    parser.add_argument("--layers", type=str, default="image,segmentation,depth")
    parser.add_argument("--efficient_rendering", action="store_true", default=False)
    parser.add_argument("--velocity_threshold", type=float, default=0.005)
    parser.add_argument("--angular_velocity_threshold", type=float, default=0.05)
    parser.add_argument("--settle_frames", type=int, default=2)
    parser.add_argument("--not_visible_stop_threshold", type=int, default=10)
    parser.add_argument("--focal_length", type=float, default=80.0)
    parser.add_argument("--sensor_width", type=float, default=32.0)
    parser.add_argument("--camera_elevation_angle", type=float, default=None)
    parser.add_argument("--camera_azimuth_angle", type=float, default=None)
    parser.add_argument("--force_focal_length", action="store_true", default=False)
    parser.add_argument("--vary_restitution_only", action="store_true", default=True,
                       help="Keep mass and friction constant, vary only restitution coefficient")
    parser.add_argument("--platform_friction", type=float, default=0.2)
    parser.add_argument("--ball_friction", type=float, default=0.2)
    parser.add_argument("--save_mp4", action="store_true", default=False)
    parser.add_argument("--save_gif", action="store_true", default=False)
    parser.add_argument("--tar", action="store_true", default=False)
    parser.add_argument("--debug_gui", action="store_true", default=False, 
                       help="Enable PyBullet GUI for debugging")
    parser.add_argument("--debug_frustum", action="store_true", default=False,
                       help="Enable camera frustum debugging and detailed logging")

    # Parallel execution arguments
    parser.add_argument("--num_workers", type=int, default=4,
        help="Number of parallel worker processes"
    )
    parser.add_argument("--num_videos", type=int, default=100,
        help="Total number of ball drop videos to generate"
    )
    
    parser.set_defaults(frame_end=15, frame_rate=10, resolution="768x432")
    return parser

if __name__ == "__main__":
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup main process logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time}</green> | <level>{level}</level> | Main | {message}",
        level="INFO"
    )
    
    # Validate arguments
    if args.num_videos <= 0:
        logger.error("num_videos must be greater than 0")
        sys.exit(1)
    
    if args.num_workers <= 0:
        logger.error("num_workers must be greater than 0")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Generating {args.num_videos} ball drop videos using {args.num_workers} workers")
    logger.info(f"Each video will vary ball and platform restitution coefficients with fixed friction")
    logger.info(f"Fixed friction values - Platform: {args.platform_friction}, Ball: {args.ball_friction}")
    
    # Run parallel generation
    generator = ParallelBallDropGenerator(args)
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        generator.cleanup_processes()
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        generator.run_parallel_generation(args.num_workers, args.num_videos)
        logger.info("All ball drop videos generated successfully!")
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        generator.cleanup_processes()
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        generator.cleanup_processes()
        sys.exit(1)
