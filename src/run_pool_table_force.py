import os
import sys; sys.path = ["kubric"] + sys.path
import uuid
import signal
import shutil
import tarfile
import logging
from math import radians
from pathlib import Path
from typing import List

import numpy as np
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from kubric.core import materials
from mathutils import Vector, Quaternion, Euler
from loguru import logger
from skimage import io
import json
import pickle as pkl
from kubric_utils import *
import bpy
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib import patches
import matplotlib.image as mpimg

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

SIM_ASSETS_DIR = os.environ.get("SIM_ASSETS_DIR", "./sim_assets")

WOOD_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "wood_textures")
CONCRETE_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "concrete_textures")
GROUND_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "ground_textures")

POOL_TABLE_ASSET_DIR = "objs/pool_table"
POOL_TABLE_NAME = "pool_table"
POOL_BALL_NAMES = [
    "white_ball",
    "1_ball", "2_ball", "3_ball", "4_ball", "5_ball", "6_ball", "7_ball", "8_ball",
    "9_ball", "10_ball", "11_ball", "12_ball", "13_ball", "14_ball", "15_ball",
]

POOL_TABLE_FRICTION = 0.2
POOL_TABLE_RESTITUTION = 0.5
POOL_BALL_FRICTION = 0.2
POOL_BALL_RESTITUTION = 0.9
POOL_BALL_MASS = 1.0  # kg, kept constant across simulations

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def time_limit(seconds):
    """Decorator to limit execution time of a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

def world_to_camera_coordinates(world_point, camera_position, camera_rotation, focal_length, sensor_width, image_width, image_height):
    """Convert world coordinates to camera/image coordinates.
    
    Args:
        world_point: 3D point in world coordinates [x, y, z]
        camera_position: Camera position in world coordinates [x, y, z]
        camera_rotation: Camera rotation matrix (3x3) or quaternion
        focal_length: Camera focal length in mm
        sensor_width: Camera sensor width in mm
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        tuple: (image_x, image_y, depth) where image_x, image_y are pixel coordinates
    """
    import numpy as np
    from mathutils import Matrix, Vector, Quaternion
    
    # Convert to numpy arrays
    world_point = np.array(world_point)
    camera_position = np.array(camera_position)
    
    # Transform world point to camera space
    # First translate to camera origin
    point_relative = world_point - camera_position
    
    # Get camera transformation matrix from Blender camera
    # The camera's local coordinate system has Z pointing toward the camera (negative viewing direction)
    # and Y pointing up in the image
    
    # If camera_rotation is a quaternion, convert to matrix
    if hasattr(camera_rotation, 'to_matrix'):
        rotation_matrix = np.array(camera_rotation.to_matrix())
    elif isinstance(camera_rotation, (list, tuple, np.ndarray)) and len(camera_rotation) == 4:
        # Assume it's a quaternion [w, x, y, z]
        quat = Quaternion(camera_rotation)
        rotation_matrix = np.array(quat.to_matrix())
    else:
        # Assume it's already a rotation matrix
        rotation_matrix = np.array(camera_rotation)
    
    # Transform point to camera coordinate system
    # In Blender's camera space: X=right, Y=up, Z=toward camera (negative view direction)
    camera_point = rotation_matrix.T @ point_relative
    
    # Check if point is behind camera
    if camera_point[2] >= 0:  # In Blender camera space, negative Z is forward
        print(f"DEBUG: Point behind camera - camera_point[2]={camera_point[2]}")
        return None, None, None
    
    # Project to image plane using pinhole camera model
    # Convert from mm to meters for calculation
    focal_length_m = focal_length / 1000.0
    sensor_width_m = sensor_width / 1000.0
    
    # Calculate sensor height maintaining aspect ratio
    aspect_ratio = image_width / image_height
    sensor_height_m = sensor_width_m / aspect_ratio
    
    # Project to normalized device coordinates (-1 to 1)
    # Note: camera_point[2] is negative (toward camera)
    x_ndc = (camera_point[0] * focal_length_m) / (-camera_point[2] * sensor_width_m / 2.0)
    y_ndc = (camera_point[1] * focal_length_m) / (-camera_point[2] * sensor_height_m / 2.0)
    
    # Convert to pixel coordinates (0 to image_width/height)
    image_x = (x_ndc + 1.0) * 0.5 * image_width
    image_y = (1.0 - y_ndc) * 0.5 * image_height  # Flip Y axis for image coordinates
    
    depth = -camera_point[2]  # Distance from camera (positive)
    
    print(f"DEBUG: world_point={world_point}, camera_point={camera_point}")
    print(f"DEBUG: x_ndc={x_ndc}, y_ndc={y_ndc}")
    print(f"DEBUG: image_x={image_x}, image_y={image_y}, depth={depth}")
    print(f"DEBUG: image_width={image_width}, image_height={image_height}")
    
    return image_x, image_y, depth

def create_force_visualization(image, force_point_world, force_vector_world, camera_position, camera_rotation, 
                              focal_length, sensor_width, force_scale=0.1):
    """Add force visualization to an image.
    
    Args:
        image: RGB image array (H, W, 3)
        force_point_world: 3D point where force is applied in world coordinates
        force_vector_world: 3D force vector in world coordinates
        camera_position: Camera position in world coordinates
        camera_rotation: Camera rotation (quaternion or matrix)
        focal_length: Camera focal length in mm
        sensor_width: Camera sensor width in mm
        force_scale: Scale factor for force vector visualization
    
    Returns:
        tuple: (annotated_image, force_metadata)
    """
    import numpy as np
    import cv2
    
    image_height, image_width = image.shape[:2]
    annotated_image = image.copy()
    
    # Convert force application point to image coordinates
    point_x, point_y, point_depth = world_to_camera_coordinates(
        force_point_world, camera_position, camera_rotation, 
        focal_length, sensor_width, image_width, image_height
    )
    
    force_metadata = {
        "force_point_world": list(force_point_world),
        "force_vector_world": list(force_vector_world),
        "force_magnitude": float(np.linalg.norm(force_vector_world)),
        "camera_position": list(camera_position),
        "focal_length": focal_length,
        "sensor_width": sensor_width
    }
    
    if point_x is None or point_y is None:
        # Point is behind camera or not visible
        force_metadata["visible"] = False
        force_metadata["reason"] = "behind_camera"
        return annotated_image, force_metadata
    
    # Check if point is within image bounds
    if (point_x < 0 or point_x >= image_width or 
        point_y < 0 or point_y >= image_height):
        force_metadata["visible"] = False
        force_metadata["reason"] = "outside_image_bounds"
        force_metadata["image_coordinates"] = [float(point_x), float(point_y)]
        return annotated_image, force_metadata
    
    force_metadata["visible"] = True
    force_metadata["image_coordinates"] = [float(point_x), float(point_y)]
    force_metadata["depth"] = float(point_depth)
    
    # Calculate force vector end point in world coordinates
    force_end_world = np.array(force_point_world) + np.array(force_vector_world) * force_scale
    
    # Convert force vector end point to image coordinates
    end_x, end_y, end_depth = world_to_camera_coordinates(
        force_end_world, camera_position, camera_rotation,
        focal_length, sensor_width, image_width, image_height
    )
    
    if end_x is not None and end_y is not None:
        force_metadata["force_end_image_coordinates"] = [float(end_x), float(end_y)]
        
        # Draw force vector as arrow
        pt1 = (int(point_x), int(point_y))
        pt2 = (int(end_x), int(end_y))
        
        # Draw arrow line
        cv2.arrowedLine(annotated_image, pt1, pt2, (0, 255, 0), 3, tipLength=0.3)
        
        print(f"DEBUG: Drew arrow from {pt1} to {pt2}")
    else:
        force_metadata["force_end_image_coordinates"] = None
        print(f"DEBUG: Could not draw arrow - end point not visible")
    
    # Always add force magnitude text and draw application point
    force_magnitude = np.linalg.norm(force_vector_world)
    text = f"F={force_magnitude:.1f}N"
    text_pos = (int(point_x) + 10, int(point_y) - 10)
    cv2.putText(annotated_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (0, 255, 0), 2)
    
    # Draw force application point as circle (always visible if point is in frame)
    center = (int(point_x), int(point_y))
    cv2.circle(annotated_image, center, 8, (255, 0, 0), 2)  # Red circle
    cv2.circle(annotated_image, center, 3, (255, 255, 255), -1)  # White center
    
    print(f"DEBUG: Drew force point circle at {center}")
    print(f"DEBUG: Added force text '{text}' at {text_pos}")
    
    return annotated_image, force_metadata

# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------

parser = kb.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--video_id", type=str, default=str(uuid.uuid4()))
parser.add_argument("--scene_save_path", type=str, default="")
parser.add_argument("--hdri_assets", type=str, default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--max_motion_blur", type=float, default=0.0)
parser.add_argument("--layers", type=str, default="image,segmentation,depth")
parser.add_argument("--efficient_rendering", action="store_true", default=False)
parser.add_argument("--velocity_threshold", type=float, default=0.005)
parser.add_argument("--angular_velocity_threshold", type=float, default=0.05)
parser.add_argument("--settle_frames", type=int, default=2)
parser.add_argument("--not_visible_stop_threshold", type=int, default=2)
parser.add_argument("--focal_length", type=float, default=80.0)
parser.add_argument("--sensor_width", type=float, default=32.0)
parser.add_argument("--camera_elevation_angle", type=float, default=None)
parser.add_argument("--camera_azimuth_angle", type=float, default=None)
parser.add_argument("--force_focal_length", action="store_true", default=False)
parser.add_argument("--composition_style", type=str, default=None, help="Choose camera style for framing the pool table")
parser.add_argument("--scenario", type=str, default="friction_slide_flat")
parser.add_argument("--ball_friction", type=float, default=None)
parser.add_argument("--ball_restitution", type=float, default=None)
parser.add_argument("--table_friction", type=float, default=None)
parser.add_argument("--table_restitution", type=float, default=None)
parser.add_argument("--force_magnitude", type=float, default=None)
parser.add_argument("--min_force", type=float, default=40.0, help="Minimum force magnitude (N)")
parser.add_argument("--max_force", type=float, default=450.0, help="Maximum force magnitude (N)")
parser.add_argument("--num_balls", type=int, default=None,
                    help="Number of pool balls to include (default: all available)")
parser.add_argument("--default_ball_positions", action="store_true", default=False,
                    help="Place balls in a standard rack layout instead of random positions")
parser.add_argument("--force_to_another_ball", action="store_true", default=False,
                    help="Aim the force from the cue ball (or chosen ball) toward another random ball")

parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--save_gif", action="store_true", default=False)
parser.add_argument("--tar", action="store_true", default=False)
parser.add_argument("--debug_gui", action="store_true", default=False, 
                   help="Enable PyBullet GUI for debugging")
parser.set_defaults(frame_end=15, frame_rate=10, resolution="768x432")
args = parser.parse_args()

# --------------------------------------------------------------------------------------
# Custom PyBullet Simulator with GUI for debugging
# --------------------------------------------------------------------------------------

class PyBulletWithGUI(PyBullet):
    """PyBullet simulator with GUI enabled for debugging."""
    
    def __init__(self, scene, scratch_dir=None):
        import tempfile
        if scratch_dir is None:
            scratch_dir = tempfile.mkdtemp()
        
        # Store these before calling parent constructor
        self.scratch_dir = scratch_dir
        
        # Import pybullet with the same redirect pattern as kubric
        from kubric.redirect_io import RedirectStream
        import sys
        with RedirectStream(stream=sys.stderr):
            import pybullet as pb
        
        # Create the bullet client with GUI mode
        from kubric.simulator.pybullet import _BulletClient
        self._physics_client = _BulletClient(pb.GUI)  # Enable GUI mode
        
        # Set physics parameters (same as original PyBullet class)
        self._physics_client.setPhysicsEngineParameter(
            restitutionVelocityThreshold=0.,
            warmStartingFactor=0.,
            useSplitImpulse=True,
            contactSlop=0.,
            enableConeFriction=False,
            deterministicOverlappingPairs=True
        )
        
        # Initialize the parent View class directly (skip PyBullet.__init__)
        from kubric import core
        core.View.__init__(
            self,
            scene,
            scene_observers={
                "gravity": [
                    lambda change: self._physics_client.setGravity(*change.new)
                ],
            }
        )
        
        # Add GUI debugging features
        self._setup_debug_gui()
    
    def _setup_debug_gui(self):
        """Setup GUI debugging features."""
        # Enable real-time simulation for interactive debugging
        self._physics_client.setRealTimeSimulation(0)
        
        # Add debug information display
        self._physics_client.configureDebugVisualizer(
            self._physics_client.COV_ENABLE_GUI, 1
        )
        self._physics_client.configureDebugVisualizer(
            self._physics_client.COV_ENABLE_SHADOWS, 1
        )
        
        print("\n" + "="*60)
        print("PyBullet GUI DEBUG MODE ENABLED")
        print("="*60)
        print("GUI Controls:")
        print("- Mouse: Rotate view")
        print("- Mouse wheel: Zoom")
        print("- Right panel: Physics parameters")
        print("- 'p' key: Pause/unpause simulation")
        print("- 'r' key: Reset simulation")
        print("- Close GUI window to continue with rendering")
        print("="*60)
    
    def pause_for_inspection(self, message="Paused for inspection"):
        """Pause simulation for manual inspection."""
        print(f"\n{message}")
        print("Press Enter to continue...")
        input()
        
    def step_simulation_slowly(self, steps=100, delay=0.01):
        """Step through simulation slowly for debugging."""
        import time
        print(f"Stepping through {steps} simulation steps slowly...")
        for i in range(steps):
            self._physics_client.stepSimulation()
            time.sleep(delay)
            if i % 20 == 0:
                print(f"Step {i}/{steps}")
        print("Slow stepping complete.")

# --------------------------------------------------------------------------------------
# Main Friction Slide Simulation class
# --------------------------------------------------------------------------------------

class FrictionSlideFlatForceSimulation:
    """Generate pool table force-interaction simulations and render outputs."""
    def __init__(self, args):
        self.args = args
        (
            self.scene,
            self.rng,
            _,
            self.scratch_dir,
            self.renderer,
            self.simulator,
        ) = self._configure_kubric()

        self.video_id = args.video_id
        self.output_dir = os.path.join(args.output_dir, self.video_id)
        os.makedirs(self.output_dir, exist_ok=True)

        self.global_segmentation_id = 0
        self.metadata = {"object_data": {}}
        
        # Store force application data for visualization
        self.applied_forces = []

        # Pool table specific state
        self.pool_table = None
        self.pool_balls = []
        self.pool_play_surface_z = None
        self.pool_play_area_bounds = None
        self.pool_ball_radius = None
        self._pool_geometry_loaded = False

        self.hdri_source = self._load_asset_sources()

    # ------------------------------------------------------------------
    # Kubric setup helpers
    # ------------------------------------------------------------------

    def _configure_kubric(self):
        scene, rng, output_dir, scratch_dir = kb.setup(self.args)
        
        # Set gravity to normal Earth gravity (9.8 m/s²)
        scene.gravity = (0, 0, -9.8)
        logging.info(f"Set gravity to normal Earth gravity: {scene.gravity}")

        # Use GUI simulator if debug flag is enabled
        if self.args.debug_gui:
            simulator = PyBulletWithGUI(scene, scratch_dir)
            logging.info("PyBullet GUI enabled for debugging")
        else:
            simulator = PyBullet(scene, scratch_dir)
        
        simulator._physics_client.setGravity(0, 0, -9.8)
        simulator._physics_client.setRealTimeSimulation(0)
        
        # Render layers
        default_layers, aux_layers = [], []
        arg_layers = self.args.layers.split(",")
        if "image" in arg_layers:
            default_layers.append("Image")
        if "depth" in arg_layers:
            default_layers.append("Depth")
        if "segmentation" in arg_layers:
            aux_layers.append("CryptoObject00")

        renderer = Blender(
            scene,
            scratch_dir,
            use_denoising=True,
            samples_per_pixel=16,
            motion_blur=self.args.max_motion_blur,
            adaptive_sampling=True,
            aux_layers=aux_layers,
            default_layers=default_layers,
        )
        return scene, rng, output_dir, scratch_dir, renderer, simulator

    def _load_asset_sources(self):
        hdri_source = kb.AssetSource.from_manifest(self.args.hdri_assets)
        return hdri_source

    def _get_segmentation_id(self):
        self.global_segmentation_id += 1
        return self.global_segmentation_id

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _setup_background_and_plane(self):
        """Add HDRI lighting and a dome ground plane."""
        hdri_id = self.rng.choice(self.hdri_source.get_test_split(fraction=0.0)[0])
        hdri_tex = self.hdri_source.create(asset_id=hdri_id)
        
        # Generate random rotation for HDRI lighting (in radians)
        # Rotate around X, Y, Z axes with reasonable ranges
        hdri_rotation_x = 0.0 #self.rng.uniform(-np.pi/4, np.pi/4)  # ±45 degrees around X
        hdri_rotation_y = 0.0 #self.rng.uniform(-np.pi/2, np.pi/2)  # ±90 degrees around Y  
        hdri_rotation_z = -np.pi/2 + self.rng.uniform(-np.pi/4, np.pi/4)  # ±30 degrees around Z
        
        hdri_rotation = (hdri_rotation_x, hdri_rotation_y, hdri_rotation_z)
        
        # Set HDRI for both lighting and background
        self.renderer._set_ambient_light_hdri(hdri_tex.filename, hdri_rotation=hdri_rotation)
        self.renderer._set_background_hdri(hdri_tex.filename, hdri_rotation=hdri_rotation)
        
        # Disable background transparency to show HDRI background
        self.renderer.background_transparency = False
        
        # Camera depth of field will be applied later during camera setup
        
        self.metadata["hdri_id"] = hdri_id
        self.metadata["hdri_rotation"] = {
            "x_radians": hdri_rotation_x,
            "y_radians": hdri_rotation_y, 
            "z_radians": hdri_rotation_z,
            "x_degrees": np.degrees(hdri_rotation_x),
            "y_degrees": np.degrees(hdri_rotation_y),
            "z_degrees": np.degrees(hdri_rotation_z)
        }

        platform_friction = 1.0
        platform_restitution = 0.0
        
        # Use a simple as the ground plane
        ground_plane = kb.FileBasedObject(
            name="ground_plane",
            simulation_filename="objs/plane.urdf",
            render_filename="objs/plane.obj",
            scale=1.0,
            position=(0, 0, 0),
            friction=platform_friction,
            restitution=platform_restitution,
            static=True, 
            background=True,
            segmentation_id=self._get_segmentation_id()
        )

        # In the blender scene I want to add texture to this ground plane with normal, displacement maps and others
        

        self.scene += ground_plane
        
        # Store dome object for later metadata collection
        self.dome = ground_plane
        
        # Apply ground textures to the plane
        self._apply_ground_textures()
        

    def _apply_ground_textures(self):
        """Apply randomly selected ground textures to the plane"""
        import os
        import random
        
        
        # Select wood or concrete texture
        texture_type = self.rng.choice(['wood', 'concrete'])
        if texture_type == 'wood':
            texture_base_path = WOOD_TEXTURE_BASE_PATH
        elif texture_type == 'concrete':
            texture_base_path = CONCRETE_TEXTURE_BASE_PATH
        
        # Access the ground plane's Blender object through Kubric's linked objects
        if not hasattr(self, 'dome') or not self.dome:
            logging.warning("Warning: Ground plane object not available for texture application")
            return False
            
        # Get the Blender representation of the ground plane
        ground_plane_blender = self.dome.linked_objects[self.renderer]
        
        if not ground_plane_blender:
            logging.warning("Warning: Ground plane Blender object not found")
            return False
        
        logging.info(f"Found ground plane Blender object for texturing: {ground_plane_blender.name}")
        
        # Get available ground texture directories
        ground_types = []
        try:
            ground_types = [d for d in os.listdir(texture_base_path) 
                           if os.path.isdir(os.path.join(texture_base_path, d))]
        except Exception as e:
            logging.error(f"Error reading ground texture directories: {e}")
            return False
        
        if not ground_types:
            logging.error(f"No ground texture directories found in {texture_base_path}")
            return False
        
        # Randomly select a ground type using the simulation's RNG for reproducibility
        selected_ground = self.rng.choice(ground_types)
        texture_path = os.path.join(texture_base_path, selected_ground, "textures")
        
        # Store selected texture info in metadata
        self.metadata["ground_texture"] = selected_ground.split(".blend")[0]
        
        logging.info(f"Selected ground texture: {selected_ground}")
        
        # Find texture files with different possible extensions
        diffuse_path = None
        normal_path = None
        roughness_path = None
        displacement_path = None
        
        try:
            # Get all files in the texture directory
            all_files = os.listdir(texture_path)
            
            # Find textures by identifying patterns in filenames
            for file in all_files:
                file_lower = file.lower()
                # Diffuse texture (always jpg)
                if "diff_4k" in file_lower and file_lower.endswith(".jpg"):
                    diffuse_path = os.path.join(texture_path, file)
                
                # Normal texture (always exr)
                elif "nor_gl_4k" in file_lower and file_lower.endswith(".exr"):
                    normal_path = os.path.join(texture_path, file)
                
                # Roughness texture (could be jpg or exr)
                elif "rough_4k" in file_lower:
                    if file_lower.endswith(".jpg") or file_lower.endswith(".exr"):
                        roughness_path = os.path.join(texture_path, file)
                
                # Displacement texture (could be jpg or png)
                elif "disp_4k" in file_lower:
                    if file_lower.endswith(".jpg") or file_lower.endswith(".png"):
                        displacement_path = os.path.join(texture_path, file)
        
        except Exception as e:
            logging.error(f"Error finding texture files: {e}")
            return False
        
        # Print found textures for debugging
        logging.info(f"Found diffuse: {os.path.basename(diffuse_path) if diffuse_path else 'None'}")
        logging.info(f"Found normal: {os.path.basename(normal_path) if normal_path else 'None'}")
        logging.info(f"Found roughness: {os.path.basename(roughness_path) if roughness_path else 'None'}")
        logging.info(f"Found displacement: {os.path.basename(displacement_path) if displacement_path else 'None'}")
        
        # Get or create the material for the plane
        if len(ground_plane_blender.material_slots) == 0:
            # Create new material if none exists
            mat = bpy.data.materials.new(name="Ground_Material")
            ground_plane_blender.data.materials.append(mat)
        else:
            mat = ground_plane_blender.material_slots[0].material
            if not mat:
                mat = bpy.data.materials.new(name="Ground_Material")
                ground_plane_blender.material_slots[0].material = mat
        
        # Enable nodes for the material
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)
        
        # Create the principled BSDF shader
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        
        # Create output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # Link principled to output
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # Add texture coordinate and mapping nodes for better control
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-800, 0)
        
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-600, 0)
        # Adjust scale to control texture tiling (smaller scale for larger tiles)
        mapping.inputs['Scale'].default_value[0] = 5.0  # Scale X
        mapping.inputs['Scale'].default_value[1] = 5.0  # Scale Y
        mapping.inputs['Scale'].default_value[2] = 1.0  # Scale Z
        
        # Use Generated coordinates instead of UV (since plane.obj has no UV data)
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        
        # Create and link texture nodes for diffuse, normal, roughness, and displacement
        # Diffuse Texture
        if diffuse_path and os.path.exists(diffuse_path):
            tex_diffuse = nodes.new(type='ShaderNodeTexImage')
            tex_diffuse.location = (-400, 200)
            tex_diffuse.image = bpy.data.images.load(diffuse_path)
            # Use box projection to avoid planar-from-top stretching
            tex_diffuse.projection = 'BOX'
            tex_diffuse.projection_blend = 0.15
            links.new(mapping.outputs['Vector'], tex_diffuse.inputs['Vector'])
            links.new(tex_diffuse.outputs['Color'], principled.inputs['Base Color'])
            logging.info(f"Applied ground diffuse texture: {diffuse_path}")
        else:
            logging.warning(f"Warning: Ground diffuse texture not found")
        
        # Normal Texture
        if normal_path and os.path.exists(normal_path):
            tex_normal = nodes.new(type='ShaderNodeTexImage')
            tex_normal.location = (-400, 0)
            tex_normal.image = bpy.data.images.load(normal_path)
            # Set correct color space for normal maps
            tex_normal.image.colorspace_settings.name = 'Non-Color'
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-200, 0)
            links.new(mapping.outputs['Vector'], tex_normal.inputs['Vector'])
            links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
            logging.info(f"Applied ground normal texture: {normal_path}")
        else:
            logging.warning(f"Warning: Ground normal texture not found")
        
        # Roughness Texture
        if roughness_path and os.path.exists(roughness_path):
            tex_roughness = nodes.new(type='ShaderNodeTexImage')
            tex_roughness.location = (-400, -200)
            tex_roughness.image = bpy.data.images.load(roughness_path)
            # Set correct color space for roughness maps
            tex_roughness.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_roughness.inputs['Vector'])
            links.new(tex_roughness.outputs['Color'], principled.inputs['Roughness'])
            logging.info(f"Applied ground roughness texture: {roughness_path}")
        else:
            logging.warning(f"Warning: Ground roughness texture not found")
        
        # Displacement Texture
        if displacement_path and os.path.exists(displacement_path):
            tex_disp = nodes.new(type='ShaderNodeTexImage')
            tex_disp.location = (-400, -400)
            tex_disp.image = bpy.data.images.load(displacement_path)
            # Set correct color space for displacement maps
            tex_disp.image.colorspace_settings.name = 'Non-Color'
            
            # Add a displacement node
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = (-200, -400)
            disp_node.inputs['Scale'].default_value = 0.05  # Reduced displacement strength
            
            links.new(mapping.outputs['Vector'], tex_disp.inputs['Vector'])
            links.new(tex_disp.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
            
            # # Set up the plane for displacement
            # # Subdivide the plane for better displacement detail without modifying object boundaries
            # if ground_plane_blender.modifiers.get("Subdivision") is None:
            #     subdiv = ground_plane_blender.modifiers.new(name="Subdivision", type='SUBSURF')
            #     subdiv.levels = 4  # Increased subdivision for better displacement
            #     subdiv.render_levels = 4
            #     subdiv.boundary_smooth = 'PRESERVE_CORNERS'  # Preserve boundaries
            #     subdiv.use_limit_surface = True  # Use limit surface for better preservation
            
            # Enable displacement in material settings
            mat.cycles.displacement_method = 'BOTH'  # Using both displacement and bump
            
            logging.info(f"Applied ground displacement texture: {displacement_path}")
        else:
            logging.warning(f"Warning: Ground displacement texture not found")

        # Physics properties are handled by Kubric/PyBullet separately
        # No need to set up Blender rigid body physics
        logging.info("Ground texture application completed successfully")
        
        return True

    def _read_urdf_origin_position(self, urdf_path: Path):
        """Read the origin offset from a URDF file and return it as a position tuple."""
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(str(urdf_path))
            root = tree.getroot()
            origin_elem = root.find(".//link/inertial/origin")
            if origin_elem is not None and "xyz" in origin_elem.attrib:
                xyz_str = origin_elem.attrib["xyz"]
                xyz = tuple(float(x) for x in xyz_str.strip().split())
                return xyz
        except FileNotFoundError:
            logging.warning(f"URDF file not found: {urdf_path}")
        except Exception as exc:
            logging.warning(f"Failed to read URDF origin from {urdf_path}: {exc}")

        return None

    def _load_pool_geometry_info(self):
        """Load and cache geometry information required for pool table layout."""
        if self._pool_geometry_loaded:
            return

        table_obj_path = Path(POOL_TABLE_ASSET_DIR) / f"{POOL_TABLE_NAME}.obj"
        try:
            table_vertices = []
            with open(table_obj_path, "r") as obj_file:
                for line in obj_file:
                    if line.startswith("v "):
                        _, x_str, y_str, z_str, *rest = line.split()
                        table_vertices.append((float(x_str), float(y_str), float(z_str)))
            if not table_vertices:
                raise ValueError("No vertices found in pool table mesh")
            table_vertices = np.array(table_vertices)
        except Exception as exc:
            logging.error(f"Failed to load pool table geometry from {table_obj_path}: {exc}")
            raise

        max_z = table_vertices[:, 2].max()
        rounded_z = np.round(table_vertices[:, 2], 3)
        unique_z, counts = np.unique(rounded_z, return_counts=True)

        playing_mask = unique_z < (max_z - 0.05)
        if np.any(playing_mask):
            playing_z_candidates = unique_z[playing_mask]
            playing_counts = counts[playing_mask]
            playing_z = float(playing_z_candidates[np.argmax(playing_counts)])
        else:
            playing_z = float(unique_z[np.argmax(counts)])

        playing_surface_mask = np.isclose(rounded_z, playing_z, atol=1e-3)
        playing_vertices = table_vertices[playing_surface_mask]
        if playing_vertices.size == 0:
            raise RuntimeError("Unable to identify playing surface vertices for pool table")

        min_xy = playing_vertices[:, :2].min(axis=0)
        max_xy = playing_vertices[:, :2].max(axis=0)

        ball_obj_path = Path(POOL_TABLE_ASSET_DIR) / "white_ball.obj"
        try:
            ball_vertices = []
            with open(ball_obj_path, "r") as obj_file:
                for line in obj_file:
                    if line.startswith("v "):
                        _, x_str, y_str, z_str, *rest = line.split()
                        ball_vertices.append((float(x_str), float(y_str), float(z_str)))
            if not ball_vertices:
                raise ValueError("No vertices found in pool ball mesh")
            ball_vertices = np.array(ball_vertices)
        except Exception as exc:
            logging.error(f"Failed to load pool ball geometry from {ball_obj_path}: {exc}")
            raise

        ball_spans = ball_vertices.max(axis=0) - ball_vertices.min(axis=0)
        ball_radius = float(np.max(ball_spans) / 2.0)

        self.pool_play_surface_z = playing_z
        self.pool_play_area_bounds = (min_xy, max_xy)
        self.pool_ball_radius = ball_radius
        self._pool_geometry_loaded = True

        logging.info(
            "Loaded pool geometry: playing_surface_z=%.3f, bounds=(%s, %s), ball_radius=%.4f",
            self.pool_play_surface_z,
            min_xy.tolist(),
            max_xy.tolist(),
            self.pool_ball_radius,
        )

        self.metadata["pool_play_area_bounds"] = {
            "min": min_xy.tolist(),
            "max": max_xy.tolist(),
        }

    def _ensure_object_ground_clearance(self, obj, clearance=0.002):
        """Lift an object so its lowest point sits above the ground plane."""
        try:
            min_corner, _ = get_world_object_bounds(obj)
        except Exception as exc:
            logging.warning(f"Could not compute bounds for {obj.name}: {exc}")
            return

        if min_corner[2] >= clearance:
            return

        delta = clearance - float(min_corner[2])
        new_position = np.array(obj.position) + np.array([0.0, 0.0, delta])
        obj.position = tuple(new_position)

        if hasattr(obj, "linked_objects"):
            blender_obj = obj.linked_objects.get(self.renderer)
            if blender_obj is not None:
                blender_obj.location[2] += delta

        logging.info(
            f"Raised object {obj.name} by {delta:.4f}m to maintain ground clearance"
        )

    def _apply_pool_table_materials(self, table):
        """Enhance pool table shading with sophisticated texture-driven materials mimicking Blender setups."""
        if not hasattr(table, "linked_objects"):
            return

        blender_obj = table.linked_objects.get(self.renderer)
        if blender_obj is None:
            logging.warning("Pool table blender object not found for material adjustment")
            return

        # Get available wood textures from external directory
        wood_texture_dirs = self._get_available_wood_textures()
        
        # Asset directory for fallback textures
        asset_dir = Path(POOL_TABLE_ASSET_DIR)
        if not asset_dir.is_absolute():
            asset_dir = (Path(__file__).resolve().parent.parent / asset_dir).resolve()

        # Helper functions
        def load_image(path):
            if path is None:
                return None
            if not isinstance(path, Path):
                path = Path(path)
            if not path.exists():
                logging.warning(f"Missing texture for pool table material: {path}")
                return None
            try:
                return bpy.data.images.load(str(path), check_existing=True)
            except RuntimeError as exc:
                logging.warning(f"Failed to load texture {path}: {exc}")
                return None

        def clear_socket_links(node_tree, socket):
            for link in list(socket.links):
                node_tree.links.remove(link)

        def get_or_create_node(nodes, node_type, name, label=None, location=None):
            node = nodes.get(name)
            if node is None:
                node = nodes.new(type=node_type)
                node.name = name
                node.label = label or name
                if location:
                    node.location = location
            return node

        def ensure_link(node_tree, from_socket, to_socket):
            for link in to_socket.links:
                if link.from_socket == from_socket:
                    return
            node_tree.links.new(from_socket, to_socket)

        def setup_texture_coordinate_system(nodes, node_tree, mat_name):
            """Setup texture coordinate system with mapping node."""
            tex_coord = get_or_create_node(nodes, 'ShaderNodeTexCoord', f"{mat_name}_TexCoord", location=(-800, 0))
            mapping = get_or_create_node(nodes, 'ShaderNodeMapping', f"{mat_name}_Mapping", location=(-600, 0))
            
            # Configure mapping node
            if hasattr(mapping, "vector_type"):
                mapping.vector_type = 'POINT'
                
            # Use Generated coordinates for better UV mapping on pool table geometry
            clear_socket_links(node_tree, mapping.inputs['Vector'])
            try:
                tex_coord_socket = tex_coord.outputs['Generated']  # Better for complex geometry
            except (KeyError, AttributeError):
                tex_coord_socket = tex_coord.outputs[0]
            ensure_link(node_tree, tex_coord_socket, mapping.inputs['Vector'])
            
            return tex_coord, mapping

        def create_advanced_wood_material(nodes, node_tree, mat, wood_textures):
            """Create advanced wood material with multiple texture maps and mixing."""
            principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
            if principled is None:
                principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                principled.location = (0, 0)
                
            output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if output is None:
                output = nodes.new(type='ShaderNodeOutputMaterial')
                output.location = (300, 0)
            ensure_link(node_tree, principled.outputs['BSDF'], output.inputs['Surface'])

            # Clear existing nodes (keep principled and output)
            for node in list(nodes):
                if node not in [principled, output]:
                    nodes.remove(node)

            # Setup coordinate system
            tex_coord, mapping = setup_texture_coordinate_system(nodes, node_tree, mat.name)
            
            # Adjust mapping scale for pool table (smaller scale = larger texture tiles)
            mapping.inputs['Scale'].default_value[0] = self.rng.uniform(0.8, 1.5)
            mapping.inputs['Scale'].default_value[1] = self.rng.uniform(0.8, 1.5)
            mapping.inputs['Scale'].default_value[2] = 1.0
            
            # Random rotation for texture variation
            mapping.inputs['Rotation'].default_value[2] = self.rng.uniform(0, np.pi/2)

            # Load texture images
            diffuse_image = load_image(wood_textures.get('diffuse'))
            normal_image = load_image(wood_textures.get('normal'))
            roughness_image = load_image(wood_textures.get('roughness'))
            displacement_image = load_image(wood_textures.get('displacement'))
            ao_image = load_image(wood_textures.get('ao'))  # Ambient occlusion

            # Create diffuse texture node
            if diffuse_image is not None:
                diffuse_node = get_or_create_node(nodes, 'ShaderNodeTexImage', f"{mat.name}_Diffuse", location=(-400, 300))
                diffuse_node.image = diffuse_image
                diffuse_node.interpolation = 'Smart'
                diffuse_node.extension = 'REPEAT'
                ensure_link(node_tree, mapping.outputs['Vector'], diffuse_node.inputs['Vector'])
                
                # Add color variation using ColorRamp
                color_ramp = get_or_create_node(nodes, 'ShaderNodeValToRGB', f"{mat.name}_ColorVariation", location=(-200, 300))
                if len(color_ramp.color_ramp.elements) >= 2:
                    color_ramp.color_ramp.elements[0].position = 0.0
                    color_ramp.color_ramp.elements[0].color = (0.8, 0.6, 0.4, 1.0)  # Lighter wood
                    color_ramp.color_ramp.elements[1].position = 1.0
                    color_ramp.color_ramp.elements[1].color = (0.3, 0.2, 0.1, 1.0)  # Darker wood
                
                # Mix diffuse with color variation
                mix_node = get_or_create_node(nodes, 'ShaderNodeMixRGB', f"{mat.name}_ColorMix", location=(-100, 250))
                if hasattr(mix_node, 'blend_type'):
                    mix_node.blend_type = 'MULTIPLY'
                mix_node.inputs['Fac'].default_value = 0.3
                
                ensure_link(node_tree, diffuse_node.outputs['Color'], mix_node.inputs['Color1'])
                ensure_link(node_tree, color_ramp.outputs['Color'], mix_node.inputs['Color2'])
                ensure_link(node_tree, mix_node.outputs['Color'], principled.inputs['Base Color'])

            # Create normal map setup
            if normal_image is not None:
                normal_tex = get_or_create_node(nodes, 'ShaderNodeTexImage', f"{mat.name}_Normal", location=(-400, 0))
                normal_tex.image = normal_image
                normal_tex.image.colorspace_settings.name = 'Non-Color'
                normal_tex.interpolation = 'Smart'
                normal_tex.extension = 'REPEAT'
                ensure_link(node_tree, mapping.outputs['Vector'], normal_tex.inputs['Vector'])
                
                normal_map = get_or_create_node(nodes, 'ShaderNodeNormalMap', f"{mat.name}_NormalMap", location=(-200, 0))
                normal_map.inputs['Strength'].default_value = self.rng.uniform(0.8, 1.5)
                ensure_link(node_tree, normal_tex.outputs['Color'], normal_map.inputs['Color'])
                ensure_link(node_tree, normal_map.outputs['Normal'], principled.inputs['Normal'])

            # Create roughness setup with variation
            if roughness_image is not None:
                rough_tex = get_or_create_node(nodes, 'ShaderNodeTexImage', f"{mat.name}_Roughness", location=(-400, -200))
                rough_tex.image = roughness_image
                rough_tex.image.colorspace_settings.name = 'Non-Color'
                rough_tex.interpolation = 'Smart'
                rough_tex.extension = 'REPEAT'
                ensure_link(node_tree, mapping.outputs['Vector'], rough_tex.inputs['Vector'])
                
                # Add roughness variation with ColorRamp
                rough_ramp = get_or_create_node(nodes, 'ShaderNodeValToRGB', f"{mat.name}_RoughRamp", location=(-200, -200))
                if len(rough_ramp.color_ramp.elements) >= 2:
                    rough_ramp.color_ramp.elements[0].position = 0.0
                    rough_ramp.color_ramp.elements[0].color = (0.1, 0.1, 0.1, 1.0)  # Smooth areas
                    rough_ramp.color_ramp.elements[1].position = 1.0
                    rough_ramp.color_ramp.elements[1].color = (0.7, 0.7, 0.7, 1.0)  # Rough areas
                
                ensure_link(node_tree, rough_tex.outputs['Color'], rough_ramp.inputs['Fac'])
                ensure_link(node_tree, rough_ramp.outputs['Color'], principled.inputs['Roughness'])
            else:
                # Fallback roughness value
                principled.inputs['Roughness'].default_value = self.rng.uniform(0.3, 0.6)

            # Add ambient occlusion for depth
            if ao_image is not None:
                ao_tex = get_or_create_node(nodes, 'ShaderNodeTexImage', f"{mat.name}_AO", location=(-400, -400))
                ao_tex.image = ao_image
                ao_tex.image.colorspace_settings.name = 'Non-Color'
                ao_tex.interpolation = 'Smart'
                ao_tex.extension = 'REPEAT'
                ensure_link(node_tree, mapping.outputs['Vector'], ao_tex.inputs['Vector'])
                
                # Mix AO with base color for subtle darkening
                ao_mix = get_or_create_node(nodes, 'ShaderNodeMixRGB', f"{mat.name}_AOMix", location=(-100, -100))
                if hasattr(ao_mix, 'blend_type'):
                    ao_mix.blend_type = 'MULTIPLY'
                ao_mix.inputs['Fac'].default_value = 0.4
                
                # Connect AO to darken the base color slightly
                if diffuse_image is not None:
                    # Insert AO mixing before the final color output
                    clear_socket_links(node_tree, principled.inputs['Base Color'])
                    ensure_link(node_tree, mix_node.outputs['Color'], ao_mix.inputs['Color1'])
                    ensure_link(node_tree, ao_tex.outputs['Color'], ao_mix.inputs['Color2'])
                    ensure_link(node_tree, ao_mix.outputs['Color'], principled.inputs['Base Color'])

            # Add displacement for micro-detail (subtle)
            if displacement_image is not None:
                disp_tex = get_or_create_node(nodes, 'ShaderNodeTexImage', f"{mat.name}_Displacement", location=(-400, -600))
                disp_tex.image = displacement_image
                disp_tex.image.colorspace_settings.name = 'Non-Color'
                disp_tex.interpolation = 'Smart'
                disp_tex.extension = 'REPEAT'
                ensure_link(node_tree, mapping.outputs['Vector'], disp_tex.inputs['Vector'])
                
                disp_node = get_or_create_node(nodes, 'ShaderNodeDisplacement', f"{mat.name}_DispNode", location=(-200, -600))
                disp_node.inputs['Scale'].default_value = 0.002  # Very subtle displacement
                ensure_link(node_tree, disp_tex.outputs['Color'], disp_node.inputs['Height'])
                ensure_link(node_tree, disp_node.outputs['Displacement'], output.inputs['Displacement'])
                
                # Enable displacement in material settings
                mat.cycles.displacement_method = 'DISPLACEMENT'

            # Set material properties
            principled.inputs['Specular'].default_value = self.rng.uniform(0.3, 0.5)
            if not roughness_image:
                principled.inputs['Roughness'].default_value = self.rng.uniform(0.3, 0.6)
            principled.inputs['IOR'].default_value = 1.45  # Typical for wood finish
            
            logging.info(f"Applied advanced wood material to {mat.name} with {len(wood_textures)} texture maps")

        def create_enhanced_felt_material(nodes, node_tree, mat):
            """Create enhanced felt material with proper fabric characteristics."""
            principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
            if principled is None:
                principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                principled.location = (0, 0)
                
            output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if output is None:
                output = nodes.new(type='ShaderNodeOutputMaterial')
                output.location = (300, 0)
            ensure_link(node_tree, principled.outputs['BSDF'], output.inputs['Surface'])

            # Clear existing nodes
            for node in list(nodes):
                if node not in [principled, output]:
                    nodes.remove(node)

            # Create procedural felt texture
            tex_coord, mapping = setup_texture_coordinate_system(nodes, node_tree, mat.name)
            mapping.inputs['Scale'].default_value[0] = 50.0  # Fine fabric texture
            mapping.inputs['Scale'].default_value[1] = 50.0
            
            # Create noise texture for fabric variation
            noise_tex = get_or_create_node(nodes, 'ShaderNodeTexNoise', f"{mat.name}_Noise", location=(-400, 0))
            noise_tex.inputs['Scale'].default_value = 25.0
            noise_tex.inputs['Detail'].default_value = 8.0
            noise_tex.inputs['Roughness'].default_value = 0.6
            ensure_link(node_tree, mapping.outputs['Vector'], noise_tex.inputs['Vector'])
            
            # Color variation for felt
            base_colors = [
                (0.016, 0.28, 0.012, 1.0),  # Traditional green
                (0.12, 0.05, 0.05, 1.0),    # Deep red
                (0.05, 0.05, 0.12, 1.0),    # Navy blue
                (0.08, 0.05, 0.08, 1.0),    # Deep purple
                (0.18, 0.18, 0.18, 1.0),    # Charcoal/black
                (0.25, 0.18, 0.05, 1.0),    # Brown/tan
                (0.02, 0.18, 0.18, 1.0),    # Teal
                (0.22, 0.22, 0.05, 1.0),    # Olive
                (0.25, 0.25, 0.25, 1.0),    # Light gray
                (0.25, 0.05, 0.18, 1.0),    # Burgundy/magenta
            ]
            # Use randint to select index, then get the color
            color_index = self.rng.randint(0, len(base_colors))
            chosen_color = base_colors[color_index]
            
            color_ramp = get_or_create_node(nodes, 'ShaderNodeValToRGB', f"{mat.name}_ColorRamp", location=(-200, 0))
            if len(color_ramp.color_ramp.elements) >= 2:
                color_ramp.color_ramp.elements[0].position = 0.0
                color_ramp.color_ramp.elements[0].color = chosen_color
                # Slightly lighter variation
                lighter_color = tuple(min(1.0, c * 1.2) if i < 3 else c for i, c in enumerate(chosen_color))
                color_ramp.color_ramp.elements[1].position = 1.0
                color_ramp.color_ramp.elements[1].color = lighter_color
            
            ensure_link(node_tree, noise_tex.outputs['Fac'], color_ramp.inputs['Fac'])
            ensure_link(node_tree, color_ramp.outputs['Color'], principled.inputs['Base Color'])
            
            # Felt material properties
            principled.inputs['Specular'].default_value = 0.1  # Very low specularity
            principled.inputs['Roughness'].default_value = 0.9  # Very rough
            principled.inputs['Sheen'].default_value = 0.3  # Fabric sheen
            principled.inputs['Sheen Tint'].default_value = 0.8
            
            logging.info(f"Applied enhanced felt material to {mat.name} with color {chosen_color[:3]}")

        # Apply materials to each slot
        for slot in blender_obj.material_slots:
            mat = slot.material
            if mat is None:
                mat = bpy.data.materials.new(name="PoolTableMaterial")
                slot.material = mat

            mat.use_nodes = True
            node_tree = mat.node_tree
            nodes = node_tree.nodes
            name_lower = mat.name.lower()

            # Enhanced material application based on material name
            if "felt" in name_lower or "fabric" in name_lower:
                create_enhanced_felt_material(nodes, node_tree, mat)
                
            elif "wood" in name_lower:
                # Select random wood texture set
                if wood_texture_dirs:
                    selected_wood = self.rng.choice(wood_texture_dirs)
                    wood_textures = self._get_wood_texture_paths(selected_wood)
                    create_advanced_wood_material(nodes, node_tree, mat, wood_textures)
                else:
                    # Fallback to original simple wood material
                    self._apply_simple_wood_material(nodes, node_tree, mat, asset_dir)
                    
            elif "metal" in name_lower:
                self._apply_metal_material(nodes, node_tree, mat)
                
            elif "plastic" in name_lower or "pocket" in name_lower:
                self._apply_plastic_material(nodes, node_tree, mat)
                
            elif "linen" in name_lower:
                self._apply_linen_material(nodes, node_tree, mat)
                
            else:
                # Default material
                principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
                if principled:
                    principled.inputs['Roughness'].default_value = 0.5
                    principled.inputs['Specular'].default_value = 0.25

        logging.info("Applied enhanced pool table materials with Blender-style complexity")

    def _get_available_wood_textures(self):
        """Get list of available wood texture directories."""
        wood_texture_dirs = []
        try:
            if os.path.exists(WOOD_TEXTURE_BASE_PATH):
                wood_texture_dirs = [d for d in os.listdir(WOOD_TEXTURE_BASE_PATH) 
                                   if os.path.isdir(os.path.join(WOOD_TEXTURE_BASE_PATH, d)) 
                                   and d.endswith('.blend')]
                logging.info(f"Found {len(wood_texture_dirs)} wood texture directories")
            else:
                logging.warning(f"Wood texture base path not found: {WOOD_TEXTURE_BASE_PATH}")
        except Exception as e:
            logging.error(f"Error reading wood texture directories: {e}")
        return wood_texture_dirs

    def _get_wood_texture_paths(self, wood_dir_name):
        """Get paths to all texture files for a specific wood texture."""
        texture_dir = os.path.join(WOOD_TEXTURE_BASE_PATH, wood_dir_name, "textures")
        textures = {}
        
        if not os.path.exists(texture_dir):
            logging.warning(f"Texture directory not found: {texture_dir}")
            return textures
            
        try:
            files = os.listdir(texture_dir)
            base_name = wood_dir_name.replace('_4k.blend', '').replace('.blend', '')
            
            # Map texture types to their common naming patterns
            texture_patterns = {
                'diffuse': ['diff_4k', 'col_4k', 'color_4k'],
                'normal': ['nor_gl_4k', 'nrm_4k', 'normal_4k'],
                'roughness': ['rough_4k', 'refl_4k', 'roughness_4k'],
                'displacement': ['disp_4k', 'height_4k', 'displacement_4k'],
                'ao': ['ao_4k', 'ambient_4k', 'occlusion_4k']
            }
            
            for tex_type, patterns in texture_patterns.items():
                for pattern in patterns:
                    for file in files:
                        if pattern in file.lower() and (file.lower().endswith('.jpg') or 
                                                       file.lower().endswith('.png') or 
                                                       file.lower().endswith('.exr')):
                            texture_path = Path(texture_dir) / file
                            if texture_path.exists():  # Verify the file actually exists
                                textures[tex_type] = texture_path
                            break
                    if tex_type in textures:
                        break
                        
            logging.info(f"Found textures for {base_name}: {list(textures.keys())}")
            
            # Log missing textures for debugging
            missing_textures = [tex_type for tex_type in texture_patterns.keys() if tex_type not in textures]
            if missing_textures:
                logging.info(f"Missing textures for {base_name}: {missing_textures}")
            
        except Exception as e:
            logging.error(f"Error finding texture files in {texture_dir}: {e}")
            
        return textures

    def _apply_simple_wood_material(self, nodes, node_tree, mat, asset_dir):
        """Apply simple wood material as fallback."""
        principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
        if principled is None:
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            
        # Use fallback textures from asset directory
        wood_color_path = asset_dir / "WoodFineDark004_COL_3K.jpg"
        wood_normal_path = asset_dir / "WoodFineDark004_NRM_3K.jpg"
        wood_roughness_path = asset_dir / "WoodFineDark004_REFL_3K.jpg"
        
        # Apply basic wood setup (simplified version of original code)
        principled.inputs['Base Color'].default_value = (0.4, 0.25, 0.15, 1.0)
        principled.inputs['Roughness'].default_value = 0.4
        principled.inputs['Specular'].default_value = 0.35

    def _apply_metal_material(self, nodes, node_tree, mat):
        """Apply enhanced metal material."""
        principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
        if principled is None:
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            
        principled.inputs['Base Color'].default_value = (0.55, 0.55, 0.55, 1.0)
        principled.inputs['Metallic'].default_value = 0.9
        principled.inputs['Roughness'].default_value = 0.25
        principled.inputs['Specular'].default_value = 0.5

    def _apply_plastic_material(self, nodes, node_tree, mat):
        """Apply enhanced plastic material."""
        principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
        if principled is None:
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            
        principled.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1.0)
        principled.inputs['Roughness'].default_value = 0.35
        principled.inputs['Specular'].default_value = 0.4

    def _apply_linen_material(self, nodes, node_tree, mat):
        """Apply enhanced linen material."""
        principled = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
        if principled is None:
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            
        principled.inputs['Base Color'].default_value = (0.84, 0.82, 0.79, 1.0)
        principled.inputs['Roughness'].default_value = 0.55
        principled.inputs['Specular'].default_value = 0.3

    def _select_pool_balls(self) -> List[str]:
        """Choose the subset of balls to include in the scene."""
        available = list(POOL_BALL_NAMES)
        num_requested = self.args.num_balls if self.args.num_balls is not None else len(available)
        num_requested = max(1, min(num_requested, len(available)))

        if self.args.default_ball_positions:
            selected = available[:num_requested]
        else:
            selected = []
            if "white_ball" in available and num_requested >= 1:
                selected.append("white_ball")
            remaining = [name for name in available if name != "white_ball"]
            self.rng.shuffle(remaining)
            for name in remaining:
                if len(selected) >= num_requested:
                    break
                selected.append(name)
            if len(selected) < num_requested:
                # Include additional balls from the start of the list if needed
                for name in available:
                    if name not in selected:
                        selected.append(name)
                    if len(selected) >= num_requested:
                        break

        logging.info(f"Selected {len(selected)} pool balls: {selected}")
        return selected

    def _sample_random_ball_positions(self, selected_names: List[str], default_positions: dict):
        """Sample random non-overlapping XY positions uniformly within the play area bounds."""
        self._load_pool_geometry_info()

        min_xy, max_xy = self.pool_play_area_bounds
        # margin = max(self.pool_ball_radius * 2.2, 0.05)
        margin = (max_xy[0] - min_xy[0]) * 0.09
        min_x, min_y = min_xy[0] + margin, min_xy[1] + margin
        max_x, max_y = max_xy[0] - margin, max_xy[1] - margin
        min_separation = self.pool_ball_radius * 2.5

        positions = {}

        ordered_names = list(selected_names)
        if "white_ball" in ordered_names:
            ordered_names.remove("white_ball")
            ordered_names.insert(0, "white_ball")

        for name in ordered_names:
            for attempt in range(200):
                x_pos = self.rng.uniform(min_x, max_x)
                y_pos = self.rng.uniform(min_y, max_y)

                if all(
                    np.linalg.norm([x_pos - px, y_pos - py]) >= min_separation
                    for px, py in positions.values()
                ):
                    positions[name] = (float(x_pos), float(y_pos))
                    break
            else:
                raise RuntimeError(f"Failed to place pool ball '{name}' without overlap")

        return positions

    def _create_pool_table(self):
        """Instantiate the pool table object and add it to the scene."""
        global POOL_TABLE_FRICTION, POOL_TABLE_RESTITUTION

        self._load_pool_geometry_info()

        urdf_path = Path(POOL_TABLE_ASSET_DIR) / f"{POOL_TABLE_NAME}.urdf"
        table_origin = self._read_urdf_origin_position(urdf_path)
        table_origin = np.array(table_origin) #+ np.array([0.0, 0.0, 0.01])
        if table_origin is None:
            table_origin = (0.0, 0.0, 0.0)

        POOL_TABLE_FRICTION = self.args.table_friction if self.args.table_friction is not None else POOL_TABLE_FRICTION
        POOL_TABLE_RESTITUTION = self.args.table_restitution if self.args.table_restitution is not None else POOL_TABLE_RESTITUTION

        table = kb.FileBasedObject(
            name=POOL_TABLE_NAME,
            simulation_filename=f"{POOL_TABLE_ASSET_DIR}/{POOL_TABLE_NAME}.urdf",
            render_filename=f"{POOL_TABLE_ASSET_DIR}/{POOL_TABLE_NAME}.obj",
            scale=1.0,
            position=table_origin,
            friction=POOL_TABLE_FRICTION,
            restitution=POOL_TABLE_RESTITUTION,
            static=True,
            background=True,
            segmentation_id=self._get_segmentation_id(),
            urdf_origin_offset=tuple(table_origin.tolist()),
        )

        self.scene += table

        if hasattr(table, "urdf_origin_offset"):
            blender_obj = table.linked_objects[self.renderer]
            offset = np.array(table.urdf_origin_offset)
            blender_obj.location = np.array(table.position) - offset

        table._object_type = "pool_table"
        table._friction_coefficient = POOL_TABLE_FRICTION
        table._restitution = POOL_TABLE_RESTITUTION

        # self._ensure_object_ground_clearance(table, clearance=0.003)
        self._apply_pool_table_materials(table)

        self.pool_table = table

        table_idx = self.simulator.get_obj_idx(self.pool_table)
        self.simulator._physics_client.changeDynamics(table_idx, -1, collisionMargin=0.0)
        
        logging.info("Added pool table to scene with friction=%.2f", POOL_TABLE_FRICTION)
        self.metadata["pool_table_origin_position"] = list(table_origin)
        return table

    def _create_pool_balls(self):
        """Instantiate pool balls according to the selected layout."""
        global POOL_BALL_FRICTION, POOL_BALL_RESTITUTION, POOL_BALL_MASS

        self._load_pool_geometry_info()

        selected_names = self._select_pool_balls()
        default_positions = {}
        for name in selected_names:
            urdf_path = Path(POOL_TABLE_ASSET_DIR) / f"{name}.urdf"
            origin = self._read_urdf_origin_position(urdf_path)
            if origin is None:
                origin = (
                    0.0,
                    0.0,
                    float(self.pool_play_surface_z + self.pool_ball_radius)
                )
            default_positions[name] = np.array(origin, dtype=float)

        # layout_type = "urdf_default"
        # positions_map = {name: default_positions[name] for name in selected_names}
        if self.args.default_ball_positions:
            layout_type = "urdf_default"
            positions_map = {name: default_positions[name] for name in selected_names}
        else:
            xy_positions = self._sample_random_ball_positions(selected_names, default_positions)
            layout_type = "random_xy"
            positions_map = {}
            for name in selected_names:
                base = default_positions[name]
                xy = xy_positions[name]
                positions_map[name] = np.array([xy[0], xy[1], base[2]], dtype=float)

        created_balls = []
        xy_metadata = {}
        default_xy_metadata = {}

        for name in selected_names:
            position = positions_map[name]
            position = position - np.array([0.0, 0.0, 0.001])

            POOL_BALL_FRICTION = self.args.ball_friction if self.args.ball_friction is not None else POOL_BALL_FRICTION
            POOL_BALL_RESTITUTION = self.args.ball_restitution if self.args.ball_restitution is not None else POOL_BALL_RESTITUTION

            ball = kb.FileBasedObject(
                name=name,
                simulation_filename=f"{POOL_TABLE_ASSET_DIR}/{name}.urdf",
                render_filename=f"{POOL_TABLE_ASSET_DIR}/{name}.obj",
                scale=1.0,
                position=tuple(position),
                mass=POOL_BALL_MASS,
                friction=POOL_BALL_FRICTION,
                restitution=POOL_BALL_RESTITUTION,
                segmentation_id=self._get_segmentation_id(),
                urdf_origin_offset=tuple(default_positions[name]),
            )

            self.scene += ball

            if hasattr(ball, "urdf_origin_offset"):
                blender_obj = ball.linked_objects[self.renderer]
                offset = np.array(ball.urdf_origin_offset)
                blender_obj.location = position - offset

            ball.velocity = [0.0, 0.0, 0.0]
            ball._object_type = "pool_ball"
            ball._friction_coefficient = POOL_BALL_FRICTION
            ball._restitution = POOL_BALL_RESTITUTION
            ball._mass = POOL_BALL_MASS
            ball.metadata["initial_layout"] = layout_type
            ball.metadata["initial_position"] = position.tolist()
            ball.metadata["urdf_origin_position"] = default_positions[name].tolist()

            created_balls.append(ball)
            xy_metadata[name] = [float(position[0]), float(position[1])]
            default_xy_metadata[name] = [float(default_positions[name][0]), float(default_positions[name][1])]

        self.pool_balls = created_balls
        # for ball in self.pool_balls:
        #     ball_idx = self.simulator.get_obj_idx(ball)
        #     self.simulator._physics_client.changeDynamics(ball_idx, -1, collisionMargin=0.0)
        self.metadata["pool_ball_positions_xy"] = xy_metadata
        self.metadata["pool_ball_positions_xyz"] = {
            name: [float(coord) for coord in positions_map[name]]
            for name in selected_names
        }
        self.metadata["pool_ball_default_xy"] = default_xy_metadata
        self.metadata["pool_ball_default_positions_xyz"] = {
            name: default_positions[name].astype(float).tolist()
            for name in selected_names
        }
        self.metadata["pool_ball_layout_type"] = layout_type
        self.metadata["pool_ball_radius"] = self.pool_ball_radius
        self.metadata["pool_play_surface_z"] = self.pool_play_surface_z

        logging.info(
            "Created %d pool balls (layout=%s, height=%.3f)",
            len(created_balls),
            layout_type,
            np.mean([pos[2] for pos in positions_map.values()]),
        )

        return created_balls

    def _collect_all_object_metadata(self):
        """Collect metadata for all objects after all modifications are complete."""
        logging.info("Collecting metadata for all objects...")
        
        # Initialize object_data if not already present
        if "object_data" not in self.metadata:
            self.metadata["object_data"] = {}
        
        all_objects = []
        
        # Add dome if it exists
        if hasattr(self, 'dome') and self.dome:
            all_objects.append(('dome', self.dome))
        
        
        # Add pool table
        if hasattr(self, 'pool_table') and self.pool_table:
            all_objects.append(('pool_table', self.pool_table))

        # Add pool balls
        if hasattr(self, 'pool_balls') and self.pool_balls:
            for ball in self.pool_balls:
                all_objects.append(('pool_ball', ball))
        
        
        logging.info(f"Collecting metadata for {len(all_objects)} objects")
        
        # Collect metadata for all objects
        for obj_type, obj in all_objects:
            try:
                # Get basic object metadata using kubric_utils function
                obj_metadata = get_object_metadata(obj)
                obj_metadata["type"] = obj_type
                obj_metadata["object_name"] = obj.name
                
                # Add material properties if they exist
                if hasattr(obj, '_color') and obj._color:
                    obj_metadata["color"] = [obj._color.r, obj._color.g, obj._color.b]
                if hasattr(obj, '_metallic'):
                    obj_metadata["metallic"] = obj._metallic
                if hasattr(obj, '_roughness'):
                    obj_metadata["roughness"] = obj._roughness
                if hasattr(obj, '_object_type'):
                    obj_metadata["object_type"] = obj._object_type
                if hasattr(obj, '_friction_coefficient'):
                    obj_metadata["friction_coefficient"] = obj._friction_coefficient
                if hasattr(obj, '_scale'):
                    obj_metadata["scale"] = obj._scale
                if hasattr(obj, '_friction'):
                    obj_metadata["friction"] = obj._friction
                if hasattr(obj, '_distance_from_prism'):
                    obj_metadata["distance_from_prism"] = obj._distance_from_prism
                if hasattr(obj, '_angle_from_prism'):
                    obj_metadata["angle_from_prism"] = obj._angle_from_prism
                
                # Store metadata in list-based format
                for key, value in obj_metadata.items():
                    if key not in self.metadata["object_data"]:
                        self.metadata["object_data"][key] = []
                    self.metadata["object_data"][key].append(value)
                
                logging.debug(f"Collected metadata for {obj_type} '{obj.name}': mass={obj_metadata.get('mass', 'N/A')}")
                
            except Exception as e:
                logging.error(f"Error collecting metadata for {obj_type} '{obj.name}': {e}")
        
        logging.info(f"Metadata collection complete. Total objects: {len(all_objects)}")

    def compute_scene_bounds(self, objects):
        """Compute the bounding box of all objects in the scene."""
        if not objects:
            return np.array([-5, -5, 0]), np.array([5, 5, 5])
        
        all_mins = []
        all_maxs = []
        
        for obj in objects:
            min_corner, max_corner = get_world_object_bounds(obj)
            all_mins.append(min_corner)
            all_maxs.append(max_corner)
        
        scene_min = np.min(all_mins, axis=0)
        scene_max = np.max(all_maxs, axis=0)
        
        return scene_min, scene_max

    def _setup_pool_camera(self, objects):
        """Setup a camera that frames the pool table and the active balls."""
        scene_min, scene_max = self.compute_scene_bounds(objects)
        scene_center = (scene_min + scene_max) / 2.0

        focal_length = self.args.focal_length if self.args.focal_length is not None else 50.0
        sensor_width = self.args.sensor_width if hasattr(self.args, 'sensor_width') else 32.0
        if sensor_width < 10.0:
            logging.warning("Sensor width too small; overriding to 32mm for pool camera")
            sensor_width = 32.0

        self.scene.camera = kb.PerspectiveCamera(focal_length=focal_length, sensor_width=sensor_width)

        resolution = tuple(map(int, self.args.resolution.split('x')))
        aspect_ratio = resolution[0] / resolution[1]

        horizontal_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        vertical_fov = 2 * np.arctan((sensor_width / aspect_ratio) / (2 * focal_length))

        table_extent = scene_max - scene_min
        horizontal_extent = max(table_extent[0], table_extent[1]) * 0.9
        vertical_extent = (table_extent[2] * 0.9) if table_extent[2] > 0 else self.pool_ball_radius * 4

        distance_horizontal = horizontal_extent / (2 * np.tan(horizontal_fov / 2))
        distance_vertical = vertical_extent / (2 * np.tan(vertical_fov / 2))
        camera_distance = max(distance_horizontal, distance_vertical, 2.0)

        pool_styles = {
            "overhead": {"elevation": (90, 91), "azimuth_options": [90, 270]},
            "corner": {"elevation": (30, 45), "azimuth_options": [45, 135, 225, 315]},
            "side": {"elevation": (18, 28), "azimuth_options": [0, 90, 180, 270]},
            "cue_line": {"elevation": (25, 40), "azimuth_options": [0, 180]},
        }

        if self.args.composition_style in pool_styles:
            style_name = self.args.composition_style
        else:
            style_name = self.rng.choice(list(pool_styles.keys()))

        style = pool_styles[style_name]

        if self.args.camera_elevation_angle is not None:
            elevation_angle = np.radians(self.args.camera_elevation_angle)
        else:
            elevation_angle = np.radians(self.rng.uniform(*style["elevation"]))

        if self.args.camera_azimuth_angle is not None:
            azimuth_angle = np.radians(self.args.camera_azimuth_angle)
        else:
            base_azimuth = self.rng.choice(style["azimuth_options"])
            # jitter = self.rng.uniform(-10, 10)
            # azimuth_angle = np.radians(base_azimuth + jitter)
            azimuth_angle = np.radians(base_azimuth)

        # distance_variation = self.rng.uniform(0.9, 1.2)
        # camera_distance *= distance_variation

        offset = np.array([
            camera_distance * np.cos(elevation_angle) * np.cos(azimuth_angle),
            camera_distance * np.cos(elevation_angle) * np.sin(azimuth_angle),
            camera_distance * np.sin(elevation_angle),
        ])

        camera_position = scene_center + offset
        look_at_point = scene_center.copy()
        look_at_point[2] = self.pool_play_surface_z

        if style_name in {"side", "cue_line"}:
            look_at_point[:2] += self.rng.normal(scale=0.05, size=2)

        self.scene.camera.position = camera_position
        self.scene.camera.look_at(look_at_point)

        view_direction = look_at_point - camera_position
        view_direction = view_direction / np.linalg.norm(view_direction)

        self._composition_style = style_name
        self._scene_center = look_at_point
        self._camera_position = camera_position
        self._camera_look_direction = view_direction
        self._camera_horizontal_fov = horizontal_fov
        self._camera_vertical_fov = vertical_fov

        self.metadata["pool_camera_position"] = list(camera_position.tolist())
        self.metadata["pool_camera_look_at"] = list(look_at_point.tolist())
        self.metadata["pool_camera_distance"] = float(np.linalg.norm(camera_position - look_at_point))
        self.metadata["pool_camera_style_name"] = style_name

        logging.info(
            "🎥 Pool camera setup: style=%s, position=%s, look_at=%s, distance=%.2f, elevation=%.1f°, azimuth=%.1f°",
            style_name,
            camera_position,
            look_at_point,
            np.linalg.norm(camera_position - look_at_point),
            np.degrees(elevation_angle),
            np.degrees(azimuth_angle),
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _is_object_in_camera_frustum(self, obj_position, margin=0.5):
        """Check if an object is within the camera's view frustum.
        
        Args:
            obj_position: 3D position of the object (x, y, z)
            margin: Additional margin factor to expand frustum (default 0.5 = 50% larger)
            
        Returns:
            bool: True if object is visible in camera frustum
        """
        if not hasattr(self, 'scene') or not hasattr(self.scene, 'camera'):
            return True  # Fallback to include all objects if no camera
            
        camera = self.scene.camera
        cam_pos = np.array(camera.position)
        obj_pos = np.array(obj_position)
        
        # Calculate view direction (camera to object)
        view_vec = obj_pos - cam_pos
        view_distance = np.linalg.norm(view_vec)
        
        if view_distance < 0.01:  # Too close
            return True
            
        view_direction = view_vec / view_distance
        
        # Get camera's look direction (stored during camera setup)
        if hasattr(self, '_camera_look_direction'):
            camera_forward = self._camera_look_direction
        else:
            # Fallback: assume camera looks at scene center
            scene_center = getattr(self, '_scene_center', np.array([0, 0, 0]))
            camera_forward = (scene_center - cam_pos)
            if np.linalg.norm(camera_forward) < 0.01:
                # Camera at scene center, default to looking along negative Y axis
                camera_forward = np.array([0, -1, 0])
            else:
                camera_forward = camera_forward / np.linalg.norm(camera_forward)
        
        # Check if object is in front of camera (dot product > 0)
        forward_dot = np.dot(view_direction, camera_forward)
        if forward_dot < 0.1:  # Behind camera or too far to the side
            return False
            
        # Get camera FOV (stored during camera setup)
        horizontal_fov = getattr(self, '_camera_horizontal_fov', np.radians(60))  # Default 60 degrees
        vertical_fov = getattr(self, '_camera_vertical_fov', np.radians(45))    # Default 45 degrees
        
        # Expand FOV by margin factor
        expanded_h_fov = horizontal_fov * (1 + margin)
        expanded_v_fov = vertical_fov * (1 + margin)
        
        # Calculate camera coordinate system
        world_up = np.array([0, 0, 1])
        right = np.cross(camera_forward, world_up)
        if np.linalg.norm(right) < 0.01:  # Camera looking straight up/down
            right = np.array([1, 0, 0])  # Use world X as right
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, camera_forward)
        up = up / np.linalg.norm(up)
        
        # Project object onto camera's right and up axes
        obj_relative = obj_pos - cam_pos
        proj_right = np.dot(obj_relative, right)
        proj_up = np.dot(obj_relative, up)
        proj_forward = np.dot(obj_relative, camera_forward)
        
        if proj_forward <= 0:  # Behind camera
            return False
            
        # Calculate angles from camera center
        angle_right = np.arctan2(abs(proj_right), proj_forward)
        angle_up = np.arctan2(abs(proj_up), proj_forward)
        
        # Check if within expanded FOV
        within_horizontal = angle_right <= expanded_h_fov / 2
        within_vertical = angle_up <= expanded_v_fov / 2
        
        return within_horizontal and within_vertical

    def _detect_motion_settled(self, animation_data):
        """Detect when motion has settled, considering only objects within camera frustum."""
        num_frames = len(next(iter(animation_data.values()))["velocity"])
        settle_counter = 0
        visible_objects = set()
        
        # First pass: identify which objects are ever visible in the camera frustum
        for obj in animation_data:
            obj_visible = False
            for f in range(num_frames):
                obj_position = animation_data[obj]["position"][f]
                if self._is_object_in_camera_frustum(obj_position):
                    obj_visible = True
                    break
            if obj_visible:
                visible_objects.add(obj)
                
        if not visible_objects:
            logging.warning("No objects detected within camera frustum - using all objects for motion detection")
            visible_objects = set(animation_data.keys())
        else:
            logging.info(f"📷 Motion detection will track {len(visible_objects)} objects visible in camera frustum")
            # Log which objects are being tracked
            obj_names = [getattr(obj, 'name', getattr(obj, 'uid', str(obj))) for obj in visible_objects]
            logging.info(f"🎯 Tracking objects: {obj_names}")
            
        # Second pass: detect when visible objects stop moving
        last_obj_visible_frame = {obj.name: 0 for obj in visible_objects}
        for f in range(num_frames):
            moving = False
            for obj in visible_objects:
                # Check if object is currently in view
                obj_position = animation_data[obj]["position"][f]
                is_obj_in_frustum = self._is_object_in_camera_frustum(obj_position)
                if not is_obj_in_frustum and f >= (last_obj_visible_frame[obj.name] + self.args.not_visible_stop_threshold):
                    continue  # Skip objects that moved out of view

                if is_obj_in_frustum:
                    last_obj_visible_frame[obj.name] = f
                    
                # Check motion thresholds
                v = np.linalg.norm(animation_data[obj]["velocity"][f])
                w = np.linalg.norm(animation_data[obj]["angular_velocity"][f])
                if v > self.args.velocity_threshold or w > self.args.angular_velocity_threshold:
                    moving = True
                    break
                    
            if not moving:
                settle_counter += 1
                if settle_counter >= self.args.settle_frames:
                    settled_frame = f - self.args.settle_frames + 1
                    logging.info(f"Motion settled at frame {settled_frame} (visible objects stopped moving)")
                    return settled_frame
                
            else:
                settle_counter = 0
                
        return None

    def _render_frames(self, animation_data):
        layers = []
        if "image" in self.args.layers:
            layers.append("rgba")
        if "segmentation" in self.args.layers:
            layers.append("segmentation")
        if "depth" in self.args.layers:
            layers.append("depth")

        total_frames = self.scene.frame_end + 1

        if self.args.efficient_rendering:
            settle_frame = self._detect_motion_settled(animation_data)
            if settle_frame is not None and settle_frame < total_frames - 1:
                # Motion settled - use efficient rendering
                frames_to_render = list(range(settle_frame + 1))
                saved_frames = total_frames - len(frames_to_render)
                efficiency_percent = (saved_frames / total_frames) * 100
                
                # Calculate time savings (assuming 5 seconds per frame as mentioned)
                estimated_time_per_frame = 5.0  # seconds
                time_saved = saved_frames * estimated_time_per_frame
                total_estimated_time = total_frames * estimated_time_per_frame
                efficient_time = len(frames_to_render) * estimated_time_per_frame
                
                logging.info(f"🎯 EFFICIENT RENDERING ACTIVATED 🎯")
                logging.info(f"Motion settled at frame {settle_frame} - objects in camera view stopped moving")
                logging.info(f"Rendering {len(frames_to_render)} frames instead of {total_frames}")
                logging.info(f"Frame {settle_frame} will be reused for remaining {saved_frames} frames")
                logging.info(f"💰 Time savings: {efficiency_percent:.1f}% fewer frames rendered")
                logging.info(f"⏱️  Estimated time: {efficient_time:.0f}s instead of {total_estimated_time:.0f}s (saving {time_saved:.0f}s)")
                if time_saved >= 60:
                    logging.info(f"🚀 That's {time_saved/60:.1f} minutes saved!")
                
                # Visual representation of rendering plan
                render_plan = ['R'] * len(frames_to_render) + ['D'] * saved_frames
                plan_str = ''.join(render_plan)
                if len(plan_str) > 50:
                    # Truncate for very long sequences
                    plan_str = plan_str[:25] + f"...{saved_frames}D..." + plan_str[-25:]
                logging.info(f"📊 Rendering plan: {plan_str} (R=Render, D=Duplicate)")
                
                # Store efficiency metadata
                self.metadata["rendering_efficiency"] = {
                    "settle_frame": settle_frame,
                    "frames_rendered": len(frames_to_render),
                    "frames_reused": saved_frames,
                    "total_frames": total_frames,
                    "efficiency_percent": efficiency_percent,
                    "estimated_time_saved_seconds": time_saved,
                    "estimated_total_time_seconds": total_estimated_time,
                    "estimated_efficient_time_seconds": efficient_time,
                    "optimization_method": "camera_frustum_motion_detection",
                    "mode": "efficient"
                }
                
                data = self.renderer.render(frames=frames_to_render, return_layers=layers)
                # Duplicate last frame
                remaining = total_frames - len(frames_to_render)
                for k in layers:
                    last = data[k][-1]
                    dup = np.tile(last[np.newaxis, ...], (remaining,) + (1,) * (last.ndim))
                    data[k] = np.concatenate([data[k], dup], axis=0)
                return data
            else:
                # Motion never settled - render all frames but still in efficient mode
                logging.info("Motion did not settle, rendering all frames normally")
                self.metadata["rendering_efficiency"] = {
                    "settle_frame": None,
                    "frames_rendered": total_frames,
                    "frames_reused": 0,
                    "total_frames": total_frames,
                    "efficiency_percent": 0.0,
                    "mode": "efficient_no_settle"
                }
                return self.renderer.render(return_layers=layers)
        else:
            # Traditional rendering mode
            logging.info("Using traditional rendering for all frames")
            self.metadata["rendering_efficiency"] = {
                "settle_frame": None,
                "frames_rendered": total_frames,
                "frames_reused": 0,
                "total_frames": total_frames,
                "efficiency_percent": 0.0,
                "mode": "traditional"
            }
            return self.renderer.render(return_layers=layers)

    def debug_pause_at_key_points(self):
        """Pause simulation at key points for debugging when GUI is enabled."""
        if hasattr(self.simulator, 'pause_for_inspection'):
            self.simulator.pause_for_inspection("Scene setup complete. Check pool setup and camera angle.")
    
    def add_force_application(self, object_name, force_point_world, force_vector_world, frame=0):
        """Store force application data for later visualization and metadata.
        
        Args:
            object_name: Name of the object the force is applied to
            force_point_world: 3D point where force is applied in world coordinates
            force_vector_world: 3D force vector in world coordinates  
            frame: Frame number when force is applied
        """
        force_data = {
            "object_name": object_name,
            "force_point_world": list(force_point_world),
            "force_vector_world": list(force_vector_world),
            "force_magnitude": float(np.linalg.norm(force_vector_world)),
            "application_frame": frame,
            "timestamp": f"frame_{frame}"
        }
        self.applied_forces.append(force_data)
        logging.info(f"Recorded force application: {force_data['force_magnitude']:.1f}N at {force_point_world} on {object_name}")
    
    def create_force_annotated_image(self, image_path, output_path=None):
        """Create force-annotated version of an image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save annotated image (optional)
            
        Returns:
            tuple: (annotated_image_array, force_visualization_metadata)
        """
        if not self.applied_forces:
            logging.warning("No applied forces recorded for visualization")
            return None, None
            
        # Load the image
        try:
            image = io.imread(image_path)
            if image.shape[-1] == 4:  # RGBA
                image = image[..., :3]  # Convert to RGB
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            return None, None
        
        # Get camera parameters
        camera = self.scene.camera
        camera_position = np.array(camera.position)
        
        # Get camera rotation (quaternion from Blender camera)
        camera_rotation = camera.quaternion if hasattr(camera, 'quaternion') else None
        logging.info(f"Camera quaternion available: {camera_rotation is not None}")
        
        if camera_rotation is None:
            # Fallback: calculate rotation from camera's look direction
            if hasattr(self, '_camera_look_direction'):
                # Create rotation matrix from look direction
                look_dir = np.array(self._camera_look_direction)
                world_up = np.array([0, 0, 1])
                right = np.cross(look_dir, world_up)
                right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1, 0, 0])
                up = np.cross(right, look_dir)
                up = up / np.linalg.norm(up)
                camera_rotation = np.column_stack([right, up, -look_dir])  # -look_dir because camera looks along -Z
                logging.info(f"Using calculated camera rotation from look direction: {look_dir}")
            else:
                logging.warning("Could not determine camera rotation, using identity")
                camera_rotation = np.eye(3)
        
        focal_length = camera.focal_length if hasattr(camera, 'focal_length') else self.args.focal_length
        sensor_width = camera.sensor_width if hasattr(camera, 'sensor_width') else self.args.sensor_width
        
        annotated_image = image.copy()
        force_viz_metadata = []
        
        # Process each applied force
        for i, force_data in enumerate(self.applied_forces):
            force_point = np.array(force_data["force_point_world"])
            force_vector = np.array(force_data["force_vector_world"])
            
            logging.info(f"Processing force {i}: point={force_point}, vector={force_vector}")
            logging.info(f"Camera position: {camera_position}")
            logging.info(f"Camera rotation type: {type(camera_rotation)}")
            
            # Create force visualization
            annotated_image, force_meta = create_force_visualization(
                annotated_image, force_point, force_vector,
                camera_position, camera_rotation, focal_length, sensor_width,
                force_scale=0.001  # Increased scale for better visibility
            )
            
            logging.info(f"Force visualization result: visible={force_meta.get('visible', 'unknown')}")
            if 'reason' in force_meta:
                logging.info(f"Force not visible reason: {force_meta['reason']}")
            if 'image_coordinates' in force_meta:
                logging.info(f"Force image coordinates: {force_meta['image_coordinates']}")
            
            # Add additional metadata
            force_meta.update({
                "object_name": force_data["object_name"],
                "application_frame": force_data["application_frame"],
                "timestamp": force_data["timestamp"]
            })
            
            force_viz_metadata.append(force_meta)
        
        # Save annotated image if output path provided
        if output_path:
            try:
                io.imsave(output_path, annotated_image)
                logging.info(f"Saved force-annotated image to {output_path}")
            except Exception as e:
                logging.error(f"Failed to save annotated image: {e}")
        
        return annotated_image, force_viz_metadata

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    @time_limit(3000)
    def run(self):
        # Phase 1: Setup base objects (ground plane only)
        self._setup_background_and_plane()
        
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()
        # Run simulation to settle the ground plane
        # logging.info("Phase 1: Settling things to the ground plane...")
        # self.simulator.run(frame_start=-20, frame_end=0)

        # Phase 2: Add pool table and balls
        self.pool_table = self._create_pool_table()
        # self.simulator.run(frame_start=-20, frame_end=0)
        # self.simulator.run(frame_start=-20, frame_end=0)

        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()

        self.pool_balls = self._create_pool_balls()
        # self.simulator.run(frame_start=-20, frame_end=0)

        # Collect all objects for camera framing and metadata
        all_objects = [self.pool_table] + list(self.pool_balls)

        # Collect metadata for all objects after all modifications are complete
        self._collect_all_object_metadata()

        # Setup camera to frame the pool table scenario
        self._setup_pool_camera(all_objects)

        # Store metadata about the simulation setup
        self.metadata["simulation_type"] = "pool_table_force"
        self.metadata["pool_table_friction"] = POOL_TABLE_FRICTION
        self.metadata["pool_table_restitution"] = POOL_TABLE_RESTITUTION
        self.metadata["pool_ball_friction"] = POOL_BALL_FRICTION
        self.metadata["pool_ball_restitution"] = POOL_BALL_RESTITUTION
        self.metadata["pool_ball_mass"] = POOL_BALL_MASS
        self.metadata["pool_ball_names"] = [ball.name for ball in self.pool_balls]
        self.metadata["pool_camera_style"] = getattr(self, "_composition_style", "pool_overview")
        self.metadata["composition_style"] = self.metadata["pool_camera_style"]

        # Another debug pause before physics simulation starts
        if self.args.debug_gui and hasattr(self.simulator, 'pause_for_inspection'):
            self.simulator.pause_for_inspection("About to start pool table force simulation.")
            import time
            time.sleep(2)

        # Select ball(s) for force application
        applied_ball = self.rng.choice(self.pool_balls)
        target_ball = None
        direction = np.array([1.0, 0.0, 0.0])

        if self.args.force_to_another_ball and len(self.pool_balls) > 1:
            candidates = [ball for ball in self.pool_balls if ball is not applied_ball]
            target_ball = self.rng.choice(candidates)
            direction = np.array(target_ball.position) - np.array(applied_ball.position)
            direction[2] = 0.0
            target_descriptor = f"towards {target_ball.name}"
        else:
            random_vec = self.rng.normal(size=3)
            random_vec[2] = 0.0
            direction = random_vec
            target_descriptor = "random direction"

        direction_norm = np.linalg.norm(direction[:2])
        if direction_norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
            direction_norm = 1.0
        direction = direction / direction_norm

        if self.args.force_magnitude is not None:
            force_magnitude = self.args.force_magnitude
        else:
            force_magnitude = self.rng.uniform(self.args.min_force, self.args.max_force)

        force_vector = direction * force_magnitude
        force_point = (0.0, 0.0, 0.0)

        ball_idx = self.simulator.get_obj_idx(applied_ball)

        # Store force vector and point for later use with animation data
        self._applied_force_vector = force_vector
        self._applied_force_point = force_point
        self._trajectory_description = target_descriptor
        self._composition_style = getattr(self, "_composition_style", "pool_overview")

        logging.info(
            "🎯 Applying force to %s: vector=%s (magnitude=%.1fN) %s",
            applied_ball.name,
            force_vector,
            np.linalg.norm(force_vector),
            f"towards {target_ball.name}" if target_ball else "in random direction",
        )

        self.simulator.apply_force(ball_idx, force_vector, force_point)
        # self.simulator.apply_torque(ball_idx, force_vector)

        self.metadata["force_applied_ball"] = applied_ball.name
        if target_ball is not None:
            self.metadata["force_target_ball"] = target_ball.name
            self.metadata["force_target_distance"] = float(
                np.linalg.norm(np.array(target_ball.position) - np.array(applied_ball.position))
            )
        self.metadata["force_application_mode"] = "towards_ball" if target_ball else "free_direction"
        self.metadata["force_magnitude"] = float(force_magnitude)
        self.metadata["min_force"] = float(self.args.min_force)
        self.metadata["max_force"] = float(self.args.max_force)
        self.metadata["trajectory_description"] = target_descriptor
        self.metadata["force_direction"] = [float(x) for x in direction]

        # Phase 3: Run main physics simulation
        logging.info("Phase 2: Starting pool table force simulation...")
        logging.info(f"Running simulation for {self.args.frame_end + 1} frames")
        anim_data, _ = self.simulator.run(frame_start=0, frame_end=self.args.frame_end + 1)

        # Debug: Check if objects actually moved
        logging.info("Simulation complete. Checking movement...")

        if applied_ball in anim_data:
            positions = anim_data[applied_ball]["position"]
            velocities = anim_data[applied_ball]["velocity"]
            start_pos = positions[0] if positions else "No data"
            end_pos = positions[-1] if positions else "No data"
            max_velocity = max([np.linalg.norm(v) for v in velocities]) if velocities else 0
            logging.info(
                "Applied ball %s: start=%s, end=%s, max_vel=%.3f",
                applied_ball.name,
                start_pos,
                end_pos,
                max_velocity,
            )

            if positions and hasattr(self, '_applied_force_vector') and hasattr(self, '_applied_force_point'):
                first_frame_position = positions[0]
                force_point_world = np.array(first_frame_position) + np.array(self._applied_force_point)

                self.add_force_application(
                    object_name=applied_ball.name,
                    force_point_world=force_point_world,
                    force_vector_world=self._applied_force_vector,
                    frame=0,
                )

                logging.info(
                    "Recorded force application at position %s (world point %s)",
                    first_frame_position,
                    force_point_world,
                )
        else:
            logging.warning("Applied ball not found in animation data")

        if target_ball and target_ball in anim_data:
            target_positions = anim_data[target_ball]["position"]
            target_velocities = anim_data[target_ball]["velocity"]
            target_max_velocity = (
                max([np.linalg.norm(v) for v in target_velocities]) if target_velocities else 0
            )
            self.metadata["target_ball_max_velocity"] = float(target_max_velocity)
        
        # Save animation data
        picklable = make_picklable(anim_data)
        with open(os.path.join(self.output_dir, "animation_data.pkl"), "wb") as f:
            pkl.dump(picklable, f)

        # Debug pause after physics simulation if GUI is enabled
        if self.args.debug_gui and hasattr(self.simulator, 'pause_for_inspection'):
            self.simulator.pause_for_inspection("Physics simulation complete. Review the results before rendering.")

        # Render frames
        data_stack = self._render_frames(anim_data)

        # Color map & saving
        if "segmentation" in data_stack:
            cmap = create_segmentation_color_map(data_stack["segmentation"])
            self.metadata["segmentation_color_map"] = cmap
            
            # Add segmentation colors to object metadata
            if "segmentation_color" not in self.metadata["object_data"]:
                self.metadata["object_data"]["segmentation_color"] = []
            
            # Add colors based on segmentation IDs stored in metadata
            if "segmentation_id" in self.metadata["object_data"]:
                for seg_id in self.metadata["object_data"]["segmentation_id"]:
                    if seg_id in cmap:
                        color = cmap[seg_id]
                    else:
                        color = [0, 0, 0]  # Default to black if not found
                    self.metadata["object_data"]["segmentation_color"].append(color)
        else:
            cmap = None

        # Save images
        for i in range(data_stack['rgba'].shape[0] if 'rgba' in data_stack else self.args.frame_end + 1):
            if "rgba" in data_stack:
                # Save regular image
                image_path = os.path.join(self.output_dir, f"rgba_{i:05d}.jpg")
                io.imsave(image_path, data_stack["rgba"][i][..., :3])
                
                # Create force-annotated version for first frame if debug_gui is enabled
                if i == 0 and self.applied_forces:
                    try:
                        annotated_image, force_viz_metadata = self.create_force_annotated_image(
                            image_path, 
                            output_path=os.path.join(self.output_dir, f"force_annotated_{i:05d}.jpg")
                        )
                        
                        if force_viz_metadata:
                            # Store force visualization metadata
                            self.metadata["applied_forces_image"] = force_viz_metadata
                            logging.info("🎯 Created force-annotated first frame for debug visualization")
                        else:
                            logging.warning("Failed to create force annotation for first frame")
                            
                    except Exception as e:
                        logging.error(f"Error creating force annotation: {e}")
                        
            if "segmentation" in data_stack and cmap is not None:
                seg_col = apply_segmentation_colors(data_stack["segmentation"][i], cmap)
                io.imsave(os.path.join(self.output_dir, f"segmentation_{i:05d}.png"), seg_col)
        
        # Save depth as npz file
        if "depth" in data_stack:
            depth = data_stack["depth"]  # shape: (frames, H, W, 1)
            max_distance = 15.0  # meters
            create_depth_video(depth[..., 0], os.path.join(self.output_dir, "depth.mp4"), fps=self.args.frame_rate, max_depth_meters=max_distance, min_depth_meters=0.01)

        # Save videos if requested
        if self.args.save_mp4 or self.args.save_gif:
            if "rgba" in data_stack:
                save_video(self.output_dir, "rgba.mp4", "rgba_%05d.jpg", fps=self.args.frame_rate)
            if "segmentation" in data_stack:
                save_video(self.output_dir, "segmentation.mp4", "segmentation_%05d.png", fps=self.args.frame_rate)
        
        # Create tar archive if requested
        if self.args.tar:
            tar_path = os.path.join(self.args.output_dir, f"{self.video_id}.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(self.output_dir, arcname=os.path.basename(self.output_dir))
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)

        # Add applied forces to metadata
        if self.applied_forces:
            self.metadata["applied_forces_simulator"] = convert_numpy_types(self.applied_forces)
            logging.info(f"Stored {len(self.applied_forces)} force applications in metadata")
        
        # Add force calculation details to metadata
        if hasattr(self, '_applied_force_vector') and hasattr(self, '_trajectory_description'):
            self.metadata["force_calculation"] = {
                "composition_style": getattr(self, '_composition_style', 'unknown'),
                "force_vector": convert_numpy_types(self._applied_force_vector),
                "force_magnitude": float(np.linalg.norm(self._applied_force_vector)),
                "trajectory_description": self._trajectory_description,
                "force_point": convert_numpy_types(self._applied_force_point)
            }
        
        # Metadata
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(convert_numpy_types(self.metadata), f, indent=4)

        shutil.rmtree(self.scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    sim = FrictionSlideFlatForceSimulation(args)
    try:
        sim.run()
    except TimeoutException:
        logging.error("Simulation timed out")
        sys.exit(1) 
