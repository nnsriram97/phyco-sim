import os
import sys; sys.path = ["kubric"] + sys.path
import uuid
import signal
import shutil
import tarfile
import logging
from math import radians
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
BRICK_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "brick_textures")
GROUND_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "ground_textures")

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
parser.add_argument("--not_visible_stop_threshold", type=int, default=10)
parser.add_argument("--focal_length", type=float, default=80.0)
parser.add_argument("--sensor_width", type=float, default=32.0)
parser.add_argument("--camera_elevation_angle", type=float, default=None)
parser.add_argument("--camera_azimuth_angle", type=float, default=None)
parser.add_argument("--force_focal_length", action="store_true", default=False)
parser.add_argument("--composition_style", type=str, default=None, help="Choose composition style for the sliding camera")
parser.add_argument("--scenario", type=str, default="friction_slide_flat")
parser.add_argument("--object_friction", type=float, default=None)
parser.add_argument("--object_restitution", type=float, default=None)
parser.add_argument("--object_mass", type=float, default=None)
parser.add_argument("--platform_friction", type=float, default=None)
parser.add_argument("--platform_restitution", type=float, default=None)
parser.add_argument("--force_magnitude", type=float, default=None)
parser.add_argument("--min_force", type=float, default=200.0, help="Minimum force magnitude (N)")
parser.add_argument("--max_force", type=float, default=450.0, help="Maximum force magnitude (N)")

parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--save_gif", action="store_true", default=False)
parser.add_argument("--tar", action="store_true", default=False)
parser.add_argument("--debug_gui", action="store_true", default=False, 
                   help="Enable PyBullet GUI for debugging")
parser.set_defaults(frame_end=15, frame_rate=10, resolution="768x432")
args = parser.parse_args()

# --------------------------------------------------------------------------------------
# Valid Prisms and Bricks
# --------------------------------------------------------------------------------------

# Use a fixed brick for the flat ground simulation
SLIDING_BRICK = "brick_slide_x_0-0"
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
    """Generate videos of friction sliding simulation with objects on a flat ground plane."""
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

        platform_friction = self.rng.uniform(0.05, 1.0) if self.args.platform_friction is None else self.args.platform_friction
        platform_restitution = 0.0 if self.args.platform_restitution is None else self.args.platform_restitution
        
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

    def _apply_brick_textures(self):
        """Apply randomly selected brick textures to the brick object"""
        import os
        import bpy
        
        # Brick texture base path
        BRICK_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "brick_textures")
        
        # Access the brick's Blender object through Kubric's linked objects
        if not hasattr(self, 'brick') or not self.brick:
            logging.warning("Warning: Brick object not available for texture application")
            return False
            
        # Get the Blender representation of the brick
        brick_blender = self.brick.linked_objects[self.renderer]
        
        if not brick_blender:
            logging.warning("Warning: Brick Blender object not found")
            return False
        
        logging.info(f"Found brick Blender object for texturing: {brick_blender.name}")

        # Apply scale and UV unwrap to avoid stretched textures
        try:
            bpy.context.view_layer.objects.active = brick_blender
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(island_margin=0.02)
            bpy.ops.object.mode_set(mode='OBJECT')
            logging.info("Applied transform scale and UV smart project to brick")
        except Exception as e:
            logging.warning(f"UV unwrap for brick failed: {e}")
        
        # Get available texture directories
        texture_dirs = []
        try:
            texture_dirs = [d for d in os.listdir(BRICK_TEXTURE_BASE_PATH) 
                           if os.path.isdir(os.path.join(BRICK_TEXTURE_BASE_PATH, d)) and d.endswith('.blend')]
        except Exception as e:
            logging.error(f"Error reading brick texture directories: {e}")
            return False
        
        if not texture_dirs:
            logging.error(f"No brick texture directories found in {BRICK_TEXTURE_BASE_PATH}")
            return False
        
        # Randomly select a texture using the simulation's RNG for reproducibility
        selected_texture = self.rng.choice(texture_dirs)
        texture_path = os.path.join(BRICK_TEXTURE_BASE_PATH, selected_texture, "textures")
        
        # Store selected texture info in metadata
        self.metadata["brick_texture"] = selected_texture.split(".blend")[0]
        
        logging.info(f"Selected brick texture: {selected_texture}")
        
        # Find texture files with different possible extensions
        diffuse_path = None
        normal_path = None
        roughness_path = None
        displacement_path = None
        
        if os.path.exists(texture_path):
            for file in os.listdir(texture_path):
                file_path = os.path.join(texture_path, file)
                if file.endswith(('.jpg', '.jpeg', '.png', '.exr')):
                    if 'diff' in file.lower():
                        diffuse_path = file_path
                    elif 'nor' in file.lower():
                        normal_path = file_path
                    elif 'rough' in file.lower():
                        roughness_path = file_path
                    elif 'disp' in file.lower():
                        displacement_path = file_path
        
        logging.info(f"Found diffuse: {os.path.basename(diffuse_path) if diffuse_path else 'None'}")
        logging.info(f"Found normal: {os.path.basename(normal_path) if normal_path else 'None'}")
        logging.info(f"Found roughness: {os.path.basename(roughness_path) if roughness_path else 'None'}")
        logging.info(f"Found displacement: {os.path.basename(displacement_path) if displacement_path else 'None'}")
        
        # Get or create the material for the brick
        if len(brick_blender.material_slots) == 0:
            # Create new material if none exists
            mat = bpy.data.materials.new(name="Brick_Material")
            brick_blender.data.materials.append(mat)
        else:
            mat = brick_blender.material_slots[0].material
            if not mat:
                mat = bpy.data.materials.new(name="Brick_Material")
                brick_blender.material_slots[0].material = mat
        
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
        # Adjust scale to control texture tiling - bricks typically need smaller scale
        mapping.inputs['Scale'].default_value[0] = 1.0  # Scale X
        mapping.inputs['Scale'].default_value[1] = 1.0  # Scale Y
        mapping.inputs['Scale'].default_value[2] = 1.0  # Scale Z
        
        # Use UV coordinates (from smart unwrap) to prevent stretching
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        
        # Create and link texture nodes for diffuse, normal, roughness, and displacement
        # Diffuse Texture
        if diffuse_path and os.path.exists(diffuse_path):
            tex_diffuse = nodes.new(type='ShaderNodeTexImage')
            tex_diffuse.location = (-400, 200)
            tex_diffuse.image = bpy.data.images.load(diffuse_path)
            tex_diffuse.projection = 'FLAT'
            links.new(mapping.outputs['Vector'], tex_diffuse.inputs['Vector'])
            links.new(tex_diffuse.outputs['Color'], principled.inputs['Base Color'])
            logging.info(f"Applied brick diffuse texture: {diffuse_path}")
        else:
            logging.warning("Warning: Brick diffuse texture not found")
        
        # Normal Texture
        if normal_path and os.path.exists(normal_path):
            tex_normal = nodes.new(type='ShaderNodeTexImage')
            tex_normal.location = (-400, 0)
            tex_normal.image = bpy.data.images.load(normal_path)
            # Set correct color space for normal maps
            tex_normal.image.colorspace_settings.name = 'Non-Color'
            tex_normal.projection = 'FLAT'
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-200, 0)
            links.new(mapping.outputs['Vector'], tex_normal.inputs['Vector'])
            links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
            logging.info(f"Applied brick normal texture: {normal_path}")
        else:
            logging.warning("Warning: Brick normal texture not found")
        
        # Roughness Texture
        if roughness_path and os.path.exists(roughness_path):
            tex_roughness = nodes.new(type='ShaderNodeTexImage')
            tex_roughness.location = (-400, -200)
            tex_roughness.image = bpy.data.images.load(roughness_path)
            # Set correct color space for roughness maps
            tex_roughness.image.colorspace_settings.name = 'Non-Color'
            tex_roughness.projection = 'FLAT'
            links.new(mapping.outputs['Vector'], tex_roughness.inputs['Vector'])
            links.new(tex_roughness.outputs['Color'], principled.inputs['Roughness'])
            logging.info(f"Applied brick roughness texture: {roughness_path}")
        else:
            logging.warning("Warning: Brick roughness texture not found")
        
        # Displacement Texture
        if displacement_path and os.path.exists(displacement_path):
            tex_disp = nodes.new(type='ShaderNodeTexImage')
            tex_disp.location = (-400, -400)
            tex_disp.image = bpy.data.images.load(displacement_path)
            # Set correct color space for displacement maps
            tex_disp.image.colorspace_settings.name = 'Non-Color'
            tex_disp.projection = 'FLAT'
            
            # Add a displacement node
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = (-200, -400)
            disp_node.inputs['Scale'].default_value = 0.02  # Slightly higher displacement for brick detail
            
            links.new(mapping.outputs['Vector'], tex_disp.inputs['Vector'])
            links.new(tex_disp.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
            
            # Enable displacement in material settings
            mat.cycles.displacement_method = 'BOTH'  # Using both displacement and bump
            
            logging.info(f"Applied brick displacement texture: {displacement_path}")
        else:
            logging.warning("Warning: Brick displacement texture not found")

        # Physics properties are handled by Kubric/PyBullet separately
        # No need to set up Blender rigid body physics
        logging.info("Brick texture application completed successfully")
        
        return True

    def _read_urdf_origin_offset(self, urdf_file):
        """Read the origin offset from the URDF file."""
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(str(urdf_file))
            root = tree.getroot()
            # Find the <origin> tag under <link>/<inertial>
            origin_elem = root.find(".//link/inertial/origin")
            if origin_elem is not None and "xyz" in origin_elem.attrib:
                xyz_str = origin_elem.attrib["xyz"]
                xyz = tuple(float(x) for x in xyz_str.strip().split())
                return xyz
        except Exception as e:
            logging.warning(f"Failed to read URDF origin offset from {urdf_file}: {e}")
        return None
        

    def _create_brick_object(self):
        """Create a brick object that will slide on the ground plane."""
        
        # Keep mass and restitution constant as requested
        object_mass = 1.0  # Constant mass
        
        # Vary only the lateral friction component as requested
        if self.args.object_friction is not None:
            object_friction = self.args.object_friction
        else:
            object_friction = self.rng.uniform(0.05, 1.0)  # Variable friction

        # object_friction = 0.2
        
        # Use the specified brick for sliding
        brick_name = SLIDING_BRICK
        self.metadata["brick_name"] = brick_name
        logging.info(f"Using brick: {brick_name}")

        # Position the brick above the ground plane
        brick_position = [0.0, 0.0, 0.3]  # Start the brick at positive x position above ground
        
        object_restitution = 0.0 if self.args.object_restitution is None else self.args.object_restitution
        object_mass = 1.0 if self.args.object_mass is None else self.args.object_mass

        brick = kb.FileBasedObject(
            name="brick",
            simulation_filename=f"objs/bricks/{brick_name}.urdf",
            render_filename=f"objs/bricks/{brick_name}.obj",
            scale=1.0,
            position=brick_position,
            mass=object_mass,
            friction=object_friction,  # Base friction for the brick
            restitution=object_restitution,
            segmentation_id=self._get_segmentation_id()
        )
        
        
        # Add to scene
        self.scene += brick
        
        # If this object has urdf_origin_offset then we need to apply it to the position in the blender object
        if hasattr(brick, 'urdf_origin_offset'):
            brick_position_blender = brick_position - np.array(brick.urdf_origin_offset)
            blender_obj = brick.linked_objects[self.renderer]
            blender_obj.location = brick_position_blender
        
        logging.info(f"Created sliding brick at position {brick_position} with friction {object_friction:.3f}")
        return brick

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
        
        
        # Add brick object
        if hasattr(self, 'brick') and self.brick:
            all_objects.append(('brick', self.brick))
        
        
        logging.info(f"Collecting metadata for {len(all_objects)} objects")
        
        # Collect metadata for all objects
        for obj_type, obj in all_objects:
            try:
                # Get basic object metadata using kubric_utils function
                obj_metadata = get_object_metadata(obj)
                obj_metadata["type"] = obj_type
                
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

    def _setup_camera_with_blender_align(self, objects):
        """Setup camera with diverse viewpoints for sliding block trajectory."""
        
        # Create camera with initial parameters
        focal_length = self.args.focal_length if self.args.focal_length is not None else 50
        sensor_width = self.args.sensor_width if hasattr(self.args, 'sensor_width') else 32.0
        
        # Override sensor width if it's too small
        if sensor_width < 10.0:
            logging.warning(f"Sensor width {sensor_width}mm is very small, using 32mm for better framing")
            sensor_width = 32.0
            
        self.scene.camera = kb.PerspectiveCamera(focal_length=focal_length, sensor_width=sensor_width)

        # Get sliding trajectory positions
        sliding_start_position = np.array([0.0, 0.0, 0.0])  # Brick starts here (slightly above ground)
        
        # Calculate trajectory parameters
        trajectory_start = sliding_start_position
        trajectory_length = 2.0
        
        # Calculate camera parameters based on resolution
        scene_resolution = tuple(map(int, self.args.resolution.split('x')))
        aspect_ratio = scene_resolution[0] / scene_resolution[1]  # width / height
        
        # Calculate field of view
        horizontal_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        vertical_fov = 2 * np.arctan((sensor_width / aspect_ratio) / (2 * focal_length))
        
        # Get camera angles with constraints
        if self.args.camera_elevation_angle is not None:
            elevation_angle = np.radians(self.args.camera_elevation_angle)
        else:
            elevation_angle = np.radians(self.rng.uniform(5, 65))  # 5-65 degrees as requested
            # elevation_angle = np.radians(25)

        if self.args.camera_azimuth_angle is not None:
            azimuth_angle = np.radians(self.args.camera_azimuth_angle)
        else:
            # azimuth_angle = np.radians(self.rng.uniform(0, 360))  # Full 360 degree range
            azimuth_angle = np.radians(self.rng.uniform(-70, -110))

        # DIVERSITY: Vary camera distance for close/far shots
        distance_variation_factor = 1.0 #self.rng.uniform(0.8, 1.4)  # 20% closer to 40% farther
        
        # Calculate required camera distance to frame the sliding trajectory
        # The horizontal extent we need to capture is the trajectory length plus margin
        horizontal_extent_needed = trajectory_length * 1.05  # 30% margin
        
        # Calculate distance needed for horizontal framing
        distance_for_horizontal = horizontal_extent_needed / (2 * np.tan(horizontal_fov / 2))
        
        # Apply distance variation for diversity
        camera_distance = distance_for_horizontal * 1.1 * distance_variation_factor
        
        # Ensure minimum distance
        camera_distance = max(camera_distance, 2.0)
        
        # DIVERSITY: Create diverse sliding composition styles
        block_composition_styles = [
            'center',
            'left_third',
            'right_third',
            'upper_center',
            'upper_left',
            'upper_right',
            'lower_center',
            'lower_left',
            'lower_right',
        ]
        
        composition_style = self.rng.choice(block_composition_styles) if self.args.composition_style is None else self.args.composition_style
        
        # Calculate base camera position using spherical coordinates around trajectory center
        base_x = camera_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
        base_y = camera_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
        base_z = camera_distance * np.sin(elevation_angle)

        base_camera_position = trajectory_start + np.array([base_x, base_y, base_z])
        
        # Calculate look-at point based on composition style
        look_at_point = self._calculate_sliding_look_at_point(
            trajectory_start, 
            composition_style, horizontal_fov, vertical_fov, camera_distance
        )
        
        camera_position = base_camera_position
        final_look_at_point = look_at_point
        
        # # Ensure trajectory visibility
        # camera_position = self._ensure_sliding_trajectory_visibility(
        #     camera_position, final_look_at_point, trajectory_start, 
        #     horizontal_fov, vertical_fov, camera_distance
        # )
        
        # Position camera and point it to look at the calculated point
        self.scene.camera.position = camera_position
        self.scene.camera.look_at(final_look_at_point)
        
        # Store composition info for logging and metadata
        self._composition_style = composition_style
        self._distance_variation_factor = distance_variation_factor
        
        # Calculate and store camera parameters for frustum calculations
        view_direction = (final_look_at_point - camera_position)
        view_direction = view_direction / np.linalg.norm(view_direction)
        
        self._scene_center = trajectory_start
        self._camera_look_direction = view_direction
        self._camera_horizontal_fov = horizontal_fov
        self._camera_vertical_fov = vertical_fov
        self._camera_position = camera_position

        # Log detailed camera setup information
        logging.info(f"🎥 CAMERA SETUP FOR SLIDING BLOCK WITH DIVERSE VIEWPOINTS")
        logging.info(f"🎨 Composition style: {composition_style}")
        logging.info(f"📏 Distance variation factor: {distance_variation_factor:.2f} ({'closer' if distance_variation_factor < 1.0 else 'farther' if distance_variation_factor > 1.0 else 'normal'})")
        logging.info(f"Block start position: {sliding_start_position}")
        logging.info(f"Look-at point: {final_look_at_point}")
        logging.info(f"Trajectory length: {trajectory_length:.2f}")
        logging.info(f"Scene resolution: {scene_resolution}")
        logging.info(f"Aspect ratio: {aspect_ratio:.3f}")
        logging.info(f"Focal length: {focal_length}mm")
        logging.info(f"Sensor width: {sensor_width}mm")
        logging.info(f"Horizontal FOV: {np.degrees(horizontal_fov):.1f}°")
        logging.info(f"Vertical FOV: {np.degrees(vertical_fov):.1f}°")
        logging.info(f"Required horizontal extent: {horizontal_extent_needed:.2f}")
        logging.info(f"Distance for horizontal framing: {distance_for_horizontal:.2f}")
        logging.info(f"Final camera distance: {camera_distance:.2f}")
        logging.info(f"Camera position: {camera_position}")
        logging.info(f"Distance from camera to look-at point: {np.linalg.norm(camera_position - final_look_at_point):.2f}")
        logging.info(f"Elevation angle: {np.degrees(elevation_angle):.1f}°")
        logging.info(f"Azimuth angle: {np.degrees(azimuth_angle):.1f}°")
        logging.info(f"Camera looking at: {final_look_at_point}")
        
        # Additional framing analysis
        block_size_in_frame = 0.2 / camera_distance  # Assuming block size ~0.2
        logging.info(f"📏 Estimated block size in frame: {block_size_in_frame:.3f} (larger = more close-up)")
        if block_size_in_frame < 0.05:
            logging.warning(f"⚠️  Block may appear small in frame - consider reducing camera distance")
        elif block_size_in_frame > 0.2:
            logging.info(f"🎯 Good close-up framing - block should be prominent")
        else:
            logging.info(f"📐 Moderate framing - block should be clearly visible")
    
    def _calculate_sliding_look_at_point(self, trajectory_start, 
                                       composition_style, horizontal_fov, vertical_fov, camera_distance):
        """Calculate look-at point based on desired composition style to vary ball position in frame."""
        
        # Base look-at point starts with trajectory center
        look_at_point = trajectory_start.copy()
        
        # Calculate offset distances based on FOV and camera distance
        # These offsets will move the look-at point, which shifts where the ball appears in frame
        # Use more conservative offsets to ensure trajectory stays in frame
        horizontal_offset_max = camera_distance * np.tan(horizontal_fov / 4)  # 1/4 of FOV width
        vertical_offset_max = camera_distance * np.tan(vertical_fov / 4)      # 1/4 of FOV height
        
        if composition_style == 'center':
            # No offset - ball stays in center (original behavior)
            pass
            
        elif composition_style == 'left_third':
            # Move look-at point to the right, so ball appears on left third
            # Use conservative offset to avoid pushing ball out of frame
            # look_at_point[0] += horizontal_offset_max * 1.5 #self.rng.uniform(0.5, 1.3)
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.55)
            
        elif composition_style == 'right_third':
            # Move look-at point to the left, so ball appears on right third  
            # look_at_point[0] -= horizontal_offset_max * 1.5 #self.rng.uniform(0.5, 1.3)
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.55)
            
        elif composition_style == 'upper_left':
            # Move look-at point right and down, so ball appears upper left
            # look_at_point[0] += horizontal_offset_max * 1.5 #self.rng.uniform(0.5, 1.3)
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.55)
            # look_at_point[2] -= vertical_offset_max * 1.2 #self.rng.uniform(0.5, 1.0)
            look_at_point[2] -= vertical_offset_max * self.rng.uniform(0.5, 1.2)
            
        elif composition_style == 'upper_right':
            # Move look-at point left and down, so ball appears upper right
            # look_at_point[0] -= horizontal_offset_max * 1.5 #self.rng.uniform(0.5, 1.3)
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.55)
            # look_at_point[2] -= vertical_offset_max * 1.2 #self.rng.uniform(0.5, 1.0)
            look_at_point[2] -= vertical_offset_max * self.rng.uniform(0.5, 1.2)

        elif composition_style == 'upper_center':
            # Move look-at point up, so ball appears in lower center (more platform visible)
            # look_at_point[2] -= vertical_offset_max * 1.2 #self.rng.uniform(0.5, 1.0)
            look_at_point[2] -= vertical_offset_max * self.rng.uniform(0.5, 1.2)
            
        elif composition_style == 'lower_center':
            # Move look-at point up, so ball appears in lower center (more platform visible)
            # look_at_point[2] += vertical_offset_max * 1.2 #self.rng.uniform(0.5, 1.0)
            look_at_point[2] += vertical_offset_max * self.rng.uniform(0.5, 1.2)
        elif composition_style == 'lower_left':
            # Move look-at point left and down, so ball appears in lower left
            # look_at_point[0] += horizontal_offset_max * 1.5 #self.rng.uniform(0.5, 1.3)
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.55)
            # look_at_point[2] += vertical_offset_max * 1.2 #self.rng.uniform(0.5, 1.0)
            look_at_point[2] += vertical_offset_max * self.rng.uniform(0.5, 1.2)
            
        elif composition_style == 'lower_right':
            # Move look-at point right and down, so ball appears in lower right
            # look_at_point[0] -= horizontal_offset_max * 1.5 #self.rng.uniform(0.5, 1.3)
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.55)
            # look_at_point[2] += vertical_offset_max * 1.2 #self.rng.uniform(0.5, 1.0)
            look_at_point[2] += vertical_offset_max * self.rng.uniform(0.5, 1.2)
        
        return look_at_point    

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
            self.simulator.pause_for_inspection("Scene setup complete. Check sliding setup and camera angle.")
    
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
    
    def _calculate_force_vector_for_composition(self, composition_style, force_magnitude=500.0):
        """Calculate force vector based on composition style to achieve desired trajectory.
        
        Args:
            composition_style: The composition style (e.g., 'top_left_to_bottom_right')
            force_magnitude: Magnitude of the force to apply
            
        Returns:
            tuple: (force_vector, trajectory_description)
        """
        base_force = force_magnitude

        if composition_style == 'center':
            # Any direction in the XY plane, uniform
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "center_arbitrary_direction"

        elif composition_style == 'left_third':
            # +x, any y (uniform angle in right half-plane)
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([np.abs(np.cos(theta)), np.sin(theta), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "left_third_pos_x"

        elif composition_style == 'right_third':
            # -x, any y (uniform angle in left half-plane)
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([-np.abs(np.cos(theta)), np.sin(theta), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "right_third_neg_x"

        elif composition_style == 'upper_center':
            # -y, any x (uniform angle in upper half-plane)
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([np.cos(theta), -np.abs(np.sin(theta)), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "upper_center_pos_y"

        elif composition_style == 'upper_left':
            # +x, -y (uniform angle in upper-left quadrant)
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([np.abs(np.cos(theta)), -np.abs(np.sin(theta)), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "upper_left_pos_x_neg_y"

        elif composition_style == 'upper_right':
            # -x, -y (uniform angle in upper-right quadrant)
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([-np.abs(np.cos(theta)), -np.abs(np.sin(theta)), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "upper_right_neg_x_neg_y"

        elif composition_style == 'lower_center':
            # +y, any x (uniform angle in lower half-plane)
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([np.cos(theta), np.abs(np.sin(theta)), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "lower_center_pos_y"

        elif composition_style == 'lower_left':
            # +x, +y (uniform angle in lower-left quadrant)
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([np.abs(np.cos(theta)), np.abs(np.sin(theta)), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "lower_left_pos_x_pos_y"

        elif composition_style == 'lower_right':
            # -x, +y (uniform angle in lower-right quadrant)
            theta = np.random.uniform(0, 2 * np.pi)
            sample_direction = np.array([-np.abs(np.cos(theta)), np.abs(np.sin(theta)), 0.0])
            force_vector = sample_direction * base_force
            trajectory_desc = "lower_right_neg_x_pos_y"

        else:
            logging.warning(f"Unknown composition style: {composition_style}")
            force_vector = np.array([-base_force, 0.0, 0.0])
            trajectory_desc = "unknown_composition_style"

        logging.info(f"🎯 Force vector for '{composition_style}': {force_vector}")
        logging.info(f"📐 Expected trajectory: {trajectory_desc}")
        logging.info(f"💪 Force magnitude: {np.linalg.norm(force_vector):.1f}N")

        return force_vector, trajectory_desc

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
        
        # Run simulation to settle the ground plane
        logging.info("Phase 1: Settling things to the ground plane...")
        self.simulator.run(frame_start=-20, frame_end=0)
        
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()

        # Phase 2: Add brick object
        self.brick = self._create_brick_object()

        # Apply brick textures
        self._apply_brick_textures()
        
        self.simulator.run(frame_start=-20, frame_end=0)
        
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()
        
        
        # Collect all objects for camera and metadata
        all_objects = [self.brick]
        
        # Collect metadata for all objects after all modifications are complete
        self._collect_all_object_metadata()
        
        # Setup camera to frame brick and origin at the ends of the view
        self._setup_camera_with_blender_align(all_objects)
        
        # Store metadata about the simulation setup
        self.metadata["simulation_type"] = "friction_slide_flat"
        self.metadata["brick_friction"] = getattr(self.brick, '_friction_coefficient', 'unknown')
        self.metadata["brick_type"] = getattr(self.brick, '_object_type', 'unknown')
        self.metadata["composition_style"] = getattr(self, '_composition_style', 'side_view')
        self.metadata["trajectory_description"] = getattr(self, '_trajectory_description', 'unknown')
        
        # Another debug pause before physics simulation starts
        if self.args.debug_gui and hasattr(self.simulator, 'pause_for_inspection'):
            self.simulator.pause_for_inspection("About to start friction sliding simulation.")
            import time
            time.sleep(2)
        
        
        # Add force to the brick based on composition style
        brick_idx = self.simulator.get_obj_idx(self.brick)
        
        # Calculate force vector based on composition style for desired trajectory
        composition_style = getattr(self, '_composition_style', 'side_view')
        force_magnitude = self.args.force_magnitude if self.args.force_magnitude is not None else 500.0
        force_vector, trajectory_description = self._calculate_force_vector_for_composition(composition_style, force_magnitude)
        force_point = (0.0, 0.0, 0.0)  # Apply force at the object's center of mass to prevent tumbling
        
        # Store force vector and point for later use with animation data
        self._applied_force_vector = force_vector
        self._applied_force_point = force_point
        self._trajectory_description = trajectory_description
        
        logging.info(f"🎯 Applying {composition_style} force: {force_vector} at point {force_point}")
        logging.info(f"📐 Expected trajectory: {trajectory_description}")
        logging.info(f"💪 Force magnitude: {np.linalg.norm(force_vector):.1f}N")
        self.simulator.apply_force(brick_idx, force_vector, force_point)

        # Phase 3: Run main physics simulation
        logging.info("Phase 2: Starting friction sliding simulation...")
        logging.info(f"Sliding object type: {getattr(self.brick, '_object_type', 'unknown')}")
        logging.info(f"Sliding object friction: {getattr(self.brick, '_friction_coefficient', 'unknown')}")
        logging.info(f"Running simulation for {self.args.frame_end + 1} frames")
        anim_data, _ = self.simulator.run(frame_start=0, frame_end=self.args.frame_end + 1)
        
        # Debug: Check if objects actually moved
        logging.info("Simulation complete. Checking movement...")
        
        # Check sliding object movement and record force application with actual first frame position
        if self.brick in anim_data:
            positions = anim_data[self.brick]["position"]
            velocities = anim_data[self.brick]["velocity"]
            start_pos = positions[0] if positions else "No data"
            end_pos = positions[-1] if positions else "No data"
            max_velocity = max([np.linalg.norm(v) for v in velocities]) if velocities else 0
            logging.info(f"Sliding object: start={start_pos}, end={end_pos}, max_vel={max_velocity:.3f}")
            
            # Now record force application using the actual first frame position
            if positions and hasattr(self, '_applied_force_vector') and hasattr(self, '_applied_force_point'):
                first_frame_position = positions[0]  # Actual position from animation data
                # Convert force point from object-local to world coordinates using first frame position
                force_point_world = np.array(first_frame_position) + np.array(self._applied_force_point)
                
                # Store force application data for visualization and metadata
                self.add_force_application(
                    object_name=self.brick.name,
                    force_point_world=force_point_world,
                    force_vector_world=self._applied_force_vector,
                    frame=0  # Force applied at start of simulation
                )
                
                logging.info(f"Recorded force application at actual first frame position: {first_frame_position}")
                logging.info(f"Force point in world coordinates: {force_point_world}")
        else:
            logging.warning(f"Sliding object not found in animation data")
        
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
        
        # Add min/max force to metadata
        self.metadata["min_force"] = float(self.args.min_force)
        self.metadata["max_force"] = float(self.args.max_force)
        
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