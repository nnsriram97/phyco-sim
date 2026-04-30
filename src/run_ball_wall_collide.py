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

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

MOVI_SHAPES = ["cube", "cylinder", "sphere", "cone", "torus", "gear",
               "torus_knot", "sponge", "spot", "teapot", "suzanne"]

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

# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------

parser = kb.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--video_id", type=str, default=str(uuid.uuid4()))
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
parser.add_argument("--scenario", type=str, default="ball_wall_collision")
parser.add_argument("--vary_restitution_only", action="store_true", default=True,
                   help="Keep mass and friction constant, vary only restitution coefficient")
parser.add_argument("--platform_friction", type=float, default=0.1)
parser.add_argument("--ball_friction", type=float, default=0.1)
parser.add_argument("--wall_friction", type=float, default=0.1)
parser.add_argument("--ball_restitution", type=float, default=None)
parser.add_argument("--wall_restitution", type=float, default=None)
parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--save_gif", action="store_true", default=False)
parser.add_argument("--tar", action="store_true", default=False)
parser.add_argument("--debug_gui", action="store_true", default=False, 
                   help="Enable PyBullet GUI for debugging")
parser.add_argument("--debug_frustum", action="store_true", default=False,
                   help="Enable camera frustum debugging and detailed logging")
parser.set_defaults(frame_end=15, frame_rate=10, resolution="768x432")
args = parser.parse_args()

# --------------------------------------------------------------------------------------
# Valid Prisms and Bricks
# --------------------------------------------------------------------------------------

# Use fixed ball for the wall collision simulation
BALL_NAME = "ball_bounce_x_2-0"  # The ball that will hit the wall
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
        self._physics_client.setRealTimeSimulation(1)
        
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

class BallWallCollisionSimulation:
    """Generate videos of ball-wall collision simulation with a ball hitting a static wall on a platform."""
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
            adaptive_sampling=False,
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
        hdri_rotation_z = self.rng.uniform(-np.pi/4, np.pi/4)  # ±30 degrees around Z
        
        hdri_rotation = (hdri_rotation_x, hdri_rotation_y, hdri_rotation_z)
        
        self.renderer._set_ambient_light_hdri(hdri_tex.filename, hdri_rotation=hdri_rotation)
        self.metadata["hdri_id"] = hdri_id
        self.metadata["hdri_rotation"] = {
            "x_radians": hdri_rotation_x,
            "y_radians": hdri_rotation_y, 
            "z_radians": hdri_rotation_z,
            "x_degrees": np.degrees(hdri_rotation_x),
            "y_degrees": np.degrees(hdri_rotation_y),
            "z_degrees": np.degrees(hdri_rotation_z)
        }

        # Use a simple as the ground plane
        ground_plane = kb.FileBasedObject(
            name="ground_plane",
            simulation_filename="objs/plane.urdf",
            render_filename="objs/plane.obj",
            scale=1.0,
            position=(0, 0, 0),
            friction=1.0,
            restitution=0.0,
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
        
        # Ground texture base path
        # GROUND_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "ground_textures")

        # Select wood or concrete texture
        texture_type = self.rng.choice(['wood', 'concrete'])
        if texture_type == 'wood':
            texture_base_path = WOOD_TEXTURE_BASE_PATH
        else:
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

    def _apply_platform_textures(self):
        """Apply randomly selected textures to the cube platform"""
        import os
        import random
        import bpy
        
        # Platform texture base paths
        WOOD_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "wood_textures")
        CONCRETE_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "concrete_textures")
        
        # Access the cube platform's Blender object through Kubric's linked objects
        if not hasattr(self, 'cube_platform') or not self.cube_platform:
            logging.warning("Warning: Cube platform object not available for texture application")
            return False
            
        # Get the Blender representation of the cube platform
        platform_blender = self.cube_platform.linked_objects[self.renderer]
        
        if not platform_blender:
            logging.warning("Warning: Cube platform Blender object not found")
            return False
        
        logging.info(f"Found cube platform Blender object for texturing: {platform_blender.name}")

        # Apply scale and UV unwrap to avoid stretched textures
        try:
            bpy.context.view_layer.objects.active = platform_blender
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(island_margin=0.02)
            bpy.ops.object.mode_set(mode='OBJECT')
            logging.info("Applied transform scale and UV smart project to platform")
        except Exception as e:
            logging.warning(f"UV unwrap for platform failed: {e}")
        
        # Randomly choose between wood and concrete textures
        texture_type = self.rng.choice(['wood', 'concrete'])
        
        if texture_type == 'wood':
            texture_base_path = WOOD_TEXTURE_BASE_PATH
        else:
            texture_base_path = CONCRETE_TEXTURE_BASE_PATH
        
        # Get available texture directories
        texture_types = []
        try:
            texture_types = [d for d in os.listdir(texture_base_path) 
                           if os.path.isdir(os.path.join(texture_base_path, d)) and d.endswith('.blend')]
        except Exception as e:
            logging.error(f"Error reading texture directories: {e}")
            return False
        
        if not texture_types:
            logging.error(f"No texture directories found in {texture_base_path}")
            return False
        
        # Randomly select a texture using the simulation's RNG for reproducibility
        selected_texture = self.rng.choice(texture_types)
        texture_path = os.path.join(texture_base_path, selected_texture, "textures")
        
        # Store selected texture info in metadata
        self.metadata["platform_texture_type"] = texture_type
        self.metadata["platform_texture"] = selected_texture.split(".blend")[0]
        
        logging.info(f"Selected platform texture: {texture_type} - {selected_texture}")
        
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
        
        # Get or create the material for the platform
        if len(platform_blender.material_slots) == 0:
            # Create new material if none exists
            mat = bpy.data.materials.new(name="Platform_Material")
            platform_blender.data.materials.append(mat)
        else:
            mat = platform_blender.material_slots[0].material
            if not mat:
                mat = bpy.data.materials.new(name="Platform_Material")
                platform_blender.material_slots[0].material = mat
        
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
        # Adjust scale to control texture tiling (box projection works best with moderate scale)
        mapping.inputs['Scale'].default_value[0] = 3.0  # Scale X
        mapping.inputs['Scale'].default_value[1] = 3.0  # Scale Y
        mapping.inputs['Scale'].default_value[2] = 3.0  # Scale Z
        
        # Use UV coordinates (from smart unwrap) to prevent top-down stretching
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
            logging.info(f"Applied platform diffuse texture: {diffuse_path}")
        else:
            logging.warning(f"Warning: Platform diffuse texture not found")
        
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
            logging.info(f"Applied platform normal texture: {normal_path}")
        else:
            logging.warning(f"Warning: Platform normal texture not found")
        
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
            logging.info(f"Applied platform roughness texture: {roughness_path}")
        else:
            logging.warning(f"Warning: Platform roughness texture not found")
        
        # Displacement Texture (mainly available for concrete textures)
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
            disp_node.inputs['Scale'].default_value = 0.05  # Reduced displacement strength for platform
            
            links.new(mapping.outputs['Vector'], tex_disp.inputs['Vector'])
            links.new(tex_disp.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
            
            # if platform_blender.modifiers.get("Subdivision") is None:
            #     subdiv = platform_blender.modifiers.new(name="Subdivision", type='SUBSURF')
            #     subdiv.levels = 4  # Number of subdivisions
            #     subdiv.render_levels = 4
            #     subdiv.subdivision_type = 'SIMPLE'  # Use simple subdivision instead of Catmull-Clark

            # Enable displacement in material settings
            mat.cycles.displacement_method = 'BOTH'  # Using both displacement and bump
            
            logging.info(f"Applied platform displacement texture: {displacement_path}")
        else:
            logging.warning(f"Warning: Platform displacement texture not found")

        # Physics properties are handled by Kubric/PyBullet separately
        # No need to set up Blender rigid body physics
        logging.info("Platform texture application completed successfully")
        
        return True

    def _apply_ball_materials(self, ball):
        """Apply random material properties to the ball object."""
        
        # Generate random color for the ball
        ball_color = kb.Color.random_color()
        
        # Create material with random properties suitable for balls
        ball.material = kb.PrincipledBSDFMaterial(
            color=ball_color,
            metallic=self.rng.uniform(0.0, 0.1),  # Keep low metallic for balls
            roughness=self.rng.uniform(0.1, 0.4),  # Smooth to slightly rough
            specular=self.rng.uniform(0.8, 1.0),   # High specular for ball-like appearance
        )
        
        # Store material properties for metadata
        ball._color = ball_color
        ball._metallic = ball.material.metallic
        ball._roughness = ball.material.roughness
        ball._specular = ball.material.specular
        
        logging.info(f"Applied material to ball '{ball.name}': color={[ball_color.r, ball_color.g, ball_color.b]}, metallic={ball.material.metallic:.3f}, roughness={ball.material.roughness:.3f}")
        
        return True

    def _apply_wall_textures(self):
        """Apply the same textures as the platform to the wall"""
        import os
        import random
        
        # Access the wall's Blender object through Kubric's linked objects
        if not hasattr(self, 'wall') or not self.wall:
            logging.warning("Warning: Wall object not available for texture application")
            return False
            
        # Get the Blender representation of the wall
        wall_blender = self.wall.linked_objects[self.renderer]
        
        if not wall_blender:
            logging.warning("Warning: Wall Blender object not found")
            return False
        
        logging.info(f"Found wall Blender object for texturing: {wall_blender.name}")

        # Apply scale and UV unwrap to avoid stretched textures on wall
        try:
            bpy.context.view_layer.objects.active = wall_blender
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(island_margin=0.02)
            bpy.ops.object.mode_set(mode='OBJECT')
            logging.info("Applied transform scale and UV smart project to wall")
        except Exception as e:
            logging.warning(f"UV unwrap for wall failed: {e}")
        
        # Use either brick or concrete textures
        texture_base_path = self.rng.choice([BRICK_TEXTURE_BASE_PATH, CONCRETE_TEXTURE_BASE_PATH])
        
        # Get available texture directories
        texture_types = []
        try:
            texture_types = [d for d in os.listdir(texture_base_path) 
                           if os.path.isdir(os.path.join(texture_base_path, d)) and d.endswith('.blend')]
        except Exception as e:
            logging.error(f"Error reading texture directories: {e}")
            return False
        
        if not texture_types:
            logging.error(f"No texture directories found in {texture_base_path}")
            return False
        
        # # Use the same texture as the platform if available, otherwise random
        # if hasattr(self, 'metadata') and "platform_texture" in self.metadata:
        #     selected_texture = f"{self.metadata['platform_texture']}.blend"
        #     if selected_texture not in texture_types:
        #         selected_texture = self.rng.choice(texture_types)
        # else:
        selected_texture = self.rng.choice(texture_types)
            
        texture_path = os.path.join(texture_base_path, selected_texture, "textures")
        
        # Store selected texture info in metadata
        self.metadata["wall_texture"] = selected_texture.split(".blend")[0]
        
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
        
        # Get or create the material for the wall
        if len(wall_blender.material_slots) == 0:
            # Create new material if none exists
            mat = bpy.data.materials.new(name="Wall_Material")
            wall_blender.data.materials.append(mat)
        else:
            mat = wall_blender.material_slots[0].material
            if not mat:
                mat = bpy.data.materials.new(name="Wall_Material")
                wall_blender.material_slots[0].material = mat
        
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
        tex_coord.location = (-1000, 0)
        
        # Use a combination of mapping nodes for better face orientation handling
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-800, 0)
        
        # Set up proper scaling for brick textures
        # Use uniform scaling to prevent distortion, then fine-tune with second mapping
        mapping.inputs['Scale'].default_value[0] = 2.0  # Scale X - reduced for better proportion
        mapping.inputs['Scale'].default_value[1] = 2.0  # Scale Y - reduced for better proportion
        mapping.inputs['Scale'].default_value[2] = 2.0  # Scale Z - reduced for better proportion
        
        # Use UV coordinates from smart unwrap to prevent vertical striping
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        
        # Add Separate XYZ and Combine XYZ nodes for better axis control
        separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
        separate_xyz.location = (-600, 100)
        combine_xyz = nodes.new(type='ShaderNodeCombineXYZ')
        combine_xyz.location = (-400, 100)
        
        # Link mapping to separate XYZ
        links.new(mapping.outputs['Vector'], separate_xyz.inputs['Vector'])
        
        # Add math nodes to independently control each axis
        math_x = nodes.new(type='ShaderNodeMath')
        math_x.location = (-500, 150)
        math_x.operation = 'MULTIPLY'
        math_x.inputs[1].default_value = 1.0  # X multiplier
        
        math_y = nodes.new(type='ShaderNodeMath')
        math_y.location = (-500, 100)
        math_y.operation = 'MULTIPLY' 
        math_y.inputs[1].default_value = 1.0  # Y multiplier
        
        math_z = nodes.new(type='ShaderNodeMath')
        math_z.location = (-500, 50)
        math_z.operation = 'MULTIPLY'
        math_z.inputs[1].default_value = 0.8  # Z multiplier - reduce stretching on vertical axis
        
        # Connect the math nodes
        links.new(separate_xyz.outputs['X'], math_x.inputs[0])
        links.new(separate_xyz.outputs['Y'], math_y.inputs[0])
        links.new(separate_xyz.outputs['Z'], math_z.inputs[0])
        
        # Combine back
        links.new(math_x.outputs['Value'], combine_xyz.inputs['X'])
        links.new(math_y.outputs['Value'], combine_xyz.inputs['Y'])
        links.new(math_z.outputs['Value'], combine_xyz.inputs['Z'])
        
        # Final mapping for overall adjustments
        mapping2 = nodes.new(type='ShaderNodeMapping')
        mapping2.location = (-300, 0)
        mapping2.inputs['Scale'].default_value[0] = 1.0
        mapping2.inputs['Scale'].default_value[1] = 1.0  
        mapping2.inputs['Scale'].default_value[2] = 1.0
        
        links.new(combine_xyz.outputs['Vector'], mapping2.inputs['Vector'])
        
        # Create and link texture nodes for diffuse, normal, roughness, and displacement
        # Diffuse Texture
        if diffuse_path and os.path.exists(diffuse_path):
            tex_diffuse = nodes.new(type='ShaderNodeTexImage')
            tex_diffuse.location = (-400, 200)
            tex_diffuse.image = bpy.data.images.load(diffuse_path)
            # Set texture extension to repeat for better tiling
            tex_diffuse.extension = 'REPEAT'
            # Set interpolation for smoother edges
            tex_diffuse.interpolation = 'Linear'
            links.new(mapping2.outputs['Vector'], tex_diffuse.inputs['Vector'])
            links.new(tex_diffuse.outputs['Color'], principled.inputs['Base Color'])
            logging.info(f"Applied wall diffuse texture: {diffuse_path}")
        else:
            logging.warning(f"Warning: Wall diffuse texture not found")
        
        # Normal Texture
        if normal_path and os.path.exists(normal_path):
            tex_normal = nodes.new(type='ShaderNodeTexImage')
            tex_normal.location = (-400, 0)
            tex_normal.image = bpy.data.images.load(normal_path)
            # Set correct color space for normal maps
            tex_normal.image.colorspace_settings.name = 'Non-Color'
            # Set texture extension to repeat for better tiling
            tex_normal.extension = 'REPEAT'
            # Set interpolation for smoother edges
            tex_normal.interpolation = 'Linear'
            # Use box projection for consistent mapping on all faces
            tex_normal.projection = 'BOX'
            tex_normal.projection_blend = 0.15
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-200, 0)
            links.new(mapping2.outputs['Vector'], tex_normal.inputs['Vector'])
            links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
            logging.info(f"Applied wall normal texture: {normal_path}")
        else:
            logging.warning(f"Warning: Wall normal texture not found")
        
        # Roughness Texture
        if roughness_path and os.path.exists(roughness_path):
            tex_roughness = nodes.new(type='ShaderNodeTexImage')
            tex_roughness.location = (-400, -200)
            tex_roughness.image = bpy.data.images.load(roughness_path)
            # Set correct color space for roughness maps
            tex_roughness.image.colorspace_settings.name = 'Non-Color'
            # Set texture extension to repeat for better tiling
            tex_roughness.extension = 'REPEAT'
            # Set interpolation for smoother edges
            tex_roughness.interpolation = 'Linear'
            # Use box projection to prevent stretching
            tex_roughness.projection = 'BOX'
            tex_roughness.projection_blend = 0.15
            links.new(mapping2.outputs['Vector'], tex_roughness.inputs['Vector'])
            links.new(tex_roughness.outputs['Color'], principled.inputs['Roughness'])
            logging.info(f"Applied wall roughness texture: {roughness_path}")
        else:
            logging.warning(f"Warning: Wall roughness texture not found")
        
        # Displacement Texture
        if displacement_path and os.path.exists(displacement_path):
            tex_disp = nodes.new(type='ShaderNodeTexImage')
            tex_disp.location = (-400, -400)
            tex_disp.image = bpy.data.images.load(displacement_path)
            # Set correct color space for displacement maps
            tex_disp.image.colorspace_settings.name = 'Non-Color'
            # Set texture extension to repeat for better tiling
            tex_disp.extension = 'REPEAT'
            # Set interpolation for smoother edges
            tex_disp.interpolation = 'Linear'
            # Use box projection to prevent stretching
            tex_disp.projection = 'BOX'
            tex_disp.projection_blend = 0.15
            
            # Add a displacement node
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = (-200, -400)
            disp_node.inputs['Scale'].default_value = 0.05  # Displacement strength
            
            links.new(mapping2.outputs['Vector'], tex_disp.inputs['Vector'])
            links.new(tex_disp.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
            
            # Enable displacement in material settings
            mat.cycles.displacement_method = 'BOTH'  # Using both displacement and bump
            
            logging.info(f"Applied wall displacement texture: {displacement_path}")
        else:
            logging.warning(f"Warning: Wall displacement texture not found")

        # Optional: Add a ColorRamp node to better control texture contrast
        # This can help make brick details more visible
        colorramp = nodes.new(type='ShaderNodeValToRGB')
        colorramp.location = (-200, 200)
        # Adjust the color ramp for better brick contrast if diffuse texture exists
        if diffuse_path and os.path.exists(diffuse_path):
            # Link the diffuse texture through the color ramp for enhanced contrast
            links.new(tex_diffuse.outputs['Color'], colorramp.inputs['Fac'])
            # Create a mix node to blend original and enhanced colors
            mix_node = nodes.new(type='ShaderNodeMixRGB')
            mix_node.location = (-100, 200)
            mix_node.blend_type = 'MULTIPLY'
            mix_node.inputs['Fac'].default_value = 0.3  # Subtle enhancement
            links.new(tex_diffuse.outputs['Color'], mix_node.inputs['Color1'])
            links.new(colorramp.outputs['Color'], mix_node.inputs['Color2'])
            # Replace the direct connection with the enhanced one
            # Remove existing links to Base Color input
            for link in principled.inputs['Base Color'].links:
                links.remove(link)
            links.new(mix_node.outputs['Color'], principled.inputs['Base Color'])

        logging.info("Wall texture application completed successfully")
        
        return True

    def _create_wall(self):
        """Create a static wall using cube_wall-0-0-0.obj."""
        # Use lower friction for the wall as requested
        wall_friction = self.args.wall_friction
        
        # Vary restitution coefficient for the wall
        if self.args.wall_restitution is not None:
            wall_restitution = self.args.wall_restitution
        else:
            wall_restitution = self.rng.uniform(0.0, 0.95)  # Variable restitution
        
        self.metadata["wall_name"] = "cube_wall"
        
        # WALL_NAME = "cube_wall-0-0-0"
        WALL_NAME="plane_wall-0-0-0"
        wall_position = self._read_urdf_origin_offset(f"objs/{WALL_NAME}.urdf")
        
        # If URDF position reading fails, use default position
        if wall_position is None:
            wall_position = (0, 2.0, 0.5)  # Default wall position
            logging.info(f"Using default wall position: {wall_position}")
        else:
            logging.info(f"Using URDF wall position: {wall_position}")
            
        # Create wall using the specified object file
        wall = kb.FileBasedObject(
            name="cube_wall",
            simulation_filename=f"objs/{WALL_NAME}.urdf",
            render_filename=f"objs/{WALL_NAME}.obj",
            scale=1.0,
            position=wall_position,
            friction=wall_friction,
            restitution=wall_restitution,
            static=True,
            background=True,
            segmentation_id=self._get_segmentation_id()
        )
        
        # Add to scene
        self.scene += wall
        
        # Give it a material
        wall_color = kb.Color.random_color()
        wall.material = kb.PrincipledBSDFMaterial(
            color=wall_color,
            metallic=self.rng.uniform(0.0, 0.3),
            roughness=self.rng.uniform(0.3, 0.8),
        )
        
        # Store wall properties and material info for metadata
        wall._color = wall_color
        wall._metallic = wall.material.metallic
        wall._roughness = wall.material.roughness
        wall._friction = wall_friction
        wall._restitution = wall_restitution
        
        logging.info(f"Created wall at position {wall.position} with friction {wall_friction:.3f} and restitution {wall_restitution:.3f}")
        return wall

    def _create_cube_platform(self):
        """Create a flat cube platform for the collision surface using cube_platform.urdf."""
        # Scale and rotation for the cube platform
        platform_scale = 1.0
        
        # Use constant friction for collision simulation
        platform_friction = self.args.platform_friction
        self.metadata["platform_name"] = "cube_platform"
        
        # Create cube platform using the URDF file
        platform = kb.FileBasedObject(
            name="cube_platform",
            simulation_filename="objs/cube_platform.urdf",
            render_filename="objs/cube_platform.obj",
            scale=platform_scale,
            position=(0, 0, 0),
            friction=platform_friction,  # Base friction for the platform
            restitution=0.0,  # Keep constant as requested
            static=True,
            background=True,
            segmentation_id=self._get_segmentation_id()
        )
        
        # Add to scene
        self.scene += platform
        
        # Give it a material
        platform_color = kb.Color.random_color()
        platform.material = kb.PrincipledBSDFMaterial(
            color=platform_color,
            metallic=self.rng.uniform(0.0, 0.3),
            roughness=self.rng.uniform(0.3, 0.8),
        )
        
        # Store material properties and geometry info for metadata
        platform._color = platform_color
        platform._metallic = platform.material.metallic
        platform._roughness = platform.material.roughness
        platform._scale = platform_scale
        platform._friction = platform_friction
        
        logging.info(f"Created cube platform at position {platform.position} with scale {platform_scale:.2f}")
        return platform
    
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
        

    def _create_collision_ball(self, platform):
        """Create a single ball object that will collide with the wall."""
        
        # Keep mass constant as requested, but use lower friction
        ball_mass = 1.0  # Constant mass
        ball_friction = self.args.ball_friction  # Lower friction
        
        # Vary restitution coefficient for the ball
        if self.args.ball_restitution is not None:
            ball_restitution = self.args.ball_restitution
        else:
            ball_restitution = self.rng.uniform(0.1, 0.9)  # Variable restitution
        
        # Store ball name in metadata
        self.metadata["ball_name"] = BALL_NAME

        # Create ball positioned away from the wall
        ball_position = self._read_urdf_origin_offset(f"objs/ball_objs/{BALL_NAME}.urdf")
        # Position ball at negative y to move toward wall at positive y
        if ball_position is None:
            ball_position = (0, -1.5, 0.5)  # Fallback position
        else:
            # Adjust y position to be away from the wall
            ball_position = (ball_position[0], -1.5, ball_position[2] if ball_position[2] > 0 else 0.5)
            
        ball = kb.FileBasedObject(
            name="ball",
            simulation_filename=f"objs/ball_objs/{BALL_NAME}.urdf",
            render_filename=f"objs/ball_objs/{BALL_NAME}.obj",
            scale=1.0,
            position=ball_position,
            mass=ball_mass,
            friction=ball_friction,
            restitution=ball_restitution,
            segmentation_id=self._get_segmentation_id()
        )
        
        # Set initial velocity toward the wall (positive y direction)
        collision_speed = 2.0
        ball.velocity = [0, collision_speed, 0]   # Moving toward wall
        
        # Add to scene
        self.scene += ball
        
        # Apply ball textures/materials
        self._apply_ball_materials(ball)
        
        # Store metadata for the ball
        ball._restitution = ball_restitution
        ball._friction_coefficient = ball_friction
        
        logging.info(f"Created ball ({BALL_NAME}) at position {ball_position} with restitution {ball_restitution:.3f}")
        logging.info(f"Applied velocity toward wall: [0, {collision_speed}, 0]")
        
        return ball

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
        
        # Add cube platform
        if hasattr(self, 'cube_platform') and self.cube_platform:
            all_objects.append(('cube_platform', self.cube_platform))
        
        # Add wall
        if hasattr(self, 'wall') and self.wall:
            all_objects.append(('wall', self.wall))
        
        # Add collision ball
        if hasattr(self, 'ball') and self.ball:
            all_objects.append(('ball', self.ball))
        
        
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
                if hasattr(obj, '_specular'):
                    obj_metadata["specular"] = obj._specular
                if hasattr(obj, '_object_type'):
                    obj_metadata["object_type"] = obj._object_type
                if hasattr(obj, '_friction_coefficient'):
                    obj_metadata["friction_coefficient"] = obj._friction_coefficient
                if hasattr(obj, '_restitution'):
                    obj_metadata["restitution"] = obj._restitution
                if hasattr(obj, '_scale'):
                    obj_metadata["platform_scale"] = obj._scale
                if hasattr(obj, '_friction'):
                    obj_metadata["platform_friction"] = obj._friction
                
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
        """Setup camera to frame the ball and wall collision, accounting for both horizontal and vertical FOV."""
        # Create camera with initial parameters
        focal_length = self.args.focal_length if self.args.focal_length is not None else 50
        self.scene.camera = kb.PerspectiveCamera(focal_length=focal_length)

        # Get ball and wall positions (default fallbacks)
        ball_position = np.array([0.0, -1.5, 0.5])
        wall_position = np.array([0.0, 0.0, 0.5])
        if hasattr(self, 'ball') and self.ball:
            ball_position = np.array(self.ball.position)

        # Calculate the center point between the ball and wall
        scene_center = (ball_position + wall_position) / 2.0

        # Calculate the distance between the ball and wall
        ball_to_wall_distance = np.linalg.norm(ball_position - wall_position)

        # Get camera angles
        if self.args.camera_elevation_angle is not None:
            elevation_angle = np.radians(self.args.camera_elevation_angle)
        else:
            elevation_angle = np.radians(self.rng.uniform(40, 70))

        if self.args.camera_azimuth_angle is not None:
            azimuth_angle = np.radians(self.args.camera_azimuth_angle)
        else:
            azimuth_angle = np.radians(self.rng.uniform(-190, -110))
            # azimuth_angle = np.radians(-90)

        # Sensor width and height
        sensor_width = self.args.sensor_width if hasattr(self.args, 'sensor_width') else 32.0
        scene_resolution = getattr(self.scene, 'resolution', [512, 512])
        aspect_ratio = scene_resolution[1] / scene_resolution[0]
        sensor_height = sensor_width * aspect_ratio

        # Calculate horizontal and vertical field of view in radians
        horizontal_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        vertical_fov = 2 * np.arctan(sensor_height / (2 * focal_length))

        # Temporary camera position for view vector calculation
        temp_distance = 5.0
        temp_x = temp_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
        temp_y = temp_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
        temp_z = temp_distance * np.sin(elevation_angle)
        temp_camera_position = scene_center + np.array([temp_x, temp_y, temp_z])

        # Camera coordinate system
        view_direction = scene_center - temp_camera_position
        view_direction = view_direction / np.linalg.norm(view_direction)
        world_up = np.array([0, 0, 1])
        right = np.cross(view_direction, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)

        # Project ball and wall positions onto camera's image plane
        ball_rel = ball_position - scene_center
        wall_rel = wall_position - scene_center

        ball_right = np.dot(ball_rel, right)
        ball_up = np.dot(ball_rel, up)
        wall_right = np.dot(wall_rel, right)
        wall_up = np.dot(wall_rel, up)

        # Calculate required field of view to contain both ball and wall
        max_right = max(abs(ball_right), abs(wall_right))
        max_up = max(abs(ball_up), abs(wall_up))

        half_horizontal_fov = horizontal_fov / 2.0
        half_vertical_fov = vertical_fov / 2.0

        distance_for_horizontal = max_right / np.tan(half_horizontal_fov) if max_right > 0 else 0
        distance_for_vertical = max_up / np.tan(half_vertical_fov) if max_up > 0 else 0

        min_distance = 1.0
        required_distance = max(distance_for_horizontal, distance_for_vertical, min_distance)

        # Add margin for safety
        margin_factor = 1.2
        initial_distance = required_distance * margin_factor

        # Calculate camera position using spherical coordinates
        x = initial_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
        y = initial_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
        z = initial_distance * np.sin(elevation_angle)
        camera_position = scene_center + np.array([x, y, z])

        # Log camera setup information
        logging.info(f"Ball position: {ball_position}")
        logging.info(f"Wall position: {wall_position}")
        logging.info(f"Scene center: {scene_center}")
        logging.info(f"Ball-to-wall distance: {ball_to_wall_distance:.2f}")
        logging.info(f"Scene resolution: {scene_resolution}")
        logging.info(f"Aspect ratio: {aspect_ratio:.3f}")
        logging.info(f"Focal length: {focal_length}mm")
        logging.info(f"Sensor width: {sensor_width}mm")
        logging.info(f"Sensor height: {sensor_height:.2f}mm")
        logging.info(f"Horizontal FOV: {np.degrees(horizontal_fov):.1f} degrees")
        logging.info(f"Vertical FOV: {np.degrees(vertical_fov):.1f} degrees")
        logging.info(f"Max horizontal extent: {max_right:.2f}")
        logging.info(f"Max vertical extent: {max_up:.2f}")
        logging.info(f"Distance for horizontal: {distance_for_horizontal:.2f}")
        logging.info(f"Distance for vertical: {distance_for_vertical:.2f}")
        logging.info(f"Required distance: {required_distance:.2f}")
        logging.info(f"Final camera distance: {initial_distance:.2f}")
        logging.info(f"Camera position: {camera_position}")
        logging.info(f"Elevation angle: {np.degrees(elevation_angle):.1f} degrees")
        logging.info(f"Azimuth angle: {np.degrees(azimuth_angle):.1f} degrees")

        self.scene.camera.position = camera_position
        self.scene.camera.look_at(scene_center)
        
        # Store camera parameters for frustum calculations
        self._scene_center = scene_center
        self._camera_look_direction = view_direction
        self._camera_horizontal_fov = horizontal_fov
        self._camera_vertical_fov = vertical_fov
        self._camera_position = camera_position
        
        logging.info(f"Stored camera frustum parameters: h_fov={np.degrees(horizontal_fov):.1f}°, v_fov={np.degrees(vertical_fov):.1f}°")

    def _move_camera_back_along_view(self, distance=1.0):
        """Move the camera back along its view direction by a given distance."""
        try:
            import bpy
        except ImportError:
            logging.error("Blender Python API not available, cannot move camera back.")
            return

        cam = bpy.context.scene.camera
        if cam is None:
            logging.error("No camera found in Blender scene.")
            return

        # Get camera's forward direction in world coordinates
        # In Blender, camera looks along its -Z local axis
        forward = -cam.matrix_world.to_quaternion() @ Vector((0, 0, 1))
        cam.location += forward * distance

    def _align_camera_to_objects_in_blender(self, objects):
        """Use Blender's align camera to selected functionality."""
        try:
            import bpy
        except ImportError:
            logging.error("Blender Python API not available, falling back to mathematical camera positioning")
            return
        
        logging.info(f"Attempting to align camera to {len(objects)} objects")
        
        # Ensure we're in object mode
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Clear current selection
        bpy.ops.object.select_all(action='DESELECT')
        
        # Select all objects in Blender (excluding dome)
        selected_count = 0
        selected_obj_names = []
        for obj in objects:
                
            blender_obj = obj.linked_objects[self.renderer]
            if blender_obj is not None and blender_obj.name != "dome":
                blender_obj.select_set(True)
                selected_count += 1
                selected_obj_names.append(blender_obj.name)
            else:
                logging.warning(f"Blender object is None or dome for object: {obj.name}")
        
        if selected_count == 0:
            logging.error("No objects could be selected in Blender")
            return
        
        logging.info(f"Selected {selected_count} objects for camera alignment: {selected_obj_names}")
        
        # Get the camera object in Blender
        camera_obj = self.scene.camera.linked_objects[self.renderer]
        if camera_obj is None:
            logging.error("Camera object not found in Blender")
            return
        
        # Try Blender's camera alignment first
        camera_alignment_success = False
        try_camera_to_view = True
        bpy.ops.view3d.camera_to_view_selected()
        logging.info("Used Blender's camera_to_view_selected")
        # Move camera back slightly to add margin around objects
        self._move_camera_back_along_view(distance=0.3)
        
        # Get the new camera position
        new_position = tuple(camera_obj.location)
        
        logging.info(f"Camera successfully aligned to objects at position: {new_position}")

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
            
            # Optional: Log first and last positions of tracked objects for debugging
            if hasattr(self.args, 'debug_frustum') and self.args.debug_frustum:
                for obj in visible_objects:
                    first_pos = animation_data[obj]["position"][0]
                    last_pos = animation_data[obj]["position"][-1]
                    obj_name = getattr(obj, 'name', getattr(obj, 'uid', 'unknown'))
                    logging.debug(f"Object {obj_name}: first_pos={first_pos}, last_pos={last_pos}")
        
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

    def _test_frustum_calculation(self, animation_data):
        """Test and debug the camera frustum calculation."""
        if not hasattr(self.args, 'debug_frustum') or not self.args.debug_frustum:
            return
            
        logging.info("🔍 Testing camera frustum calculation...")
        
        # Test key object positions
        test_objects = ['ball', 'wall', 'cube_platform']
        for obj in animation_data:
            obj_name = getattr(obj, 'name', getattr(obj, 'uid', 'unknown'))
            if any(name in str(obj_name).lower() for name in test_objects):
                first_pos = animation_data[obj]["position"][0]
                last_pos = animation_data[obj]["position"][-1]
                
                in_frustum_first = self._is_object_in_camera_frustum(first_pos)
                in_frustum_last = self._is_object_in_camera_frustum(last_pos)
                
                logging.info(f"Object {obj_name}:")
                logging.info(f"  First position {first_pos} -> In frustum: {in_frustum_first}")
                logging.info(f"  Last position {last_pos} -> In frustum: {in_frustum_last}")
                
        # Test camera parameters
        if hasattr(self, '_camera_position'):
            logging.info(f"Camera position: {self._camera_position}")
            logging.info(f"Camera look direction: {getattr(self, '_camera_look_direction', 'unknown')}")
            logging.info(f"Horizontal FOV: {np.degrees(getattr(self, '_camera_horizontal_fov', 0)):.1f}°")
            logging.info(f"Vertical FOV: {np.degrees(getattr(self, '_camera_vertical_fov', 0)):.1f}°")

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
            # Test frustum calculation for debugging
            self._test_frustum_calculation(animation_data)
            
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
            self.simulator.pause_for_inspection("Scene setup complete. Check ball collision setup and camera angle.")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    @time_limit(3000)
    def run(self):
        # Phase 1: Setup base objects (cube platform)
        self._setup_background_and_plane()
        self.cube_platform = self._create_cube_platform()
        
        # Apply textures to the cube platform
        self._apply_platform_textures()
        
        # Phase 2: Add wall and collision ball
        self.wall = self._create_wall()
        
        # Apply textures to the wall (same as platform)
        self._apply_wall_textures()

        # # Run simulation to settle the platform (though it's static)
        # logging.info("Phase 1: Settling cube platform...")
        # self.simulator.run(frame_start=-50, frame_end=0)
        
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()
        
        self.ball = self._create_collision_ball(self.cube_platform)
        
        # Collect all objects for camera and metadata
        all_objects = [self.cube_platform, self.wall, self.ball]
        
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()
        
        # Collect metadata for all objects after all modifications are complete
        self._collect_all_object_metadata()
        
        # Setup camera to frame the ball and wall
        self._setup_camera_with_blender_align(all_objects)
        
        # Store metadata about the simulation setup
        self.metadata["simulation_type"] = "ball_wall_collision"
        self.metadata["platform_scale"] = getattr(self.cube_platform, '_scale', 'unknown')
        self.metadata["platform_friction"] = getattr(self.cube_platform, '_friction', 'unknown')
        self.metadata["wall_friction"] = getattr(self.wall, '_friction', 'unknown')
        self.metadata["wall_restitution"] = getattr(self.wall, '_restitution', 'unknown')
        self.metadata["ball_friction"] = getattr(self.ball, '_friction_coefficient', 'unknown')
        self.metadata["ball_restitution"] = getattr(self.ball, '_restitution', 'unknown')
        
        # Another debug pause before physics simulation starts
        if self.args.debug_gui and hasattr(self.simulator, 'pause_for_inspection'):
            self.simulator.pause_for_inspection("About to start ball-wall collision simulation.")
        
        # Phase 3: Run main physics simulation
        logging.info("Phase 3: Starting ball-wall collision simulation...")
        logging.info(f"Platform scale: {getattr(self.cube_platform, '_scale', 'unknown'):.2f}")
        logging.info(f"Platform friction: {getattr(self.cube_platform, '_friction', 'unknown'):.3f}")
        logging.info(f"Wall friction: {getattr(self.wall, '_friction', 'unknown'):.3f}, restitution: {getattr(self.wall, '_restitution', 'unknown'):.3f}")
        logging.info(f"Ball ({BALL_NAME}) friction: {getattr(self.ball, '_friction_coefficient', 'unknown'):.3f}, restitution: {getattr(self.ball, '_restitution', 'unknown'):.3f}")
        logging.info(f"Running simulation for {self.args.frame_end + 1} frames")
        anim_data, _ = self.simulator.run(frame_start=0, frame_end=self.args.frame_end + 1)
        
        # Debug: Check if objects actually moved
        logging.info("Simulation complete. Checking movement...")
        
        # Check ball movement
        if self.ball in anim_data:
            positions = anim_data[self.ball]["position"]
            velocities = anim_data[self.ball]["velocity"]
            start_pos = positions[0] if positions else "No data"
            end_pos = positions[-1] if positions else "No data"
            max_velocity = max([np.linalg.norm(v) for v in velocities]) if velocities else 0
            logging.info(f"Ball: start={start_pos}, end={end_pos}, max_vel={max_velocity:.3f}")
        else:
            logging.warning(f"Ball not found in animation data")
            
        # Check wall (should be static)
        if self.wall in anim_data:
            positions = anim_data[self.wall]["position"]
            velocities = anim_data[self.wall]["velocity"]
            start_pos = positions[0] if positions else "No data"
            end_pos = positions[-1] if positions else "No data"
            max_velocity = max([np.linalg.norm(v) for v in velocities]) if velocities else 0
            logging.info(f"Wall: start={start_pos}, end={end_pos}, max_vel={max_velocity:.3f}")
        else:
            logging.warning(f"Wall not found in animation data")
        
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
                io.imsave(os.path.join(self.output_dir, f"rgba_{i:05d}.jpg"), data_stack["rgba"][i][..., :3])
            if "segmentation" in data_stack and cmap is not None:
                seg_col = apply_segmentation_colors(data_stack["segmentation"][i], cmap)
                io.imsave(os.path.join(self.output_dir, f"segmentation_{i:05d}.png"), seg_col)
        
        # Save depth as npz file
        if "depth" in data_stack:
            depth = data_stack["depth"]  # shape: (frames, H, W)
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

        # Metadata
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(convert_numpy_types(self.metadata), f, indent=4)

        shutil.rmtree(self.scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    sim = BallWallCollisionSimulation(args)
    try:
        sim.run()
    except TimeoutException:
        logging.error("Simulation timed out")
        sys.exit(1) 