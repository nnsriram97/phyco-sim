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
parser.add_argument("--max_motion_blur", type=float, default=0.0)
parser.add_argument("--layers", type=str, default="image,segmentation,depth")
parser.add_argument("--efficient_rendering", action="store_true", default=False)
parser.add_argument("--velocity_threshold", type=float, default=0.005)
parser.add_argument("--angular_velocity_threshold", type=float, default=0.05)
parser.add_argument("--vertex_deformation_threshold", type=float, default=0.005)
parser.add_argument("--settle_frames", type=int, default=2)
parser.add_argument("--not_visible_stop_threshold", type=int, default=10)
parser.add_argument("--focal_length", type=float, default=80.0)
parser.add_argument("--sensor_width", type=float, default=32)
parser.add_argument("--camera_elevation_angle", type=float, default=None)
parser.add_argument("--camera_azimuth_angle", type=float, default=None)
parser.add_argument("--force_focal_length", action="store_true", default=False)
parser.add_argument("--scenario", type=str, default="ball_drop")
parser.add_argument("--vary_restitution_only", action="store_true", default=True,
                   help="Keep mass and friction constant, vary only restitution coefficient")
parser.add_argument("--composition_style", type=str, default=None,
                   help="Choose composition style for the camera")
parser.add_argument("--platform_friction", type=float, default=0.2)
parser.add_argument("--platform_restitution", type=float, default=0.05)
parser.add_argument("--ball_friction", type=float, default=0.2)
parser.add_argument("--ball_restitution", type=float, default=0.05)
parser.add_argument("--ball_neo_hookean_mu", type=float, default=60.0)
parser.add_argument("--ball_neo_hookean_damping", type=float, default=0.01)
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

# Use fixed ball for the ball drop simulation
# BALL_NAME = "ball_drop_z_1-0"  # The ball that will drop from height
BALL_NAME = "ball_drop_z_0-0"
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

class BallDropSimulation:
    """Generate videos of ball drop simulation with a ball dropping from height onto a platform."""
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
        
        simulator.resetSimulationToDeformableWorld()
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
        hdri_rotation_z = self.rng.uniform(-np.pi/4, np.pi/4)  # ±30 degrees around Z
        
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
        GROUND_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "ground_textures")

        # Select wood or concrete texture
        texture_type = self.rng.choice(['wood', 'concrete', 'ground'])
        if texture_type == 'wood':
            texture_base_path = WOOD_TEXTURE_BASE_PATH
        elif texture_type == 'concrete':
            texture_base_path = CONCRETE_TEXTURE_BASE_PATH
        else:
            texture_base_path = GROUND_TEXTURE_BASE_PATH

        
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

    def _setup_camera_depth_of_field(self, ball_position, camera_position, look_at_point):
        """Setup camera depth of field to focus on the ball and blur the background."""
        import bpy
        
        try:
            # Access the camera object in Blender
            camera_blender = self.scene.camera.linked_objects[self.renderer]
            if not camera_blender:
                logging.warning("Camera Blender object not found for depth of field setup")
                return
            
            # Enable depth of field
            camera_blender.data.dof.use_dof = True
            
            # Calculate focus distance to the ball
            focus_distance = np.linalg.norm(camera_position - ball_position)
            camera_blender.data.dof.focus_distance = focus_distance
            
            # Set aperture (f-stop) for background blur
            # Lower f-stop = more blur, higher f-stop = less blur
            # Good range: f/1.4 to f/4.0 for cinematic look
            f_stop = self.rng.uniform(1.2, 2.8)  # Random f-stop for variety
            camera_blender.data.dof.aperture_fstop = f_stop
            
            # Set aperture blades for bokeh shape (optional)
            camera_blender.data.dof.aperture_blades = 6  # Hexagonal bokeh
            
            # Store DOF info in metadata
            self.metadata["depth_of_field"] = {
                "focus_distance": focus_distance,
                "f_stop": f_stop,
                "aperture_blades": 6,
                "focus_object": "ball"
            }
            
            logging.info(f"🎯 Applied depth of field: focus distance {focus_distance:.2f}, f-stop f/{f_stop:.1f}")
            
        except Exception as e:
            logging.error(f"Error setting up camera depth of field: {e}")

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


    def _create_cube_platform(self):
        """Create a flat cube platform for the collision surface using cube_platform.urdf."""
        # Scale and rotation for the cube platform
        platform_scale = 1.0
        
        # Use constant friction for collision simulation
        platform_friction = self.args.platform_friction

        # NOTE: Changing restiution will not affect bounciness for soft bodies. They don't bounce in PyBullet.        
        platform_restitution = self.args.platform_restitution if self.args.platform_restitution is not None else self.rng.uniform(0.0, 0.90)
        self.metadata["platform_name"] = "cube_platform"
        
        # Create cube platform using the URDF file
        platform = kb.FileBasedObject(
            name="cube_platform",
            simulation_filename="objs/ball_drop/cube_platform.urdf",
            render_filename="objs/ball_drop/cube_platform.obj",
            scale=platform_scale,
            position=(0, 0, 0),
            friction=platform_friction,  # Base friction for the platform
            restitution=platform_restitution,  # Keep constant as requested
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
        platform._restitution = platform_restitution
        
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
        

    def _create_drop_ball(self, platform):
        """Create a single ball object that will drop from height onto the platform."""
        
        # Keep mass constant as requested, but use lower friction
        ball_mass = 1.0  # Constant mass
        ball_friction = self.args.ball_friction  # Lower friction
        
        # Vary restitution coefficient for the ball
        # NOTE: Changing restiution will not affect bounciness for soft bodies. They don't bounce in PyBullet.
        if self.args.ball_restitution is not None:
            ball_restitution = self.args.ball_restitution
        else:
            ball_restitution = self.rng.uniform(0.1, 0.9)  # Variable restitution
        
        # Store ball name in metadata
        self.metadata["ball_name"] = BALL_NAME

        # Create ball positioned at drop height
        urdf_origin_offset = self._read_urdf_origin_offset(f"objs/ball_drop/{BALL_NAME}.urdf")
        ball_position = [0.0, 0.0, 1.0]

        neo_hookean_mu = self.args.ball_neo_hookean_mu if self.args.ball_neo_hookean_mu is not None else 60
        neo_hookean_damping = self.args.ball_neo_hookean_damping if self.args.ball_neo_hookean_damping is not None else 0.01
            
        # ball = kb.FileBasedObject(
        #     name="ball",
        #     simulation_filename=f"objs/ball_drop/{BALL_NAME}.urdf",
        #     render_filename=f"objs/ball_drop/{BALL_NAME}.obj",
        #     scale=1.0,
        #     position=ball_position,
        #     mass=ball_mass,
        #     friction=ball_friction,
        #     restitution=ball_restitution,
        #     segmentation_id=self._get_segmentation_id()
        # )
        
        ball = kb.SoftBody(
            name="ball",
            simulation_filename=f"objs/ball_drop/{BALL_NAME}.vtk",
            render_filename=f"objs/ball_drop/{BALL_NAME}_sf.obj",
            tri_to_tet_mapping_filename=f"objs/ball_drop/{BALL_NAME}_mapping.txt",
            urdf_origin_offset=urdf_origin_offset,
            scale=1.0,
            position=ball_position,
            mass=ball_mass,
            friction=ball_friction,
            restitution=ball_restitution,
            segmentation_id=self._get_segmentation_id(),
            use_mass_spring=False,
            use_bending_constraints=False,
            use_neo_hookean=True,
            neo_hookean_mu=neo_hookean_mu,
            neo_hookean_lambda=300,
            neo_hookean_damping=neo_hookean_damping,
        )
        
        # Set initial velocity to zero for natural drop
        ball.velocity = [0, 0, 0]   # No initial velocity for natural drop
        
        # Add to scene
        self.scene += ball
        
        # Apply ball textures/materials
        self._apply_ball_materials(ball)
        
        # Store metadata for the ball
        ball._restitution = ball_restitution
        ball._friction_coefficient = ball_friction
        
        logging.info(f"Created ball ({BALL_NAME}) at drop position {ball_position} with restitution {ball_restitution:.3f}")
        logging.info(f"Ball will drop naturally under gravity")
        
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
        
        # Add drop ball
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
        """Setup camera to frame the ball drop trajectory with diverse positioning for varied visual composition."""
        
        # Create camera with initial parameters
        focal_length = self.args.focal_length if self.args.focal_length is not None else 50
        sensor_width = self.args.sensor_width if self.args.sensor_width is not None else 32.0
        
        # Override sensor width if it's too small (user may have set it incorrectly)
        if sensor_width < 10.0:
            logging.warning(f"Sensor width {sensor_width}mm is very small, using 32mm for better framing")
            sensor_width = 32.0
            
        self.scene.camera = kb.PerspectiveCamera(focal_length=focal_length, sensor_width=sensor_width)

        # Get ball initial and final positions
        ball_initial_position = np.array([0.0, 0.0, 1.0])  # Default
        ball_final_position = np.array([0.0, 0.0, 0.0])    # Default
        
        if hasattr(self, 'ball') and self.ball:
            ball_initial_position = np.array(self.ball.position)
            
        if hasattr(self, 'cube_platform') and self.cube_platform:
            platform_position = np.array(self.cube_platform.position)
            # Ball will land on top of platform, so add platform height
            ball_final_position = platform_position.copy()
            ball_final_position[2] -= 0.2  # Add some height for ball radius

        # Calculate trajectory parameters
        trajectory_start = ball_initial_position
        trajectory_end = ball_final_position
        trajectory_center = (trajectory_start + trajectory_end) / 2.0
        trajectory_height = abs(trajectory_start[2] - trajectory_end[2])
        
        # Calculate camera parameters based on resolution
        scene_resolution = tuple(map(int, self.args.resolution.split('x')))
        aspect_ratio = scene_resolution[0] / scene_resolution[1]
        
        # Calculate field of view
        horizontal_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        vertical_fov = 2 * np.arctan((sensor_width / aspect_ratio) / (2 * focal_length))
        
        # Get camera angles - prefer angled view for better depth perception
        if self.args.camera_elevation_angle is not None:
            elevation_angle = np.radians(self.args.camera_elevation_angle)
        else:
            elevation_angle = np.radians(self.rng.uniform(5, 45))  # Good angle for ball drop

        if self.args.camera_azimuth_angle is not None:
            azimuth_angle = np.radians(self.args.camera_azimuth_angle)
        else:
            # azimuth_angle = np.radians(self.rng.uniform(0, 360))
            azimuth_angle = np.radians(self.rng.uniform(-80, -100))

        # DIVERSITY: Vary camera distance for close/far shots
        # Instead of always using the same framing, introduce distance variation
        distance_variation_factor = self.rng.uniform(0.8, 1.4)  # 20% closer to 40% farther
        
        # Calculate required camera distance to frame the trajectory
        # We want the ball's initial position at the top of the frame and final position visible
        
        # The vertical extent we need to capture is the trajectory height plus some margin
        vertical_extent_needed = trajectory_height * 1.2  # 20% margin
        
        # Calculate distance needed for vertical framing
        # We want the trajectory to fill 80-90% of the vertical FOV for close-up action
        distance_for_vertical = vertical_extent_needed / (2 * np.tan(vertical_fov / 2))
        
        # Apply distance variation for diversity
        camera_distance = distance_for_vertical * 1.1 * distance_variation_factor
        
        # Ensure minimum distance
        camera_distance = max(camera_distance, 1.5)
        
        # DIVERSITY: Create diverse camera look-at points instead of always trajectory center
        # This will place the ball at different locations in the frame
        
        # Define different composition styles
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
        
        composition_style = self.rng.choice(composition_styles) if self.args.composition_style is None else self.args.composition_style
        
        # Calculate camera position using spherical coordinates around trajectory center
        x = camera_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
        y = camera_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
        z = camera_distance * np.sin(elevation_angle)
        base_camera_position = trajectory_center + np.array([x, y, z])
        
        # Calculate look-at point based on composition style
        look_at_point = self._calculate_diverse_look_at_point(
            trajectory_start, trajectory_end, trajectory_center, 
            composition_style, horizontal_fov, vertical_fov, camera_distance
        )
        
        # Adjust camera position to ensure trajectory is still visible
        camera_position = self._adjust_camera_for_visibility(
            base_camera_position, look_at_point, trajectory_start, trajectory_end,
            horizontal_fov, vertical_fov, camera_distance, elevation_angle, azimuth_angle
        )
        
        # Position camera and point it to look at the calculated point
        self.scene.camera.position = camera_position
        self.scene.camera.look_at(look_at_point)
        
        # Store composition info for logging
        self._composition_style = composition_style
        self._distance_variation_factor = distance_variation_factor
        
        # Calculate and store camera parameters for frustum calculations
        view_direction = (look_at_point - camera_position)
        view_direction = view_direction / np.linalg.norm(view_direction)
        
        self._scene_center = trajectory_center
        self._camera_look_direction = view_direction
        self._camera_horizontal_fov = horizontal_fov
        self._camera_vertical_fov = vertical_fov
        self._camera_position = camera_position

        # Log detailed camera setup information
        logging.info(f"🎥 CAMERA SETUP FOR BALL DROP TRAJECTORY WITH DIVERSITY")
        logging.info(f"🎨 Composition style: {composition_style}")
        logging.info(f"📏 Distance variation factor: {distance_variation_factor:.2f} ({'closer' if distance_variation_factor < 1.0 else 'farther' if distance_variation_factor > 1.0 else 'normal'})")
        logging.info(f"Ball initial position: {ball_initial_position}")
        logging.info(f"Ball final position: {ball_final_position}")
        logging.info(f"Trajectory center: {trajectory_center}")
        logging.info(f"Look-at point: {look_at_point}")
        logging.info(f"Trajectory height: {trajectory_height:.2f}")
        logging.info(f"Scene resolution: {scene_resolution}")
        logging.info(f"Aspect ratio: {aspect_ratio:.3f}")
        logging.info(f"Focal length: {focal_length}mm")
        logging.info(f"Sensor width: {sensor_width}mm")
        logging.info(f"Horizontal FOV: {np.degrees(horizontal_fov):.1f}°")
        logging.info(f"Vertical FOV: {np.degrees(vertical_fov):.1f}°")
        logging.info(f"Required vertical extent: {vertical_extent_needed:.2f}")
        logging.info(f"Distance for vertical framing: {distance_for_vertical:.2f}")
        logging.info(f"Final camera distance: {camera_distance:.2f}")
        logging.info(f"Camera position: {camera_position}")
        logging.info(f"Distance from camera to look-at point: {np.linalg.norm(camera_position - look_at_point):.2f}")
        logging.info(f"Elevation angle: {np.degrees(elevation_angle):.1f}°")
        logging.info(f"Azimuth angle: {np.degrees(azimuth_angle):.1f}°")
        logging.info(f"Camera looking at: {look_at_point}")
        
        # Additional framing analysis
        ball_size_in_frame = 0.1 / camera_distance  # Assuming ball radius ~0.1
        logging.info(f"📏 Estimated ball size in frame: {ball_size_in_frame:.3f} (larger = more close-up)")
        if ball_size_in_frame < 0.05:
            logging.warning(f"⚠️  Ball may appear small in frame - consider reducing camera distance")
        elif ball_size_in_frame > 0.2:
            logging.info(f"🎯 Good close-up framing - ball should be prominent")
        else:
            logging.info(f"📐 Moderate framing - ball should be clearly visible")
        
        # Verify framing by checking if key points are in view
        self._verify_trajectory_framing(ball_initial_position, ball_final_position)
        
        # Apply depth of field to focus on the ball and blur background
        if elevation_angle < np.radians(15):
            self._setup_camera_depth_of_field(ball_initial_position, camera_position, look_at_point)
        
        # Log camera diversity verification
        self._log_camera_diversity_verification(ball_initial_position, ball_final_position, look_at_point)

    def _calculate_diverse_look_at_point(self, trajectory_start, trajectory_end, trajectory_center, 
                                       composition_style, horizontal_fov, vertical_fov, camera_distance):
        """Calculate look-at point based on desired composition style to vary ball position in frame."""
        
        # Base look-at point starts with trajectory center
        look_at_point = trajectory_center.copy()
        
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
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            
        elif composition_style == 'right_third':
            # Move look-at point to the left, so ball appears on right third  
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            
        elif composition_style == 'upper_left':
            # Move look-at point right and down, so ball appears upper left
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            look_at_point[2] -= vertical_offset_max * self.rng.uniform(0.5, 1.0)
            
        elif composition_style == 'upper_right':
            # Move look-at point left and down, so ball appears upper right
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            look_at_point[2] -= vertical_offset_max * self.rng.uniform(0.5, 1.0)
            
        elif composition_style == 'lower_center':
            # Move look-at point up, so ball appears in lower center (more platform visible)
            look_at_point[2] += vertical_offset_max * self.rng.uniform(0.5, 1.0)
        elif composition_style == 'lower_left':
            # Move look-at point left and down, so ball appears in lower left
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            look_at_point[2] += vertical_offset_max * self.rng.uniform(0.5, 1.0)
            
        elif composition_style == 'lower_right':
            # Move look-at point right and down, so ball appears in lower right
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            look_at_point[2] += vertical_offset_max * self.rng.uniform(0.5, 1.0)
        
        return look_at_point
    
    def _adjust_camera_for_visibility(self, base_camera_position, look_at_point, trajectory_start, 
                                    trajectory_end, horizontal_fov, vertical_fov, camera_distance,
                                    elevation_angle, azimuth_angle):
        """Adjust camera position to ensure both trajectory start and end points are visible."""
        
        # Start with the base camera position
        camera_position = base_camera_position.copy()
        
        # Check if both trajectory points are visible with current setup
        max_iterations = 10
        best_camera_position = camera_position.copy()
        best_score = -1
        
        for iteration in range(max_iterations):
            # Calculate view direction from camera to look-at point
            view_direction = (look_at_point - camera_position)
            view_direction = view_direction / np.linalg.norm(view_direction)
            
            # Check if both trajectory points are within the camera frustum
            start_visible = self._is_point_in_frustum(trajectory_start, camera_position, view_direction, 
                                                    horizontal_fov, vertical_fov)
            end_visible = self._is_point_in_frustum(trajectory_end, camera_position, view_direction,
                                                  horizontal_fov, vertical_fov)
            
            # Calculate visibility score (2 = both visible, 1 = one visible, 0 = none visible)
            visibility_score = int(start_visible) + int(end_visible)
            
            # Also check how well-centered the trajectory is
            start_screen_coords = self._get_screen_coordinates(trajectory_start, camera_position, view_direction, horizontal_fov, vertical_fov)
            end_screen_coords = self._get_screen_coordinates(trajectory_end, camera_position, view_direction, horizontal_fov, vertical_fov)
            
            # Penalize if points are too close to frame edges
            edge_penalty = 0
            if start_screen_coords:
                edge_penalty += max(0, abs(start_screen_coords[0]) - 0.8) + max(0, abs(start_screen_coords[1]) - 0.8)
            if end_screen_coords:
                edge_penalty += max(0, abs(end_screen_coords[0]) - 0.8) + max(0, abs(end_screen_coords[1]) - 0.8)
            
            total_score = visibility_score - edge_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_camera_position = camera_position.copy()
            
            if start_visible and end_visible and edge_penalty < 0.2:
                # Both points are visible and well-positioned, we're good
                logging.debug(f"Visibility achieved in {iteration + 1} iterations")
                break
                
            # Adjust camera position for next iteration
            if iteration < max_iterations - 1:
                if not start_visible or not end_visible:
                    # Move camera back to get wider view
                    direction_to_camera = camera_position - look_at_point
                    direction_to_camera = direction_to_camera / np.linalg.norm(direction_to_camera)
                    camera_position = look_at_point + direction_to_camera * camera_distance * (1.2 + iteration * 0.1)
                elif edge_penalty > 0.2:
                    # Points visible but too close to edges, move back slightly
                    direction_to_camera = camera_position - look_at_point
                    direction_to_camera = direction_to_camera / np.linalg.norm(direction_to_camera)
                    camera_position = look_at_point + direction_to_camera * camera_distance * (1.05 + iteration * 0.05)
                
                logging.debug(f"Iteration {iteration + 1}: Adjusting camera - visibility_score={visibility_score}, edge_penalty={edge_penalty:.2f}")
        
        if best_score < 2:
            logging.warning(f"Could not achieve full trajectory visibility. Best score: {best_score}")
        
        return best_camera_position
    
    def _is_point_in_frustum(self, point, camera_position, view_direction, horizontal_fov, vertical_fov):
        """Check if a point is within the camera frustum."""
        
        # Vector from camera to point
        to_point = point - camera_position
        distance = np.linalg.norm(to_point)
        
        if distance < 0.01:  # Too close
            return True
            
        to_point_normalized = to_point / distance
        
        # Check if point is in front of camera
        forward_dot = np.dot(to_point_normalized, view_direction)
        if forward_dot < 0.1:  # Behind camera
            return False
            
        # Calculate camera coordinate system
        world_up = np.array([0, 0, 1])
        right = np.cross(view_direction, world_up)
        if np.linalg.norm(right) < 0.01:  # Camera looking straight up/down
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)
        
        # Project point onto camera's right and up axes
        proj_right = np.dot(to_point, right)
        proj_up = np.dot(to_point, up)
        proj_forward = np.dot(to_point, view_direction)
        
        if proj_forward <= 0:  # Behind camera
            return False
            
        # Calculate angles from camera center
        angle_right = np.arctan2(abs(proj_right), proj_forward)
        angle_up = np.arctan2(abs(proj_up), proj_forward)
        
        # Check if within FOV (with conservative margin to ensure visibility)
        margin_factor = 0.9  # 10% safety margin (inside the frame)
        within_horizontal = angle_right <= (horizontal_fov / 2) * margin_factor
        within_vertical = angle_up <= (vertical_fov / 2) * margin_factor
        
        return within_horizontal and within_vertical

    def _get_screen_coordinates(self, point, camera_position, view_direction, horizontal_fov, vertical_fov):
        """Get normalized screen coordinates (-1 to 1) for a point. Returns None if behind camera."""
        
        # Vector from camera to point
        to_point = point - camera_position
        distance = np.linalg.norm(to_point)
        
        if distance < 0.01:  # Too close
            return None
            
        to_point_normalized = to_point / distance
        
        # Check if point is in front of camera
        forward_dot = np.dot(to_point_normalized, view_direction)
        if forward_dot < 0.1:  # Behind camera
            return None
            
        # Calculate camera coordinate system
        world_up = np.array([0, 0, 1])
        right = np.cross(view_direction, world_up)
        if np.linalg.norm(right) < 0.01:  # Camera looking straight up/down
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)
        
        # Project point onto camera's right and up axes
        proj_right = np.dot(to_point, right)
        proj_up = np.dot(to_point, up)
        proj_forward = np.dot(to_point, view_direction)
        
        if proj_forward <= 0:  # Behind camera
            return None
            
        # Calculate angles from camera center
        angle_right = np.arctan2(proj_right, proj_forward)
        angle_up = np.arctan2(proj_up, proj_forward)
        
        # Normalize by half FOV to get screen coordinates (-1 to 1)
        screen_x = angle_right / (horizontal_fov / 2)
        screen_y = angle_up / (vertical_fov / 2)
        
        return (screen_x, screen_y)

    def _log_camera_diversity_verification(self, ball_initial_pos, ball_final_pos, look_at_point):
        """Log information to verify that camera diversity is working as expected."""
        
        logging.info(f"🔍 CAMERA DIVERSITY VERIFICATION")
        
        # Calculate expected ball position in frame based on composition style
        composition_style = getattr(self, '_composition_style', 'unknown')
        distance_factor = getattr(self, '_distance_variation_factor', 1.0)
        
        # Calculate offset from trajectory center to look-at point
        trajectory_center = (ball_initial_pos + ball_final_pos) / 2.0
        look_at_offset = look_at_point - trajectory_center
        
        logging.info(f"🎨 Composition: {composition_style}")
        logging.info(f"📏 Distance factor: {distance_factor:.2f}")
        logging.info(f"📍 Look-at offset from trajectory center: {look_at_offset}")
        
        # Describe expected visual result
        expected_descriptions = {
            'center': "Ball should appear in center of frame (original behavior)",
            'left_third': "Ball should appear on LEFT side of frame",
            'right_third': "Ball should appear on RIGHT side of frame", 
            'upper_left': "Ball should appear in UPPER LEFT area of frame",
            'upper_right': "Ball should appear in UPPER RIGHT area of frame",
            'lower_center': "Ball should appear in LOWER CENTER with more platform visible",
            'lower_left': "Ball should appear in LOWER LEFT area of frame",
            'lower_right': "Ball should appear in LOWER RIGHT area of frame"
        }
        
        expected_desc = expected_descriptions.get(composition_style, "Unknown composition style")
        logging.info(f"📺 Expected visual result: {expected_desc}")
        
        # Distance description
        if distance_factor < 0.9:
            distance_desc = "CLOSER shot - ball should appear larger"
        elif distance_factor > 1.1:
            distance_desc = "FARTHER shot - ball should appear smaller, more context visible"
        else:
            distance_desc = "NORMAL distance - standard ball size"
        logging.info(f"🔭 Distance effect: {distance_desc}")
        
        # Log magnitude of offset to verify diversity
        offset_magnitude = np.linalg.norm(look_at_offset[:2])  # X,Y offset (ignore Z)
        if offset_magnitude > 0.1:
            logging.info(f"✅ DIVERSITY ACTIVE: Significant look-at offset ({offset_magnitude:.2f}) - ball position should vary")
        else:
            logging.info(f"⚠️  MINIMAL DIVERSITY: Small look-at offset ({offset_magnitude:.2f}) - ball may appear centered")
        
        logging.info(f"🎬 This rendering should show: {expected_desc} with {distance_desc}")

    def _verify_trajectory_framing(self, ball_initial_pos, ball_final_pos):
        """Verify that the ball trajectory is properly framed in the camera view."""
        
        if not hasattr(self.scene, 'camera') or not self.scene.camera:
            logging.warning("No camera available for trajectory verification")
            return
            
        camera = self.scene.camera
        cam_pos = np.array(camera.position)
        
        # Get camera coordinate system
        view_direction = self._camera_look_direction
        world_up = np.array([0, 0, 1])
        right = np.cross(view_direction, world_up)
        if np.linalg.norm(right) < 0.01:  # Camera looking straight up/down
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)
        
        # Project trajectory points onto camera plane
        def project_to_camera_plane(world_pos):
            relative_pos = world_pos - cam_pos
            proj_forward = np.dot(relative_pos, view_direction)
            proj_right = np.dot(relative_pos, right)
            proj_up = np.dot(relative_pos, up)
            
            if proj_forward <= 0:
                return None, None  # Behind camera
                
            # Calculate normalized screen coordinates (-1 to 1)
            angle_right = np.arctan2(proj_right, proj_forward)
            angle_up = np.arctan2(proj_up, proj_forward)
            
            # Normalize by half FOV
            screen_x = angle_right / (self._camera_horizontal_fov / 2)
            screen_y = angle_up / (self._camera_vertical_fov / 2)
            
            return screen_x, screen_y
        
        # Check ball initial position
        init_x, init_y = project_to_camera_plane(ball_initial_pos)
        final_x, final_y = project_to_camera_plane(ball_final_pos)
        
        logging.info(f"📋 TRAJECTORY FRAMING VERIFICATION:")
        
        if init_x is not None and init_y is not None:
            logging.info(f"Ball initial position screen coords: ({init_x:.2f}, {init_y:.2f})")
            if abs(init_x) <= 1.0 and abs(init_y) <= 1.0:
                logging.info(f"✅ Ball initial position is IN FRAME")
                if init_y > 0.5:
                    logging.info(f"✅ Ball starts near TOP of frame (y={init_y:.2f})")
                elif init_y > 0:
                    logging.info(f"📍 Ball starts in UPPER half of frame (y={init_y:.2f})")
                else:
                    logging.info(f"⚠️  Ball starts in LOWER half of frame (y={init_y:.2f})")
            else:
                logging.warning(f"❌ Ball initial position is OUT OF FRAME")
        else:
            logging.warning(f"❌ Ball initial position is BEHIND CAMERA")
            
        if final_x is not None and final_y is not None:
            logging.info(f"Ball final position screen coords: ({final_x:.2f}, {final_y:.2f})")
            if abs(final_x) <= 1.0 and abs(final_y) <= 1.0:
                logging.info(f"✅ Ball final position is IN FRAME")
                if final_y < -0.5:
                    logging.info(f"✅ Ball ends near BOTTOM of frame (y={final_y:.2f})")
                elif final_y < 0:
                    logging.info(f"📍 Ball ends in LOWER half of frame (y={final_y:.2f})")
                else:
                    logging.info(f"⚠️  Ball ends in UPPER half of frame (y={final_y:.2f})")
            else:
                logging.warning(f"❌ Ball final position is OUT OF FRAME")
        else:
            logging.warning(f"❌ Ball final position is BEHIND CAMERA")
            
        # Calculate trajectory coverage
        if (init_x is not None and init_y is not None and 
            final_x is not None and final_y is not None):
            vertical_span = abs(init_y - final_y)
            horizontal_span = abs(init_x - final_x)
            logging.info(f"📏 Trajectory span: vertical={vertical_span:.2f}, horizontal={horizontal_span:.2f}")
            
            if vertical_span > 1.0:
                logging.info(f"🎯 EXCELLENT: Trajectory uses {vertical_span:.1f}x the frame height - very close framing!")
            elif vertical_span > 0.7:
                logging.info(f"✅ GOOD: Trajectory uses {vertical_span:.1f}x the frame height - nice close-up")
            elif vertical_span > 0.4:
                logging.info(f"📐 OK: Trajectory uses {vertical_span:.1f}x the frame height - moderate framing")
            else:
                logging.warning(f"📉 DISTANT: Trajectory only uses {vertical_span:.1f}x the frame height - camera too far away")
                
            # Provide actionable feedback
            if vertical_span < 0.6:
                logging.info(f"💡 TIP: For closer framing, reduce camera distance or increase focal length")
            elif vertical_span > 1.2:
                logging.info(f"💡 TIP: Trajectory may be cut off - consider increasing camera distance slightly")

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
        """Detect when motion has settled, considering only objects within camera frustum and soft body deformation."""
        num_frames = len(next(iter(animation_data.values()))["velocity"])
        settle_counter = 0
        visible_objects = set()
        
        # First pass: identify which objects are ever visible in the camera frustum
        for obj in animation_data:
            obj_visible = True
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
        
        # Second pass: detect when visible objects stop moving (including soft body deformation)
        last_obj_visible_frame = {obj.name: 0 for obj in visible_objects}
        for f in range(num_frames):
            moving = False
            for obj in visible_objects:
                if not isinstance(obj, kb.SoftBody):
                    # Check if object is currently in view
                    obj_position = animation_data[obj]["position"][f]
                    is_obj_in_frustum = self._is_object_in_camera_frustum(obj_position)
                    if not is_obj_in_frustum and f >= (last_obj_visible_frame[obj.name] + self.args.not_visible_stop_threshold):
                        continue  # Skip objects that moved out of view

                    if is_obj_in_frustum:
                        last_obj_visible_frame[obj.name] = f
                        
                    #Check rigid body motion thresholds
                    v = np.linalg.norm(animation_data[obj]["velocity"][f])
                    w = np.linalg.norm(animation_data[obj]["angular_velocity"][f])
                    if v > self.args.velocity_threshold or w > self.args.angular_velocity_threshold:
                        moving = True
                        break
                else:
                    moving = True    
            if not moving:
                settle_counter += 1
                if settle_counter >= self.args.settle_frames:
                    settled_frame = f - self.args.settle_frames + 1
                    logging.info(f"Motion settled at frame {settled_frame} (visible objects stopped moving)")
                    return settled_frame
            else:
                settle_counter = 0
        
        # If there are soft bodies in the scene, check when the deformation settles across all the frames
        if any(isinstance(obj, kb.SoftBody) for obj in visible_objects):
            for obj in visible_objects:
                if isinstance(obj, kb.SoftBody):
                    deformation_settle_frame = self._check_soft_body_deformation(obj, animation_data[obj])
                    if deformation_settle_frame is not None:
                        return deformation_settle_frame
        return None

    def _check_soft_body_deformation(self, obj, obj_animation_data):
        """Check the rate of deformation for a soft body object.
        
        Args:
            obj: The soft body object
            obj_animation_data: Animation data for this object
            
        Returns:
            float: Magnitude of vertex velocity (deformation rate)
        """
        if "vertex_positions" not in obj_animation_data:
            return None
            
        vertex_positions = obj_animation_data["vertex_positions"]
        
        # Make sure we have enough frames
        if len(vertex_positions) < 2:
            return None
            
        # Check the deformations between every consecutive frame
        vertex_positions = np.array(vertex_positions)
        vertex_velocities = vertex_positions[2:] - vertex_positions[:-2]
        velocity_magnitudes = np.linalg.norm(vertex_velocities, axis=-1)
        # In the reverse order check frame where any vertex has a velocity greater than velocity threshold
        for i in range(len(velocity_magnitudes) - 1, -1, -1):
            if np.any(velocity_magnitudes[i] > self.args.vertex_deformation_threshold):
                return i + 2
        
        return None
    def _test_frustum_calculation(self, animation_data):
        """Test and debug the camera frustum calculation."""
        if not hasattr(self.args, 'debug_frustum') or not self.args.debug_frustum:
            return
            
        logging.info("🔍 Testing camera frustum calculation...")
        
        # Test key object positions
        test_objects = ['ball', 'cube_platform']
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
            self.simulator.pause_for_inspection("Scene setup complete. Check ball drop setup and camera angle.")

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
        
        # Phase 2: Add drop ball
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()
        
        self.ball = self._create_drop_ball(self.cube_platform)
        
        # Collect all objects for camera and metadata
        all_objects = [self.cube_platform, self.ball]
        
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()
        
        # Collect metadata for all objects after all modifications are complete
        self._collect_all_object_metadata()
        
        # Setup camera to frame the ball and wall
        self._setup_camera_with_blender_align(all_objects)
        
        # Store metadata about the simulation setup
        self.metadata["simulation_type"] = "ball_drop"
        self.metadata["platform_scale"] = getattr(self.cube_platform, '_scale', 'unknown')
        self.metadata["platform_friction"] = getattr(self.cube_platform, '_friction', 'unknown')
        self.metadata["ball_friction"] = getattr(self.ball, '_friction_coefficient', 'unknown')
        self.metadata["ball_restitution"] = getattr(self.ball, '_restitution', 'unknown')
        
        # Store camera diversity metadata
        self.metadata["camera_diversity"] = {
            "composition_style": getattr(self, '_composition_style', 'unknown'),
            "distance_variation_factor": getattr(self, '_distance_variation_factor', 1.0),
            "camera_position": getattr(self, '_camera_position', np.array([0,0,0])).tolist(),
        }
        
        # Another debug pause before physics simulation starts
        if self.args.debug_gui and hasattr(self.simulator, 'pause_for_inspection'):
            self.simulator.pause_for_inspection("About to start ball drop simulation.")
        
        # Phase 3: Run main physics simulation
        logging.info("Phase 3: Starting ball drop simulation...")
        logging.info(f"Platform scale: {getattr(self.cube_platform, '_scale', 'unknown'):.2f}")
        logging.info(f"Platform friction: {getattr(self.cube_platform, '_friction', 'unknown'):.3f}")
        logging.info(f"Ball ({BALL_NAME}) friction: {getattr(self.ball, '_friction_coefficient', 'unknown'):.3f}, restitution: {getattr(self.ball, '_restitution', 'unknown'):.3f}")
        logging.info(f"Drop height: {getattr(self.ball, 'position', [0, 0, 0])[2]:.2f}")
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
            
        # Check platform (should be static)
        if self.cube_platform in anim_data:
            positions = anim_data[self.cube_platform]["position"]
            velocities = anim_data[self.cube_platform]["velocity"]
            start_pos = positions[0] if positions else "No data"
            end_pos = positions[-1] if positions else "No data"
            max_velocity = max([np.linalg.norm(v) for v in velocities]) if velocities else 0
            logging.info(f"Platform: start={start_pos}, end={end_pos}, max_vel={max_velocity:.3f}")
        else:
            logging.warning(f"Platform not found in animation data")
        
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
    sim = BallDropSimulation(args)
    try:
        sim.run()
    except TimeoutException:
        logging.error("Simulation timed out")
        sys.exit(1) 