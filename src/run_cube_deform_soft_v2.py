import os
import sys; sys.path = ["kubric"] + sys.path
import uuid
import signal
import shutil
import tarfile
import logging
from math import radians

import numpy as np
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from skimage import io
import json
import pickle as pkl
from kubric_utils import *
import bpy

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
parser.add_argument("--vertex_deformation_threshold", type=float, default=0.02)
parser.add_argument("--settle_frames", type=int, default=2)
parser.add_argument("--not_visible_stop_threshold", type=int, default=10)
parser.add_argument("--focal_length", type=float, default=80.0)
parser.add_argument("--sensor_width", type=float, default=32)
parser.add_argument("--camera_elevation_angle", type=float, default=None)
parser.add_argument("--camera_azimuth_angle", type=float, default=None)
parser.add_argument("--force_focal_length", action="store_true", default=False)
parser.add_argument("--composition_style", type=str, default=None, help="Choose composition style for the camera")
## Weight deformation specific configuration
parser.add_argument("--weight_type", type=str, default="ball",
                   help="Type of weight to place on cube: 'ball' or 'plate'")
parser.add_argument("--weight_mass", type=float, default=5.0,
                   help="Mass of the weight object for deformation strength")
parser.add_argument("--cube_mass", type=float, default=1.0,
                   help="Mass of the jelly cube soft body")
parser.add_argument("--cube_friction", type=float, default=1.0,
                   help="Friction coefficient for the jelly cube")
parser.add_argument("--cube_restitution", type=float, default=0.05,
                   help="Restitution for the jelly cube (usually 0 for soft bodies)")
parser.add_argument("--cube_neo_hookean_mu", type=float, default=60.0,
                   help="Shear modulus of the jelly cube neo-hookean model")
parser.add_argument("--cube_neo_hookean_damping", type=float, default=0.1,
                   help="Damping of the jelly cube neo-hookean model")
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
# Asset names for weight deformation
# --------------------------------------------------------------------------------------

WEIGHT_BALL_NAME = "ball_weight_z-0-4"
WEIGHT_PLATE_NAME = "weight_plate_z-0-4"
JELLY_CUBE_BASENAME = "jelly_cube"
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
# Main Weight Deformation Simulation class
# --------------------------------------------------------------------------------------

class WeightDeformSimulation:
    """Simulate a weight deforming a soft jelly cube and render the results."""
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
    # Weight deform scene construction
    # ------------------------------------------------------------------

    def _create_jelly_cube(self):
        """Create a soft jelly cube that deforms under weight."""
        urdf_origin_offset = self._read_urdf_origin_offset("objs/weight_deform/jelly_cube.urdf")

        cube_position = [0.0, 0.0, 0.2]

        cube = kb.SoftBody(
            name="jelly_cube",
            simulation_filename="objs/weight_deform/jelly_cube.vtk",
            render_filename="objs/weight_deform/jelly_cube_sf.obj",
            tri_to_tet_mapping_filename="objs/weight_deform/jelly_cube_mapping.txt",
            urdf_origin_offset=urdf_origin_offset,
            scale=1.0,
            position=cube_position,
            mass=self.args.cube_mass,
            friction=self.args.cube_friction,
            restitution=self.args.cube_restitution,
            segmentation_id=self._get_segmentation_id(),
            use_mass_spring=False,
            use_neo_hookean=True,
            neo_hookean_mu=self.args.cube_neo_hookean_mu,
            neo_hookean_lambda=600,
            neo_hookean_damping=self.args.cube_neo_hookean_damping,
        )

        # Keep stationary initially
        cube.velocity = [0, 0, 0]

        self.scene += cube

        self.jelly_cube = cube

        # Apply a simple material
        self._apply_jelly_textures()
        # self._apply_ball_materials(cube)

        # Store metadata
        cube._neo_hookean_mu = self.args.cube_neo_hookean_mu
        cube._neo_hookean_damping = self.args.cube_neo_hookean_damping

        logging.info(f"Created jelly cube at position {cube_position} with mass {self.args.cube_mass:.2f}")
        return cube

    def _create_weight_object(self):
        """Create a rigid weight (ball or plate) placed above the cube."""
        weight_type = (self.args.weight_type or "ball").lower()
        if weight_type not in ["ball", "plate"]:
            logging.warning(f"Unknown weight_type '{self.args.weight_type}', defaulting to 'ball'")
            weight_type = "ball"

        if weight_type == "ball":
            base_name = WEIGHT_BALL_NAME
        else:
            base_name = WEIGHT_PLATE_NAME

        # Place above cube so it can drop and deform the cube
        # weight_position = [0.0, 0.0, 0.9]
        if weight_type == "ball":
            weight_position = self._read_urdf_origin_offset(f"objs/weight_deform/{base_name}.urdf")
            weight_position = np.array(weight_position) + np.array([0.0, 0.0, 0.2])
            weight_position = tuple(weight_position.tolist())
            urdf_origin_offset = weight_position
        else:
            weight_position = [0.0, 0.0, 0.6]
            urdf_origin_offset = None

        weight = kb.FileBasedObject(
            name=f"weight_{weight_type}",
            simulation_filename=f"objs/weight_deform/{base_name}.urdf",
            render_filename=f"objs/weight_deform/{base_name}.obj",
            scale=1.0,
            position=weight_position,
            urdf_origin_offset=urdf_origin_offset,
            mass= self.args.weight_mass,
            friction=1.0,
            restitution=0.05,
            static=False,
            background=False,
            segmentation_id=self._get_segmentation_id(),
        )

        self.weight_object = weight

        self.scene += weight

        # Give it a material for rendering variety
        if weight_type == "ball":
            self._apply_ball_materials(weight)
        else:
            self._apply_plate_textures()

        logging.info(f"Created '{weight.name}' at position {weight_position} with mass {self.args.weight_mass:.2f}")
        return weight

    
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

    def _apply_jelly_textures(self):
        """Apply randomly selected textures to the cube using improved mapping similar to brick textures"""
        import os
        import random
        
        # Sponge texture base path
        SPONGE_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "sponge_textures")

        # Get the Blender representation of the jelly cube
        jelly_cube_blender = self.jelly_cube.linked_objects[self.renderer]
        
        if not jelly_cube_blender:
            logging.warning("Warning: Jelly cube Blender object not found")
            return False
        
        logging.info(f"Found jelly cube Blender object for texturing: {jelly_cube_blender.name}")
        
        # Get available texture directories (similar to brick texture method)
        try:
            texture_dirs = [d for d in os.listdir(SPONGE_TEXTURE_BASE_PATH)
                           if os.path.isdir(os.path.join(SPONGE_TEXTURE_BASE_PATH, d))]
        except Exception as e:
            logging.error(f"Error reading sponge texture directories: {e}")
            return False
        
        if not texture_dirs:
            logging.error(f"No sponge texture directories found in {SPONGE_TEXTURE_BASE_PATH}")
            return False
        
        # Randomly select a texture using the simulation's RNG for reproducibility
        selected_texture = self.rng.choice(texture_dirs)
        texture_path = os.path.join(SPONGE_TEXTURE_BASE_PATH, selected_texture, "textures")
        
        # Store selected texture info in metadata
        self.metadata["jelly_texture"] = selected_texture.split(".blend")[0]
        
        logging.info(f"Selected jelly texture: {selected_texture}")
        
        # Find texture files with proper 4K naming convention (like other texture methods)
        diffuse_path = None
        normal_path = None
        roughness_path = None
        displacement_path = None
        
        all_files = os.listdir(texture_path)
        for file in all_files:
            file_lower = file.lower()
            file_path = os.path.join(texture_path, file)
            if file_lower.endswith(('.jpg', '.jpeg', '.png', '.exr')):
                if 'diff' in file_lower and not diffuse_path:
                    diffuse_path = file_path
                elif 'nor_gl' in file_lower and not normal_path:
                    normal_path = file_path
                elif 'rough' in file_lower and not roughness_path:
                    roughness_path = file_path
                elif 'disp' in file_lower and not displacement_path:
                    displacement_path = file_path   
        
        # Print found textures for debugging
        logging.info(f"Found diffuse: {os.path.basename(diffuse_path) if diffuse_path else 'None'}")
        logging.info(f"Found normal: {os.path.basename(normal_path) if normal_path else 'None'}")
        logging.info(f"Found roughness: {os.path.basename(roughness_path) if roughness_path else 'None'}")
        logging.info(f"Found displacement: {os.path.basename(displacement_path) if displacement_path else 'None'}")
        
        # Get or create the material for the jelly cube (similar to brick texture method)
        if len(jelly_cube_blender.material_slots) == 0:
            mat = bpy.data.materials.new(name="Jelly_Material")
            jelly_cube_blender.data.materials.append(mat)
        else:
            mat = jelly_cube_blender.material_slots[0].material
            if not mat:
                mat = bpy.data.materials.new(name="Jelly_Material")
                jelly_cube_blender.material_slots[0].material = mat
        
        # Enable nodes for the material
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing nodes (using brick texture method)
        while nodes:
            nodes.remove(nodes[0])
        
        # Create output and principled nodes (like brick texture method)
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (100, 0)
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # Add texture coordinate and mapping nodes with better scaling
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-800, 0)
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-600, 0)
        # Use larger scale like platform textures to avoid streaking on cube sides
        mapping.inputs['Scale'].default_value = (20.0, 20.0, 20.0)
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        
        # Create and link texture nodes using brick texture method (without box projection)
        # Diffuse Texture
        # if diffuse_path and os.path.exists(diffuse_path):
        #     tex_diff = nodes.new(type='ShaderNodeTexImage')
        #     tex_diff.location = (-400, 200)
        #     tex_diff.image = bpy.data.images.load(diffuse_path)
        #     links.new(mapping.outputs['Vector'], tex_diff.inputs['Vector'])
        #     links.new(tex_diff.outputs['Color'], principled.inputs['Base Color'])
        #     logging.info(f"Applied jelly diffuse texture: {diffuse_path}")
        # else:
        #     logging.warning("Warning: Jelly diffuse texture not found")
        #     # Set a random fallback color to make the material visible
        #     rand_color = (random.uniform(0.2, 1.0), random.uniform(0.2, 1.0), random.uniform(0.2, 1.0), 1.0)
        #     principled.inputs['Base Color'].default_value = rand_color
        #     logging.info(f"Set random base color for jelly: {rand_color}")
        rand_color = (np.random.uniform(0.2, 1.0), np.random.uniform(0.2, 1.0), np.random.uniform(0.2, 1.0), 1.0)
        principled.inputs['Base Color'].default_value = rand_color
        logging.info(f"Set random base color for jelly: {rand_color}")
        
        # Normal Texture
        if normal_path and os.path.exists(normal_path):
            tex_nor = nodes.new(type='ShaderNodeTexImage')
            tex_nor.location = (-400, 0)
            tex_nor.image = bpy.data.images.load(normal_path)
            tex_nor.image.colorspace_settings.name = 'Non-Color'
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-200, 0)
            links.new(mapping.outputs['Vector'], tex_nor.inputs['Vector'])
            links.new(tex_nor.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
            logging.info(f"Applied jelly normal texture: {normal_path}")
        else:
            logging.warning("Warning: Jelly normal texture not found")
        
        # Roughness Texture
        if roughness_path and os.path.exists(roughness_path):
            tex_rough = nodes.new(type='ShaderNodeTexImage')
            tex_rough.location = (-400, -200)
            tex_rough.image = bpy.data.images.load(roughness_path)
            tex_rough.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_rough.inputs['Vector'])
            links.new(tex_rough.outputs['Color'], principled.inputs['Roughness'])
            logging.info(f"Applied jelly roughness texture: {roughness_path}")
        else:
            logging.warning("Warning: Jelly roughness texture not found")
        
        # Displacement Texture
        if displacement_path and os.path.exists(displacement_path):
            tex_disp = nodes.new(type='ShaderNodeTexImage')
            tex_disp.location = (-400, -400)
            tex_disp.image = bpy.data.images.load(displacement_path)
            tex_disp.image.colorspace_settings.name = 'Non-Color'
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = (-200, -400)
            disp_node.inputs['Scale'].default_value = 0.05  # Small scale to avoid rendering issues
            links.new(mapping.outputs['Vector'], tex_disp.inputs['Vector'])
            links.new(tex_disp.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
            mat.cycles.displacement_method = 'BOTH'
            logging.info(f"Applied jelly displacement texture: {displacement_path}")
        else:
            logging.warning("Warning: Jelly displacement texture not found")

        # # Set additional material properties to ensure visibility
        # principled.inputs['Roughness'].default_value = 0.5
        # principled.inputs['Specular'].default_value = 0.5
        
        # Force material update in Blender
        jelly_cube_blender.data.update()
        
        # Ensure the material is active
        if jelly_cube_blender.active_material != mat:
            jelly_cube_blender.active_material = mat
        
        logging.info("Jelly texture application completed successfully")
        logging.info(f"Material '{mat.name}' applied to object '{jelly_cube_blender.name}'")
        
        return True


    def _apply_plate_textures(self):
        """Apply randomly selected plate textures to the plate"""
        import os
        import random
        
        # Ground texture base path
        RUBBER_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "rubber_textures")

        # Select wood or concrete texture
        texture_type = self.rng.choice(['rubber'])
        texture_base_path = RUBBER_TEXTURE_BASE_PATH
            
        # Get the Blender representation of the ground plane
        weight_object_blender = self.weight_object.linked_objects[self.renderer]
        
        if not weight_object_blender:
            logging.warning("Warning: Weight object Blender object not found")
            return False
        
        logging.info(f"Found weight object Blender object for texturing: {weight_object_blender.name}")
        
        # Get available ground texture directories
        material_types = []
        try:
            material_types = [d for d in os.listdir(texture_base_path) 
                           if os.path.isdir(os.path.join(texture_base_path, d))]
        except Exception as e:
            logging.error(f"Error reading material texture directories: {e}")
            return False
        
        if not material_types:
            logging.error(f"No material texture directories found in {texture_base_path}")
            return False
        
        # Randomly select a ground type using the simulation's RNG for reproducibility
        selected_material = self.rng.choice(material_types)
        texture_path = os.path.join(texture_base_path, selected_material, "textures")
        
        # Store selected texture info in metadata
        self.metadata["material_texture"] = selected_material.split(".blend")[0]
        
        logging.info(f"Selected material texture: {selected_material}")
        
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
                if "diff_4k" in file_lower and file_lower.endswith(".jpg") or file_lower.endswith(".png"):
                    diffuse_path = os.path.join(texture_path, file)
                
                # Normal texture (always exr)
                elif "nor_gl_4k" in file_lower and file_lower.endswith(".exr") or file_lower.endswith(".png"):
                    normal_path = os.path.join(texture_path, file)
                
                # Roughness texture (could be jpg or exr)
                elif "rough_4k" in file_lower:
                    if file_lower.endswith(".jpg") or file_lower.endswith(".exr") or file_lower.endswith(".png"):
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
        if len(weight_object_blender.material_slots) == 0:
            # Create new material if none exists
            mat = bpy.data.materials.new(name="Weight_Material")
            weight_object_blender.data.materials.append(mat)
        else:
            mat = weight_object_blender.material_slots[0].material
            if not mat:
                mat = bpy.data.materials.new(name="Weight_Material")
                weight_object_blender.material_slots[0].material = mat
        
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
            logging.info(f"Applied weight diffuse texture: {diffuse_path}")
        else:
            logging.warning(f"Warning: Weight diffuse texture not found")
        
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
            logging.info(f"Applied weight normal texture: {normal_path}")
        else:
            logging.warning(f"Warning: Weight normal texture not found")
        
        # Roughness Texture
        if roughness_path and os.path.exists(roughness_path):
            tex_roughness = nodes.new(type='ShaderNodeTexImage')
            tex_roughness.location = (-400, -200)
            tex_roughness.image = bpy.data.images.load(roughness_path)
            # Set correct color space for roughness maps
            tex_roughness.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_roughness.inputs['Vector'])
            links.new(tex_roughness.outputs['Color'], principled.inputs['Roughness'])
            logging.info(f"Applied weight roughness texture: {roughness_path}")
        else:
            logging.warning(f"Warning: Weight roughness texture not found")
        
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
            
            logging.info(f"Applied weight displacement texture: {displacement_path}")
        else:
            logging.warning(f"Warning: Weight displacement texture not found")

        # Physics properties are handled by Kubric/PyBullet separately
        # No need to set up Blender rigid body physics
        logging.info("Weight texture application completed successfully")
        
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
        
        # (Removed ball/platform objects)

        # Add jelly cube
        if hasattr(self, 'jelly_cube') and self.jelly_cube:
            all_objects.append(('jelly_cube', self.jelly_cube))

        # Add weight
        if hasattr(self, 'weight') and self.weight:
            all_objects.append(('weight', self.weight))
        
        
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
                if obj_type == "jelly_cube":
                    self.metadata["object_data"]["object_of_interest"] = obj_metadata["segmentation_id"]
                
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
            azimuth_angle = np.radians(self.rng.uniform(-70, -110))

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


    def _verify_trajectory_framing(self, *_args, **_kwargs):
        return



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
        test_objects = ['jelly_cube', 'weight']
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
            self.simulator.pause_for_inspection("Scene setup complete. Check weight deformation setup and camera angle.")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    @time_limit(3000)
    def run(self):
        # Phase 1: Setup background and ground plane
        self._setup_background_and_plane()

        # Phase 2: Create deformable cube and weight
        if self.args.debug_gui:
            self.debug_pause_at_key_points()

        self.jelly_cube = self._create_jelly_cube()
        self.simulator.run(frame_start=-1, frame_end=0)
        self.weight = self._create_weight_object()

        # Collect objects and setup camera
        all_objects = [self.jelly_cube, self.weight]

        if self.args.debug_gui:
            self.debug_pause_at_key_points()

        # Collect metadata
        self._collect_all_object_metadata()

        # Camera
        self._setup_camera_with_blender_align(all_objects)

        # Store metadata about the simulation setup
        self.metadata["simulation_type"] = "weight_deform"

        if self.args.debug_gui and hasattr(self.simulator, 'pause_for_inspection'):
            self.simulator.pause_for_inspection("About to start weight deform simulation.")

        logging.info("Starting weight deform simulation (jelly cube + weight)...")
        logging.info(f"Weight: type={self.args.weight_type}, mass={self.args.weight_mass:.2f}")
        logging.info(f"Running simulation for {self.args.frame_end + 1} frames")
        anim_data, _ = self.simulator.run(frame_start=0, frame_end=self.args.frame_end + 1)
        
        # Debug: Check if objects actually moved
        logging.info("Simulation complete. Checking movement...")
        
        # Check jelly cube and weight movement
        if hasattr(self, 'jelly_cube') and self.jelly_cube in anim_data:
            positions = anim_data[self.jelly_cube]["position"]
            start_pos = positions[0] if positions else "No data"
            end_pos = positions[-1] if positions else "No data"
            logging.info(f"Jelly cube: start={start_pos}, end={end_pos}")
        if hasattr(self, 'weight') and self.weight in anim_data:
            positions = anim_data[self.weight]["position"]
            velocities = anim_data[self.weight]["velocity"]
            start_pos = positions[0] if positions else "No data"
            end_pos = positions[-1] if positions else "No data"
            max_velocity = max([np.linalg.norm(v) for v in velocities]) if velocities else 0
            logging.info(f"Weight: start={start_pos}, end={end_pos}, max_vel={max_velocity:.3f}")
        
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
    sim = WeightDeformSimulation(args)
    try:
        sim.run()
    except TimeoutException:
        logging.error("Simulation timed out")
        sys.exit(1) 