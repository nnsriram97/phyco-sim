import os
import sys; sys.path = ["kubric"] + sys.path
import uuid
import signal
import shutil
import tarfile
import logging
from math import radians
from typing import List, Dict

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

SIM_ASSETS_DIR = os.environ.get("SIM_ASSETS_DIR", "./sim_assets")

MOVI_SHAPES = ["cube", "cylinder", "sphere", "cone", "torus", "gear",
               "torus_knot", "sponge", "spot", "teapot", "suzanne"]

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

def world_to_camera_coordinates(
    world_point,
    camera_position,
    camera_rotation,
    focal_length,
    sensor_width,
    image_width,
    image_height,
):
    """Convert world coordinates to camera/image coordinates."""
    world_point = np.array(world_point)
    camera_position = np.array(camera_position)
    point_relative = world_point - camera_position

    if hasattr(camera_rotation, "to_matrix"):
        rotation_matrix = np.array(camera_rotation.to_matrix())
    elif isinstance(camera_rotation, (list, tuple, np.ndarray)) and len(camera_rotation) == 4:
        quat = Quaternion(camera_rotation)
        rotation_matrix = np.array(quat.to_matrix())
    else:
        rotation_matrix = np.array(camera_rotation)

    camera_point = rotation_matrix.T @ point_relative
    if camera_point[2] >= 0:
        return None, None, None

    focal_length_m = focal_length / 1000.0
    sensor_width_m = sensor_width / 1000.0
    aspect_ratio = image_width / image_height
    sensor_height_m = sensor_width_m / aspect_ratio

    x_ndc = (camera_point[0] * focal_length_m) / (-camera_point[2] * sensor_width_m / 2.0)
    y_ndc = (camera_point[1] * focal_length_m) / (-camera_point[2] * sensor_height_m / 2.0)

    image_x = (x_ndc + 1.0) * 0.5 * image_width
    image_y = (1.0 - y_ndc) * 0.5 * image_height
    depth = -camera_point[2]

    return image_x, image_y, depth


def create_velocity_visualization(
    image,
    force_point_world,
    force_vector_world,
    camera_position,
    camera_rotation,
    focal_length,
    sensor_width,
    force_scale=0.1,
):
    """Overlay a velocity arrow on an image."""
    image_height, image_width = image.shape[:2]
    annotated_image = image.copy()

    # Determine camera rotation matrix for later reuse
    if hasattr(camera_rotation, "to_matrix"):
        rotation_matrix = np.array(camera_rotation.to_matrix())
    elif isinstance(camera_rotation, (list, tuple, np.ndarray)) and len(camera_rotation) == 4:
        rotation_matrix = np.array(Quaternion(camera_rotation).to_matrix())
    else:
        rotation_matrix = np.array(camera_rotation)

    point_x, point_y, point_depth = world_to_camera_coordinates(
        force_point_world,
        camera_position,
        camera_rotation,
        focal_length,
        sensor_width,
        image_width,
        image_height,
    )

    velocity_metadata = {
        "velocity_point_world": list(force_point_world),
        "velocity_vector_world": list(force_vector_world),
        "velocity_magnitude": float(np.linalg.norm(force_vector_world)),
        "camera_position": list(camera_position),
        "focal_length": focal_length,
        "sensor_width": sensor_width,
    }

    if point_x is None or point_y is None:
        velocity_metadata["visible"] = False
        velocity_metadata["reason"] = "behind_camera"
        return annotated_image, velocity_metadata

    if point_x < 0 or point_x >= image_width or point_y < 0 or point_y >= image_height:
        velocity_metadata["visible"] = False
        velocity_metadata["reason"] = "outside_image_bounds"
        velocity_metadata["image_coordinates"] = [float(point_x), float(point_y)]
        return annotated_image, velocity_metadata

    velocity_metadata["visible"] = True
    velocity_metadata["image_coordinates"] = [float(point_x), float(point_y)]
    velocity_metadata["depth"] = float(point_depth)

    force_end_world = np.array(force_point_world) + np.array(force_vector_world) * force_scale
    end_x, end_y, _ = world_to_camera_coordinates(
        force_end_world,
        camera_position,
        camera_rotation,
        focal_length,
        sensor_width,
        image_width,
        image_height,
    )

    pt1 = (int(point_x), int(point_y))
    cv2.circle(annotated_image, pt1, 8, (255, 0, 0), 2)
    cv2.circle(annotated_image, pt1, 3, (255, 255, 255), -1)

    # Determine arrow direction in image space
    arrow_vector = None
    if end_x is not None and end_y is not None:
        arrow_vector = np.array([float(end_x) - point_x, float(end_y) - point_y], dtype=float)

    if arrow_vector is None or np.linalg.norm(arrow_vector) < 1e-6:
        force_vec_world = np.array(force_vector_world, dtype=float)
        if rotation_matrix.size:
            camera_force = rotation_matrix.T @ force_vec_world
            arrow_vector = np.array([camera_force[0], -camera_force[1]], dtype=float)
        else:
            arrow_vector = np.array([force_vec_world[0], -force_vec_world[1]], dtype=float)

    if np.linalg.norm(arrow_vector) < 1e-6:
        arrow_vector = np.array([1.0, 0.0], dtype=float)

    arrow_unit = arrow_vector / np.linalg.norm(arrow_vector)
    base_point = np.array([point_x, point_y], dtype=float)
    desired_length = max(image_width, image_height) * 0.12
    arrow_end = base_point + arrow_unit * desired_length
    arrow_end = np.clip(arrow_end, [0.0, 0.0], [image_width - 1.0, image_height - 1.0])

    actual_vector = arrow_end - base_point
    actual_length = np.linalg.norm(actual_vector)
    if actual_length < 1.0:
        arrow_end = base_point + arrow_unit
        arrow_end = np.clip(arrow_end, [0.0, 0.0], [image_width - 1.0, image_height - 1.0])
        actual_vector = arrow_end - base_point
        actual_length = np.linalg.norm(actual_vector)

    pt2 = (int(round(arrow_end[0])), int(round(arrow_end[1])))
    cv2.arrowedLine(annotated_image, pt1, pt2, (0, 255, 0), 3, tipLength=0.3)

    velocity_metadata["velocity_end_image_coordinates"] = [float(arrow_end[0]), float(arrow_end[1])]
    velocity_metadata["velocity_arrow_unit_vector"] = [float(arrow_unit[0]), float(arrow_unit[1])]
    velocity_metadata["velocity_arrow_length_pixels"] = float(actual_length)

    velocity_mag = np.linalg.norm(force_vector_world)
    text = f"|v|={velocity_mag:.2f}m/s"
    cv2.putText(
        annotated_image,
        text,
        (int(point_x) + 10, int(point_y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return annotated_image, velocity_metadata

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
parser.add_argument("--scenario", type=str, default="friction_slide_flat")
parser.add_argument("--vary_friction_only", action="store_true", default=True,
                   help="Keep mass and restitution constant, vary only friction coefficient")
parser.add_argument("--prism_friction", type=float, default=None)
parser.add_argument("--object_friction", type=float, default=None)
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

# Use a fixed brick for the flat platform simulation
PLATFORM_BRICK = "brick_slide_x_2-0"
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
        
        # Initialize persistent forces and velocities
        self._persistent_forces = []
        self._persistent_velocities = []
        
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

class FrictionSlideFlatSimulation:
    """Generate videos of friction sliding simulation with objects on a flat platform."""
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
        self.metadata: Dict[str, dict] = {"object_data": {}}
        self.applied_velocities: List[dict] = []
        self._applied_velocity_vector = None
        self._applied_velocity_point = None

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
        """Apply randomly selected ground textures to the plane using a more correct node setup."""
        import os

        # Texture base paths
        GROUND_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "ground_textures")
        WOOD_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "wood_textures")
        CONCRETE_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "concrete_textures")

        # Access the ground plane's Blender object through Kubric's linked objects
        if not hasattr(self, 'dome') or not self.dome:
            logging.warning("Warning: Ground plane object not available for texture application")
            return False

        ground_plane_blender = self.dome.linked_objects[self.renderer]
        if not ground_plane_blender:
            logging.warning("Warning: Ground plane Blender object not found")
            return False

        logging.info(f"Found ground plane Blender object for texturing: {ground_plane_blender.name}")

        # Randomly choose between wood and concrete textures
        texture_type = self.rng.choice(['wood', 'concrete', 'ground'])
        if texture_type == 'wood':
            texture_base_path = WOOD_TEXTURE_BASE_PATH
        elif texture_type == 'concrete':
            texture_base_path = CONCRETE_TEXTURE_BASE_PATH
        else:
            texture_base_path = GROUND_TEXTURE_BASE_PATH

        # Get available texture directories
        try:
            texture_dirs = [d for d in os.listdir(texture_base_path)
                            if os.path.isdir(os.path.join(texture_base_path, d)) and d.endswith('.blend')]
        except Exception as e:
            logging.error(f"Error reading ground texture directories: {e}")
            return False

        if not texture_dirs:
            logging.error(f"No ground texture directories found in {texture_base_path}")
            return False

        selected_texture = self.rng.choice(texture_dirs)
        texture_path = os.path.join(texture_base_path, selected_texture, "textures")

        # Store selected texture info in metadata
        self.metadata["ground_texture_type"] = texture_type
        self.metadata["ground_texture"] = selected_texture.split(".blend")[0]
        logging.info(f"Selected ground texture: {texture_type} - {selected_texture}")

        # Find texture files
        diffuse_path = None
        normal_path = None
        roughness_path = None
        displacement_path = None
        if os.path.exists(texture_path):
            try:
                all_files = os.listdir(texture_path)
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
                logging.error(f"Error finding ground texture files: {e}")
                return False
        else:
            logging.error(f"Texture path does not exist: {texture_path}")
            return False

        logging.info(f"Found diffuse: {os.path.basename(diffuse_path) if diffuse_path else 'None'}")
        logging.info(f"Found normal: {os.path.basename(normal_path) if normal_path else 'None'}")
        logging.info(f"Found roughness: {os.path.basename(roughness_path) if roughness_path else 'None'}")
        logging.info(f"Found displacement: {os.path.basename(displacement_path) if displacement_path else 'None'}")

        # Get or create the material for the plane
        if len(ground_plane_blender.material_slots) == 0:
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
        for node in list(nodes):
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
        mapping.inputs['Scale'].default_value[0] = 5.0
        mapping.inputs['Scale'].default_value[1] = 5.0
        mapping.inputs['Scale'].default_value[2] = 1.0

        # Use Generated coordinates instead of UV (since plane.obj has no UV data)
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])

        # Diffuse Texture
        if diffuse_path and os.path.exists(diffuse_path):
            tex_diffuse = nodes.new(type='ShaderNodeTexImage')
            tex_diffuse.location = (-400, 200)
            tex_diffuse.image = bpy.data.images.load(diffuse_path)
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
            tex_disp.image.colorspace_settings.name = 'Non-Color'
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = (-200, -400)
            disp_node.inputs['Scale'].default_value = 0.05
            links.new(mapping.outputs['Vector'], tex_disp.inputs['Vector'])
            links.new(tex_disp.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
            mat.cycles.displacement_method = 'BOTH'
            logging.info(f"Applied ground displacement texture: {displacement_path}")
        else:
            logging.warning(f"Warning: Ground displacement texture not found")

        logging.info("Ground texture application completed successfully")
        return True

    def _apply_platform_textures(self):
        """Apply randomly selected textures to the cube platform"""
        import os
        import bpy

        # Platform texture base paths
        WOOD_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "wood_textures")
        CONCRETE_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "concrete_textures")

        # Access the cube platform's Blender object through Kubric's linked objects
        if not hasattr(self, 'cube_platform') or not self.cube_platform:
            logging.warning("Warning: Cube platform object not available for texture application")
            return False

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

            # Enable displacement in material settings
            mat.cycles.displacement_method = 'BOTH'  # Using both displacement and bump

            logging.info(f"Applied platform displacement texture: {displacement_path}")
        else:
            logging.warning(f"Warning: Platform displacement texture not found")

        # Physics properties are handled by Kubric/PyBullet separately
        # No need to set up Blender rigid body physics
        logging.info("Platform texture application completed successfully")

        return True

    def _apply_brick_textures(self, brick_blender):
        """Apply brick textures to the brick object using a node setup similar to _apply_brick_textures in run_friction_slide_flat_force.py."""
        import os
        import bpy

        BRICK_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "brick_textures")

        if not brick_blender:
            logging.warning("Warning: Brick Blender object not found")
            return False

        logging.info(f"Found brick Blender object for texturing: {brick_blender.name}")

        # (Optional) Apply scale and UV unwrap to avoid stretched textures
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
        for node in list(nodes):
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

        logging.info("Brick texture application completed successfully")

        return True


    def _create_cube_platform(self):
        """Create a flat cube platform for the sliding surface using cube_platform.urdf."""
        # Scale and rotation for the cube platform
        platform_scale = 1.0
        
        if self.args.prism_friction is not None:
            platform_friction = self.args.prism_friction
        else:
            platform_friction = self.rng.uniform(0.05, 1.0)
        # platform_friction = 0.2
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
        

    def _create_sliding_object(self, platform):
        """Create a sliding brick object that will slide on the platform."""
        
        # Keep mass and restitution constant as requested
        object_mass = 1.0  # Constant mass
        
        # Vary only the lateral friction component as requested
        if self.args.object_friction is not None:
            object_friction = self.args.object_friction
        else:
            object_friction = self.rng.uniform(0.05, 1.0)  # Variable friction

        # object_friction = 0.2
        
        # Use the specified brick for sliding
        brick_name = PLATFORM_BRICK
        self.metadata["brick_name"] = brick_name

        # Account for the URDF origin offset
        brick_position = self._read_urdf_origin_offset(f"objs/bricks/{brick_name}.urdf")
        # if brick_position is None:
        #     logging.warning(f"Warning: Brick origin offset not found for {brick_name}. Using default position.")
        #     # Position the brick on top of the platform with some clearance
        #     brick_position = [2.0, 0.0, 1.0]  # Start the brick at positive x position on the platform
        # else:
        #     # Adjust the position to be on top of the platform
        #     brick_position = [2.0, brick_position[1], brick_position[2]]
        
        brick = kb.FileBasedObject(
            name="brick",
            simulation_filename=f"objs/bricks/{brick_name}.urdf",
            render_filename=f"objs/bricks/{brick_name}.obj",
            scale=1.0,
            position=brick_position,
            mass=object_mass,
            friction=object_friction,  # Base friction for the brick
            restitution=0.0,  # Keep constant as requested
            segmentation_id=self._get_segmentation_id()
        )
        
        # Add initial velocity in positive y direction
        initial_velocity = [0.0, 2.0, 0.0]  # Positive y direction
        brick.velocity = initial_velocity
        
        # Store velocity information for visualization
        self._applied_velocity_vector = np.array(initial_velocity)
        self._applied_velocity_point = np.array(brick_position)
        
        # Add to scene
        self.scene += brick

        # If this object has urdf_origin_offset then we need to apply it to the position in the blender object
        if hasattr(brick, 'urdf_origin_offset'):
            brick_position_blender = brick_position - np.array(brick.urdf_origin_offset)
            blender_obj = brick.linked_objects[self.renderer]
            blender_obj.location = brick_position_blender
        
        # Apply brick textures
        self._apply_brick_textures(brick.linked_objects[self.renderer])
        
        logging.info(f"Created sliding brick at position {brick_position} with friction {object_friction:.3f}")
        logging.info(f"Applied initial velocity {initial_velocity} to brick")
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
        
        # Add cube platform
        if hasattr(self, 'cube_platform') and self.cube_platform:
            all_objects.append(('cube_platform', self.cube_platform))
        
        # Add sliding object
        if hasattr(self, 'sliding_object') and self.sliding_object:
            all_objects.append(('sliding_object', self.sliding_object))
        
        
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
                    obj_metadata["platform_scale"] = obj._scale
                if hasattr(obj, '_friction'):
                    obj_metadata["platform_friction"] = obj._friction
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
        """Setup camera to frame both the brick and origin (0,0,0) at the ends of the view."""
        # Create camera with initial parameters
        focal_length = self.args.focal_length if self.args.focal_length is not None else 50
        self.scene.camera = kb.PerspectiveCamera(focal_length=focal_length)
        
        # Get brick position (assuming it's the sliding object)
        brick_position = np.array([2.0, 0.0, 0.0])  # Default fallback
        if hasattr(self, 'sliding_object') and self.sliding_object:
            brick_position = np.array(self.sliding_object.position)
        
        # Origin position
        origin_position = np.array([0.0, 0.0, 0.0])
        
        # Calculate the center point between brick and origin
        scene_center = (brick_position + origin_position) / 2.0
        
        # Calculate the distance between brick and origin
        brick_to_origin_distance = np.linalg.norm(brick_position - origin_position)
        
        # Get camera angles
        if self.args.camera_elevation_angle is not None:
            elevation_angle = np.radians(self.args.camera_elevation_angle)
        else:
            elevation_angle = np.radians(self.rng.uniform(15, 70))
            
        if self.args.camera_azimuth_angle is not None:
            azimuth_angle = np.radians(self.args.camera_azimuth_angle)
        else:
            azimuth_angle = np.radians(self.rng.uniform(0, 360))  # Default to 0 for side view
            # azimuth_angle = np.radians(-90)
        
        # Calculate required camera distance based on focal length and sensor dimensions
        sensor_width = self.args.sensor_width if hasattr(self.args, 'sensor_width') else 32.0
        
        # Calculate sensor height based on scene resolution aspect ratio
        # This matches kubric's internal calculation
        scene_resolution = getattr(self.scene, 'resolution', [512, 512])  # Default resolution
        aspect_ratio = scene_resolution[1] / scene_resolution[0]  # height / width
        sensor_height = sensor_width * aspect_ratio
        
        # Calculate horizontal and vertical field of view in radians
        horizontal_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        vertical_fov = 2 * np.arctan(sensor_height / (2 * focal_length))
        
        # Calculate initial camera position for distance calculation
        # We'll refine this distance based on the viewing geometry
        temp_distance = 5.0  # Temporary distance for calculating view vectors
        
        # Calculate temporary camera position using spherical coordinates
        temp_x = temp_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
        temp_y = temp_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
        temp_z = temp_distance * np.sin(elevation_angle)
        temp_camera_position = scene_center + np.array([temp_x, temp_y, temp_z])
        
        # Calculate camera coordinate system
        # View direction (from camera to scene center)
        view_direction = scene_center - temp_camera_position
        view_direction = view_direction / np.linalg.norm(view_direction)
        
        # Up vector (world Z-axis projected onto plane perpendicular to view direction)
        world_up = np.array([0, 0, 1])
        right = np.cross(view_direction, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)
        
        # Project brick and origin positions onto camera's image plane
        brick_rel = brick_position - scene_center
        origin_rel = origin_position - scene_center
        
        # Project onto camera right and up axes
        brick_right = np.dot(brick_rel, right)
        brick_up = np.dot(brick_rel, up)
        origin_right = np.dot(origin_rel, right)
        origin_up = np.dot(origin_rel, up)
        
        # Calculate required field of view to contain both points
        max_right = max(abs(brick_right), abs(origin_right))
        max_up = max(abs(brick_up), abs(origin_up))
        
        # Calculate distances needed for horizontal and vertical constraints
        half_horizontal_fov = horizontal_fov / 2.0
        half_vertical_fov = vertical_fov / 2.0
        
        # Distance needed to fit horizontal span
        distance_for_horizontal = max_right / np.tan(half_horizontal_fov) if max_right > 0 else 0
        
        # Distance needed to fit vertical span  
        distance_for_vertical = max_up / np.tan(half_vertical_fov) if max_up > 0 else 0
        
        # Use the larger distance to ensure both constraints are satisfied
        # Add a minimum distance to avoid camera being too close
        min_distance = 1.0
        required_distance = max(distance_for_horizontal, distance_for_vertical, min_distance)
        
        # Add margin for safety
        margin_factor = 1.2
        initial_distance = required_distance * margin_factor
        
        # Calculate camera position using spherical coordinates
        x = initial_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
        y = initial_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
        z = initial_distance * np.sin(elevation_angle)
        
        # Position camera relative to scene center
        camera_position = scene_center + np.array([x, y, z])
        
        # Log camera setup information
        logging.info(f"Brick position: {brick_position}")
        logging.info(f"Origin position: {origin_position}")
        logging.info(f"Scene center: {scene_center}")
        logging.info(f"Brick-to-origin distance: {brick_to_origin_distance:.2f}")
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
        
        
        # Skip Blender camera alignment since we're using precise mathematical positioning
        # to frame the brick and origin at the ends of the view

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

    # ------------------------------------------------------------------
    # Velocity visualization helpers
    # ------------------------------------------------------------------

    def add_velocity_application(self, object_name, velocity_point_world, velocity_vector_world, frame=0):
        """Record a velocity application for later visualization."""
        velocity_data = {
            "object_name": object_name,
            "velocity_point_world": list(velocity_point_world),
            "velocity_vector_world": list(velocity_vector_world),
            "velocity_magnitude": float(np.linalg.norm(velocity_vector_world)),
            "application_frame": frame,
            "timestamp": f"frame_{frame}",
        }
        self.applied_velocities.append(velocity_data)
        logging.info(
            "Recorded velocity application: %.2fm/s at %s on %s",
            velocity_data["velocity_magnitude"],
            velocity_point_world,
            object_name,
        )

    def create_velocity_annotated_image(self, image_path, output_path=None):
        """Create a velocity-annotated image showing the applied velocity arrow."""
        if not self.applied_velocities:
            logging.warning("No applied velocities recorded for visualization")
            return None, None

        try:
            image = io.imread(image_path)
            if image.shape[-1] == 4:
                image = image[..., :3]
        except Exception as exc:
            logging.error("Failed to load image %s: %s", image_path, exc)
            return None, None

        camera = self.scene.camera
        camera_position = np.array(camera.position)
        camera_rotation = camera.quaternion if hasattr(camera, "quaternion") else None
        if camera_rotation is None:
            camera_rotation = np.eye(3)

        focal_length = camera.focal_length if hasattr(camera, "focal_length") else self.args.focal_length
        sensor_width = camera.sensor_width if hasattr(camera, "sensor_width") else self.args.sensor_width

        annotated_image = image.copy()
        velocity_viz_metadata = []

        for velocity in self.applied_velocities:
            annotated_image, metadata = create_velocity_visualization(
                annotated_image,
                force_point_world=velocity["velocity_point_world"],
                force_vector_world=velocity["velocity_vector_world"],
                camera_position=camera_position,
                camera_rotation=camera_rotation,
                focal_length=focal_length,
                sensor_width=sensor_width,
            )
            velocity_viz_metadata.append(metadata)

        if output_path is not None:
            io.imsave(output_path, annotated_image)

        return annotated_image, velocity_viz_metadata

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
        
        # Run simulation to settle the platform (though it's static)
        logging.info("Phase 1: Settling cube platform...")
        self.simulator.run(frame_start=-50, frame_end=0)
        
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()

        # Phase 2: Add sliding object
        self.sliding_object = self._create_sliding_object(self.cube_platform)
        
        # Collect all objects for camera and metadata
        all_objects = [self.cube_platform, self.sliding_object]
        
        # Debug pause after scene setup if GUI is enabled
        if self.args.debug_gui:
            self.debug_pause_at_key_points()
        
        # Collect metadata for all objects after all modifications are complete
        self._collect_all_object_metadata()
        
        # Setup camera to frame brick and origin at the ends of the view
        self._setup_camera_with_blender_align(all_objects)
        
        # Store metadata about the simulation setup
        self.metadata["simulation_type"] = "friction_slide_flat"
        self.metadata["platform_scale"] = getattr(self.cube_platform, '_scale', 'unknown')
        self.metadata["platform_friction"] = getattr(self.cube_platform, '_friction', 'unknown')
        self.metadata["sliding_object_friction"] = getattr(self.sliding_object, '_friction_coefficient', 'unknown')
        self.metadata["sliding_object_type"] = getattr(self.sliding_object, '_object_type', 'unknown')
        
        # Another debug pause before physics simulation starts
        if self.args.debug_gui and hasattr(self.simulator, 'pause_for_inspection'):
            self.simulator.pause_for_inspection("About to start friction sliding simulation.")
        
        # Phase 3: Run main physics simulation
        logging.info("Phase 2: Starting friction sliding simulation...")
        logging.info(f"Platform scale: {getattr(self.cube_platform, '_scale', 'unknown'):.2f}")
        logging.info(f"Platform friction: {getattr(self.cube_platform, '_friction', 'unknown'):.3f}")
        logging.info(f"Sliding object type: {getattr(self.sliding_object, '_object_type', 'unknown')}")
        logging.info(f"Sliding object friction: {getattr(self.sliding_object, '_friction_coefficient', 'unknown')}")
        logging.info(f"Running simulation for {self.args.frame_end + 1} frames")
        anim_data, _ = self.simulator.run(frame_start=0, frame_end=self.args.frame_end + 1)
        
        # Debug: Check if objects actually moved
        logging.info("Simulation complete. Checking movement...")
        
        # Check sliding object movement
        if self.sliding_object in anim_data:
            positions = anim_data[self.sliding_object]["position"]
            velocities = anim_data[self.sliding_object]["velocity"]
            start_pos = positions[0] if positions else "No data"
            end_pos = positions[-1] if positions else "No data"
            max_velocity = max([np.linalg.norm(v) for v in velocities]) if velocities else 0
            logging.info(f"Sliding object: start={start_pos}, end={end_pos}, max_vel={max_velocity:.3f}")
            
            # Record velocity application for visualization
            if self._applied_velocity_point is not None and self._applied_velocity_vector is not None:
                self.add_velocity_application(
                    object_name=self.sliding_object.name,
                    velocity_point_world=self._applied_velocity_point,
                    velocity_vector_world=self._applied_velocity_vector,
                    frame=0,
                )
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
                
                # Create velocity-annotated version for first frame if velocities were applied
                if i == 0 and self.applied_velocities:
                    try:
                        annotated_image, velocity_viz_metadata = self.create_velocity_annotated_image(
                            image_path, 
                            output_path=os.path.join(self.output_dir, f"velocity_annotated_{i:05d}.jpg")
                        )
                        
                        if velocity_viz_metadata:
                            # Store velocity visualization metadata
                            self.metadata["applied_velocities_image"] = velocity_viz_metadata
                            logging.info("🎯 Created velocity-annotated first frame for visualization")
                        else:
                            logging.warning("Failed to create velocity annotation for first frame")
                            
                    except Exception as e:
                        logging.error(f"Error creating velocity annotation: {e}")
                        
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

        # Add velocity metadata
        if self.applied_velocities:
            self.metadata["applied_velocities_simulator"] = convert_numpy_types(self.applied_velocities)
        
        if self._applied_velocity_vector is not None and self._applied_velocity_point is not None:
            self.metadata["velocity_profile"] = {
                "velocity_vector": convert_numpy_types(self._applied_velocity_vector),
                "velocity_magnitude": float(np.linalg.norm(self._applied_velocity_vector)),
                "velocity_point": convert_numpy_types(self._applied_velocity_point),
                "trajectory_description": "friction_slide_flat",
            }

        # Metadata
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(convert_numpy_types(self.metadata), f, indent=4)

        shutil.rmtree(self.scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    sim = FrictionSlideFlatSimulation(args)
    try:
        sim.run()
    except TimeoutException:
        logging.error("Simulation timed out")
        sys.exit(1) 