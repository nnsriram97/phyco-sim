import os
import sys; sys.path = ["kubric"] + sys.path
import uuid
import signal
import shutil
import tarfile
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from mathutils import Vector, Quaternion
import bpy
from skimage import io
import json
import pickle as pkl
import cv2
import pybullet as pb

from kubric_utils import (
    make_picklable,
    create_segmentation_color_map,
    apply_segmentation_colors,
    create_depth_video,
    save_video,
    get_object_metadata,
    get_world_object_bounds,
)


logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

SIM_ASSETS_DIR = os.environ.get("SIM_ASSETS_DIR", "./sim_assets")

WOOD_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "wood_textures")
CONCRETE_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "concrete_textures")
GROUND_TEXTURE_BASE_PATH = os.path.join(SIM_ASSETS_DIR, "ground_textures")

JENGA_BLOCK_URDF = "objs/jenga_block.urdf"
JENGA_BLOCK_OBJ = "objs/jenga_block.obj"
JENGA_BLOCK_FRICTION = 0.6
JENGA_BLOCK_RESTITUTION = 0.0
JENGA_BLOCK_MASS = 1.0
JENGA_BLOCK_SIZE = np.array([0.45, 0.15, 0.09])  # (length, width, height)
JENGA_LAYER_BLOCKS = 3
JENGA_RANDOM_HORIZONTAL_JITTER = 0.0 #0.004  # meters
JENGA_RANDOM_VERTICAL_JITTER = 0.0 #0.0015  # meters
JENGA_RANDOM_ROTATION_JITTER_DEG = 0.0 #2.5


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def time_limit(seconds):
    """Decorator to limit execution time of a function."""

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
parser.add_argument("--max_motion_blur", type=float, default=0.0)
parser.add_argument("--layers", type=str, default="image,segmentation,depth")
parser.add_argument("--efficient_rendering", action="store_true", default=False)
parser.add_argument("--velocity_threshold", type=float, default=0.05)
parser.add_argument("--angular_velocity_threshold", type=float, default=0.05)
parser.add_argument("--settle_frames", type=int, default=5)
parser.add_argument("--not_visible_stop_threshold", type=int, default=10)
parser.add_argument("--focal_length", type=float, default=80.0)
parser.add_argument("--sensor_width", type=float, default=32.0)
parser.add_argument("--camera_elevation_angle", type=float, default=None)
parser.add_argument("--camera_azimuth_angle", type=float, default=None)
parser.add_argument("--force_focal_length", action="store_true", default=False)
parser.add_argument(
    "--composition_style",
    type=str,
    default=None,
    help="Choose composition style for framing the Jenga tower: center, left_third, right_third, upper_left, upper_right, lower_center, lower_left, lower_right",
)
parser.add_argument(
    "--min_layers",
    type=int,
    default=15,
    help="Minimum number of Jenga layers to build",
)
parser.add_argument(
    "--max_layers",
    type=int,
    default=18,
    help="Maximum number of Jenga layers to build",
)
parser.add_argument(
    "--max_horizontal_jitter",
    type=float,
    default=JENGA_RANDOM_HORIZONTAL_JITTER,
    help="Maximum horizontal offset applied to blocks (meters)",
)
parser.add_argument(
    "--max_vertical_jitter",
    type=float,
    default=JENGA_RANDOM_VERTICAL_JITTER,
    help="Maximum vertical offset applied to blocks (meters)",
)
parser.add_argument(
    "--max_rotation_jitter_deg",
    type=float,
    default=JENGA_RANDOM_ROTATION_JITTER_DEG,
    help="Maximum random yaw applied to blocks (degrees)",
)
parser.add_argument(
    "--tower_stability_attempts",
    type=int,
    default=4,
    help="Maximum number of attempts to rebuild the tower if it fails stability checks",
)
parser.add_argument(
    "--tower_stability_settle_frames",
    type=int,
    default=180,
    help="Number of frames to simulate when evaluating tower stability",
)
parser.add_argument(
    "--tower_stability_displacement",
    type=float,
    default=0.05,
    help="Maximum displacement (meters) allowed during the stability settling pass",
)
parser.add_argument(
    "--tower_stability_velocity",
    type=float,
    default=0.9,
    help="Maximum linear velocity (m/s) allowed in the stability settling window",
)
parser.add_argument(
    "--tower_stability_angular_velocity",
    type=float,
    default=0.2,
    help="Maximum angular velocity (rad/s) allowed in the stability settling window",
)
parser.add_argument(
    "--tower_stability_tail_frames",
    type=int,
    default=12,
    help="Trailing frame count inspected when evaluating tower stability velocities",
)
parser.add_argument(
    "--missing_block_probability",
    type=float,
    default=0.1,
    help="Probability that a block is missing from non-edge layers",
)
parser.add_argument(
    "--force_magnitude",
    type=float,
    default=None,
    help="Deprecated: use to specify constant push speed magnitude (m/s)",
)
parser.add_argument(
    "--min_force",
    type=float,
    default=0.05,
    help="Minimum push speed (treated as m/s when constant velocity mode is active)",
)
parser.add_argument(
    "--max_force",
    type=float,
    default=0.5,
    help="Maximum push speed (treated as m/s when constant velocity mode is active)",
)
parser.add_argument(
    "--velocity_magnitude",
    type=float,
    default=None,
    help="Constant push speed to enforce on the selected block (m/s)",
)
parser.add_argument(
    "--min_velocity",
    type=float,
    default=None,
    help="Minimum push speed when sampling constant velocity (m/s)",
)
parser.add_argument(
    "--max_velocity",
    type=float,
    default=None,
    help="Maximum push speed when sampling constant velocity (m/s)",
)
parser.add_argument(
    "--force_layer_bias",
    type=float,
    default=0.5,
    help="Preference for middle layers when selecting a block (0=uniform, 1=strong middle bias)",
)
parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--save_gif", action="store_true", default=False)
parser.add_argument("--tar", action="store_true", default=False)
parser.add_argument(
    "--debug_gui",
    action="store_true",
    default=False,
    help="Enable PyBullet GUI for debugging",
)
parser.set_defaults(frame_end=15, frame_rate=10, resolution="768x432")
args = parser.parse_args()

if args.velocity_magnitude is None:
    args.velocity_magnitude = args.force_magnitude
if args.min_velocity is None:
    args.min_velocity = args.min_force
if args.max_velocity is None:
    args.max_velocity = args.max_force


# --------------------------------------------------------------------------------------
# PyBullet helper with GUI
# --------------------------------------------------------------------------------------


class PyBulletWithGUI(PyBullet):
    """PyBullet simulator with GUI enabled for debugging."""

    def __init__(self, scene, scratch_dir=None):
        import tempfile

        if scratch_dir is None:
            scratch_dir = tempfile.mkdtemp()

        from kubric.redirect_io import RedirectStream

        with RedirectStream(stream=sys.stderr):
            import pybullet as pb

        from kubric.simulator.pybullet import _BulletClient

        self._physics_client = _BulletClient(pb.GUI)
        self.scratch_dir = scratch_dir
        self._persistent_forces = []
        self._persistent_velocities = []

        self._physics_client.setPhysicsEngineParameter(
            restitutionVelocityThreshold=0.0,
            warmStartingFactor=0.0,
            useSplitImpulse=True,
            contactSlop=0.0,
            enableConeFriction=False,
            deterministicOverlappingPairs=True,
        )
        # self._physics_client.setPhysicsEngineParameter(
        #     fixedTimeStep=1.0/500.0,      # keep Bullet’s tuned default
        #     numSubSteps=2,                # micro-steps inside each frame
        #     numSolverIterations=150,      # more iterations => tighter contacts
        #     useSplitImpulse=1,            # separate penetration correction from velocities
        #     splitImpulsePenetrationThreshold=-0.01,
        #     contactERP=0.2,               # mild contact position correction
        #     frictionERP=0.2,              # helps static friction anchors
        #     erp=0.1,                      # for non-contact constraints
        #     contactSlop=0.001,            # small slop stabilizes resting contacts
        #     restitutionVelocityThreshold=0.2,  # kill tiny “bouncy” jitter
        #     deterministicOverlappingPairs=1    # more stable/repeatable pair sorting
        # )

        from kubric import core

        core.View.__init__(
            self,
            scene,
            scene_observers={
                "gravity": [lambda change: self._physics_client.setGravity(*change.new)],
            },
        )

        self._setup_debug_gui()

    def _setup_debug_gui(self):
        self._physics_client.setRealTimeSimulation(0)
        self._physics_client.configureDebugVisualizer(self._physics_client.COV_ENABLE_GUI, 1)
        self._physics_client.configureDebugVisualizer(self._physics_client.COV_ENABLE_SHADOWS, 1)

        print("\n" + "=" * 60)
        print("PyBullet GUI DEBUG MODE ENABLED")
        print("=" * 60)
        print("GUI Controls:")
        print("- Mouse: Rotate view")
        print("- Mouse wheel: Zoom")
        print("- Right panel: Physics parameters")
        print("- 'p' key: Pause/unpause simulation")
        print("- 'r' key: Reset simulation")
        print("- Close GUI window to continue with rendering")
        print("=" * 60)

    def pause_for_inspection(self, message="Paused for inspection"):
        print(f"\n{message}")
        print("Press Enter to continue...")
        input()

    def step_simulation_slowly(self, steps=100, delay=0.01):
        import time

        print(f"Stepping through {steps} simulation steps slowly...")
        for i in range(steps):
            self._physics_client.stepSimulation()
            time.sleep(delay)
            if i % 20 == 0:
                print(f"Step {i}/{steps}")
        print("Slow stepping complete.")


# --------------------------------------------------------------------------------------
# Main Jenga simulation class
# --------------------------------------------------------------------------------------


class JengaForceSimulation:
    """Generate randomized Jenga tower simulations with a constant push velocity."""

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

        self.jenga_blocks: List = []
        self.jenga_layer_count = 0
        self.jenga_base_height = 0.0
        self._tower_bounds = None
        self._target_block = None
        self._jenga_material_cache = {}
        self._applied_velocity_vector = None
        self._applied_velocity_point = None
        self._trajectory_description = "push_block"
        self._last_stability_metrics = {}

        self.hdri_source = self._load_asset_sources()

    # ------------------------------------------------------------------
    # Kubric setup helpers
    # ------------------------------------------------------------------

    def _configure_kubric(self):
        scene, rng, output_dir, scratch_dir = kb.setup(self.args)
        scene.gravity = (0, 0, -9.8)
        logging.info("Set gravity to normal Earth gravity: %s", scene.gravity)

        if self.args.debug_gui:
            simulator = PyBulletWithGUI(scene, scratch_dir)
            logging.info("PyBullet GUI enabled for debugging")
        else:
            simulator = PyBullet(scene, scratch_dir)

        simulator._physics_client.setGravity(0, 0, -9.8)
        simulator._physics_client.setRealTimeSimulation(0)

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
        return kb.AssetSource.from_manifest(self.args.hdri_assets)

    def _get_segmentation_id(self):
        self.global_segmentation_id += 1
        return self.global_segmentation_id

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _setup_background_and_plane(self):
        hdri_id = self.rng.choice(self.hdri_source.get_test_split(fraction=0.0)[0])
        hdri_tex = self.hdri_source.create(asset_id=hdri_id)

        hdri_rotation_z = -np.pi / 2 + self.rng.uniform(-np.pi / 4, np.pi / 4)
        hdri_rotation = (0.0, 0.0, hdri_rotation_z)

        self.renderer._set_ambient_light_hdri(hdri_tex.filename, hdri_rotation=hdri_rotation)
        self.renderer._set_background_hdri(hdri_tex.filename, hdri_rotation=hdri_rotation)
        self.renderer.background_transparency = False

        self.metadata["hdri_id"] = hdri_id
        self.metadata["hdri_rotation"] = {
            "x_radians": 0.0,
            "y_radians": 0.0,
            "z_radians": hdri_rotation_z,
            "z_degrees": float(np.degrees(hdri_rotation_z)),
        }

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
            segmentation_id=self._get_segmentation_id(),
        )
        self.scene += ground_plane
        self.dome = ground_plane
        ground_plane_idx = self.simulator.get_obj_idx(self.dome)
        self.simulator._physics_client.changeDynamics(ground_plane_idx, -1, collisionMargin=0.0)
        self._apply_ground_textures()

    def _apply_ground_textures(self):
        import os

        if not hasattr(self, "dome") or not self.dome:
            logging.warning("Ground plane object not available for texture application")
            return

        ground_plane_blender = self.dome.linked_objects[self.renderer]
        if not ground_plane_blender:
            logging.warning("Ground plane Blender object not found")
            return

        # texture_type = self.rng.choice(["wood", "concrete"])
        texture_type = "concrete"
        texture_base_path = WOOD_TEXTURE_BASE_PATH if texture_type == "wood" else CONCRETE_TEXTURE_BASE_PATH

        try:
            ground_types = [
                d
                for d in os.listdir(texture_base_path)
                if os.path.isdir(os.path.join(texture_base_path, d))
            ]
        except Exception as exc:
            logging.error("Error reading ground texture directories: %s", exc)
            return

        if not ground_types:
            logging.error("No ground texture directories found in %s", texture_base_path)
            return

        selected_ground = self.rng.choice(ground_types)
        texture_path = os.path.join(texture_base_path, selected_ground, "textures")
        self.metadata["ground_texture"] = selected_ground.split(".blend")[0]
        logging.info("Selected ground texture: %s", selected_ground)

        diffuse_path = normal_path = roughness_path = displacement_path = None
        try:
            for file in os.listdir(texture_path):
                file_lower = file.lower()
                full_path = os.path.join(texture_path, file)
                if "diff_4k" in file_lower and file_lower.endswith(".jpg"):
                    diffuse_path = full_path
                elif "nor_gl_4k" in file_lower and file_lower.endswith(".exr"):
                    normal_path = full_path
                elif "rough_4k" in file_lower and (file_lower.endswith(".jpg") or file_lower.endswith(".exr")):
                    roughness_path = full_path
                elif "disp_4k" in file_lower and (file_lower.endswith(".jpg") or file_lower.endswith(".png")):
                    displacement_path = full_path
        except Exception as exc:
            logging.error("Error finding texture files: %s", exc)
            return

        if len(ground_plane_blender.material_slots) == 0:
            mat = bpy.data.materials.new(name="Ground_Material")
            ground_plane_blender.data.materials.append(mat)
        else:
            mat = ground_plane_blender.material_slots[0].material
            if not mat:
                mat = bpy.data.materials.new(name="Ground_Material")
                ground_plane_blender.material_slots[0].material = mat

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        principled = nodes.new(type="ShaderNodeBsdfPrincipled")
        principled.location = (0, 0)
        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-800, 0)
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.location = (-600, 0)
        mapping.inputs["Scale"].default_value[0] = 5.0
        mapping.inputs["Scale"].default_value[1] = 5.0
        mapping.inputs["Scale"].default_value[2] = 1.0
        links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

        if diffuse_path and os.path.exists(diffuse_path):
            tex_diffuse = nodes.new(type="ShaderNodeTexImage")
            tex_diffuse.location = (-400, 200)
            tex_diffuse.image = bpy.data.images.load(diffuse_path)
            tex_diffuse.projection = "BOX"
            tex_diffuse.projection_blend = 0.15
            links.new(mapping.outputs["Vector"], tex_diffuse.inputs["Vector"])
            links.new(tex_diffuse.outputs["Color"], principled.inputs["Base Color"])

        if normal_path and os.path.exists(normal_path):
            tex_normal = nodes.new(type="ShaderNodeTexImage")
            tex_normal.location = (-400, 0)
            tex_normal.image = bpy.data.images.load(normal_path)
            tex_normal.image.colorspace_settings.name = "Non-Color"
            normal_map = nodes.new(type="ShaderNodeNormalMap")
            normal_map.location = (-200, 0)
            links.new(mapping.outputs["Vector"], tex_normal.inputs["Vector"])
            links.new(tex_normal.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])

        if roughness_path and os.path.exists(roughness_path):
            tex_rough = nodes.new(type="ShaderNodeTexImage")
            tex_rough.location = (-400, -200)
            tex_rough.image = bpy.data.images.load(roughness_path)
            tex_rough.image.colorspace_settings.name = "Non-Color"
            links.new(mapping.outputs["Vector"], tex_rough.inputs["Vector"])
            links.new(tex_rough.outputs["Color"], principled.inputs["Roughness"])

        if displacement_path and os.path.exists(displacement_path):
            tex_disp = nodes.new(type="ShaderNodeTexImage")
            tex_disp.location = (-400, -400)
            tex_disp.image = bpy.data.images.load(displacement_path)
            tex_disp.image.colorspace_settings.name = "Non-Color"
            disp_node = nodes.new(type="ShaderNodeDisplacement")
            disp_node.location = (-200, -400)
            disp_node.inputs["Scale"].default_value = 0.05
            links.new(mapping.outputs["Vector"], tex_disp.inputs["Vector"])
            links.new(tex_disp.outputs["Color"], disp_node.inputs["Height"])
            links.new(disp_node.outputs["Displacement"], output.inputs["Displacement"])
            mat.cycles.displacement_method = "BOTH"

        logging.info("Ground texture application completed successfully")

    def _select_jenga_wood_texture(self):
        texture_dirs = []
        try:
            texture_dirs = [
                d
                for d in os.listdir(WOOD_TEXTURE_BASE_PATH)
                if (Path(WOOD_TEXTURE_BASE_PATH) / d / "textures").is_dir()
            ]
        except Exception as exc:
            logging.warning("Unable to list wood textures at %s: %s", WOOD_TEXTURE_BASE_PATH, exc)

        selected = self.rng.choice(texture_dirs)
        texture_dir = Path(WOOD_TEXTURE_BASE_PATH) / selected / "textures"

        paths = {"diffuse": None, "normal": None, "roughness": None}
        try:
            for file in os.listdir(texture_dir):
                file_lower = file.lower()
                full_path = texture_dir / file
                if "diff_4k" in file_lower and file_lower.endswith(".jpg"):
                    paths["diffuse"] = full_path
                elif "nor_gl_4k" in file_lower and file_lower.endswith(".exr"):
                    paths["normal"] = full_path
                elif "rough_4k" in file_lower and (file_lower.endswith(".jpg") or file_lower.endswith(".exr")):
                    paths["roughness"] = full_path
        except Exception as exc:
            logging.warning("Failed to enumerate wood texture files in %s: %s", texture_dir, exc)

        images = {}
        for key, path in paths.items():
            if path is None:
                continue
            try:
                images[key] = bpy.data.images.load(str(path), check_existing=True)
            except Exception as exc:
                logging.warning("Could not load %s texture %s: %s", key, path, exc)

        texture_data = {
            "name": selected,
            "paths": paths,
            "images": images,
        }
        self.metadata.setdefault("jenga_texture_selection", selected.split(".blend")[0])
        return texture_data

    def _apply_jenga_wood_material(self, block, unique_block_idx):
        texture_data = self._select_jenga_wood_texture()
        material_key = f"jenga_wood_material_{unique_block_idx}"
        mat = self._jenga_material_cache.get(material_key)

        if mat is None:
            mat = bpy.data.materials.new(name="Jenga_Wood_Material")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            principled = nodes.new(type="ShaderNodeBsdfPrincipled")
            principled.location = (0, 0)
            principled.inputs['Specular'].default_value = 0.35
            principled.inputs['Roughness'].default_value = 0.45

            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (280, 0)
            links.new(principled.outputs['BSDF'], output.inputs['Surface'])

            tex_coord = nodes.new(type="ShaderNodeTexCoord")
            tex_coord.location = (-600, 0)
            mapping = nodes.new(type="ShaderNodeMapping")
            mapping.location = (-400, 0)
            mapping.inputs['Scale'].default_value[0] = 4.0
            mapping.inputs['Scale'].default_value[1] = 1.5
            mapping.inputs['Scale'].default_value[2] = 2.0
            links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])

            images = texture_data.get("images", {}) if texture_data else {}

            if "diffuse" in images:
                diffuse_node = nodes.new(type="ShaderNodeTexImage")
                diffuse_node.location = (-200, 220)
                diffuse_node.image = images["diffuse"]
                diffuse_node.interpolation = 'Smart'
                diffuse_node.extension = 'REPEAT'
                links.new(mapping.outputs['Vector'], diffuse_node.inputs['Vector'])
                links.new(diffuse_node.outputs['Color'], principled.inputs['Base Color'])
            else:
                principled.inputs['Base Color'].default_value = (0.59, 0.38, 0.22, 1.0)

            if "normal" in images:
                normal_tex = nodes.new(type="ShaderNodeTexImage")
                normal_tex.location = (-200, -60)
                normal_tex.image = images["normal"]
                normal_tex.image.colorspace_settings.name = 'Non-Color'
                normal_tex.extension = 'REPEAT'
                normal_tex.interpolation = 'Smart'
                normal_map = nodes.new(type="ShaderNodeNormalMap")
                normal_map.location = (0, -60)
                normal_map.inputs['Strength'].default_value = 0.8
                links.new(mapping.outputs['Vector'], normal_tex.inputs['Vector'])
                links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
                links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])

            if "roughness" in images:
                rough_tex = nodes.new(type="ShaderNodeTexImage")
                rough_tex.location = (-200, -260)
                rough_tex.image = images["roughness"]
                rough_tex.image.colorspace_settings.name = 'Non-Color'
                rough_tex.extension = 'REPEAT'
                rough_tex.interpolation = 'Smart'
                links.new(mapping.outputs['Vector'], rough_tex.inputs['Vector'])
                links.new(rough_tex.outputs['Color'], principled.inputs['Roughness'])

            self._jenga_material_cache[material_key] = mat

        blender_obj = block.linked_objects.get(self.renderer)
        if blender_obj is None:
            logging.debug("No Blender object linked for block %s to apply material", block.name)
            return

        if not blender_obj.material_slots:
            blender_obj.data.materials.append(mat)
        else:
            blender_obj.material_slots[0].material = mat
        blender_obj.active_material = mat

    # ------------------------------------------------------------------
    # Jenga tower construction
    # ------------------------------------------------------------------

    def _build_jenga_tower(self):
        min_layers = max(1, int(self.args.min_layers))
        max_layers = max(min_layers, int(self.args.max_layers))
        layer_count = int(self.rng.randint(min_layers, max_layers + 1))

        block_height = JENGA_BLOCK_SIZE[2] + 0.0005
        base_clearance = 0.0005
        base_center = np.array([0.0, 0.0, block_height / 2.0 + base_clearance])
        self.jenga_base_height = base_center[2] - block_height / 2.0
        self.jenga_layer_count = layer_count

        lateral_gap = 0.00075
        horizontal_jitter = float(
            getattr(self, "_current_horizontal_jitter", self.args.max_horizontal_jitter)
        )
        vertical_jitter = float(
            getattr(self, "_current_vertical_jitter", self.args.max_vertical_jitter)
        )
        missing_prob_override = getattr(self, "_current_missing_prob", None)
        missing_prob = float(
            np.clip(
                missing_prob_override
                if missing_prob_override is not None
                else self.args.missing_block_probability,
                0.0,
                0.95,
            )
        )

        blocks = []
        tower_layout = []
        layer_orientations = []
        layer_block_counts = []
        total_missing = 0

        for layer_idx in range(layer_count):
            axis = layer_idx % 2
            layer_orientations.append("x" if axis == 0 else "y")
            layer_center = base_center.copy()
            layer_center[2] += layer_idx * block_height

            present_flags = [True] * JENGA_LAYER_BLOCKS
            if layer_idx not in (0, layer_count - 1):
                for idx in range(JENGA_LAYER_BLOCKS):
                    if self.rng.uniform() < missing_prob:
                        present_flags[idx] = False
            if not any(present_flags):
                present_flags[1] = True

            created_this_layer = 0

            for block_idx in range(JENGA_LAYER_BLOCKS):
                if not present_flags[block_idx]:
                    total_missing += 1
                    tower_layout.append(
                        {
                            "name": None,
                            "layer": layer_idx,
                            "index": block_idx,
                            "orientation_axis": axis,
                            "present": False,
                        }
                    )
                    continue

                base_angle = 0.0 if axis == 0 else np.pi / 2.0
                orientation_quat = Quaternion((0.0, 0.0, 1.0), base_angle)

                spacing = JENGA_BLOCK_SIZE[1] + lateral_gap
                lateral_offset = (block_idx - (JENGA_LAYER_BLOCKS - 1) / 2.0) * spacing
                local_width_offset = Vector((0.0, lateral_offset, 0.0))

                jitter_local = Vector(
                    (
                        self.rng.uniform(-horizontal_jitter, horizontal_jitter),
                        self.rng.uniform(-horizontal_jitter, horizontal_jitter),
                        self.rng.uniform(-vertical_jitter, vertical_jitter),
                    )
                )

                world_offset = orientation_quat @ local_width_offset
                world_jitter = orientation_quat @ jitter_local
                world_position = layer_center + np.array(world_offset) + np.array(world_jitter)

                if layer_idx > 0 and world_position[2] < layer_center[2]:
                    world_position[2] = layer_center[2]

                bottom_z = world_position[2] - JENGA_BLOCK_SIZE[2] / 2.0
                if bottom_z < 0.0:
                    world_position[2] += abs(bottom_z) + 1e-4

                block = kb.FileBasedObject(
                    name=f"jenga_block_{layer_idx:02d}_{block_idx}",
                    simulation_filename=JENGA_BLOCK_URDF,
                    render_filename=JENGA_BLOCK_OBJ,
                    scale=1.0,
                    position=tuple(world_position),
                    mass=JENGA_BLOCK_MASS,
                    friction=JENGA_BLOCK_FRICTION,
                    restitution=JENGA_BLOCK_RESTITUTION,
                    segmentation_id=self._get_segmentation_id(),
                )

                self.scene += block
                block_idx = self.simulator.get_obj_idx(block)
                self.simulator._physics_client.changeDynamics(block_idx, -1, collisionMargin=0.0)

                unique_block_idx = f"{layer_idx}_{block_idx}"
                self._apply_jenga_wood_material(block, unique_block_idx)

                block.quaternion = orientation_quat[:]
                block.velocity = [0.0, 0.0, 0.0]
                block.angular_velocity = [0.0, 0.0, 0.0]
                block._object_type = "jenga_block"
                block.metadata["layer_index"] = layer_idx
                block.metadata["block_index"] = block_idx
                block.metadata["orientation_axis"] = axis
                block.metadata["orientation_angle_deg"] = float(np.degrees(base_angle))
                block.metadata["initial_position"] = [float(x) for x in world_position]
                block.metadata["initial_quaternion"] = list(block.quaternion)

                blocks.append(block)
                created_this_layer += 1
                tower_layout.append(
                    {
                        "name": block.name,
                        "layer": layer_idx,
                        "index": block_idx,
                        "orientation_axis": axis,
                        "orientation_angle_deg": float(np.degrees(base_angle)),
                        "present": True,
                    }
                )

            layer_block_counts.append(created_this_layer)

        self.jenga_blocks = blocks
        if blocks:
            self._tower_bounds = self.compute_scene_bounds(blocks)
        else:
            self._tower_bounds = (np.array([-0.5, -0.5, 0.0]), np.array([0.5, 0.5, 0.5]))

        self.metadata["simulation_type"] = "jenga_velocity"
        self.metadata["jenga_layer_count"] = layer_count
        self.metadata["jenga_blocks_total"] = len(blocks)
        self.metadata["jenga_layer_block_counts"] = layer_block_counts
        self.metadata["jenga_layer_orientations"] = layer_orientations
        self.metadata["jenga_missing_block_probability"] = missing_prob
        self.metadata["jenga_missing_blocks"] = total_missing
        self.metadata["jenga_tower_layout"] = tower_layout
        self.metadata["jenga_base_height"] = float(self.jenga_base_height)

        logging.info(
            "Constructed Jenga tower with %d layers and %d blocks (missing=%d)",
            layer_count,
            len(blocks),
            total_missing,
        )

    def _clear_existing_jenga_blocks(self):
        if not self.jenga_blocks:
            return

        for block in list(self.jenga_blocks):
            try:
                self.scene.remove(block)
            except ValueError:
                logging.debug("Jenga block %s already removed from scene", block.name)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to remove Jenga block %s: %s", block.name, exc)

        self.jenga_blocks = []
        self._tower_bounds = None
        self._jenga_material_cache.clear()

    def _simulate_tower_settle(self, settle_frames):
        settle_frames = max(1, int(settle_frames))
        animation, _ = self.simulator.run(frame_start=-10, frame_end=0)
        return animation

    def _tower_is_stable(self, animation):
        if not self.jenga_blocks:
            logging.warning("No Jenga blocks present when assessing tower stability")
            return False
        if not animation:
            logging.warning("No animation data returned during tower stability assessment")
            return False

        displacement_threshold = max(float(self.args.tower_stability_displacement), 0.0)
        velocity_threshold = max(float(self.args.tower_stability_velocity), 0.0)
        angular_threshold = max(float(self.args.tower_stability_angular_velocity), 0.0)
        tail_frames = max(1, int(self.args.tower_stability_tail_frames))

        max_displacement = 0.0
        max_velocity = 0.0
        max_angular_velocity = 0.0
        observed_blocks = 0
        unstable = False

        for block in self.jenga_blocks:
            data = animation.get(block)
            if not data:
                continue

            positions = np.asarray(data.get("position", []), dtype=float)
            velocities = np.asarray(data.get("velocity", []), dtype=float)
            angular_velocities = np.asarray(data.get("angular_velocity", []), dtype=float)

            if positions.size:
                displacement = float(np.linalg.norm(positions[-1] - positions[0]))
                max_displacement = max(max_displacement, displacement)
            else:
                displacement = 0.0

            tail_slice = slice(-tail_frames, None)

            if velocities.size:
                velocity_norms = np.linalg.norm(velocities, axis=1)
                tail_velocity_norms = velocity_norms[tail_slice] if velocity_norms.size else np.array([0.0])
                block_max_velocity = float(np.max(tail_velocity_norms)) if tail_velocity_norms.size else 0.0
                max_velocity = max(max_velocity, block_max_velocity)
            else:
                block_max_velocity = 0.0

            if angular_velocities.size:
                angular_norms = np.linalg.norm(angular_velocities, axis=1)
                tail_angular_norms = angular_norms[tail_slice] if angular_norms.size else np.array([0.0])
                block_max_angular = float(np.max(tail_angular_norms)) if tail_angular_norms.size else 0.0
                max_angular_velocity = max(max_angular_velocity, block_max_angular)
            else:
                block_max_angular = 0.0

            if (
                displacement > displacement_threshold
                or block_max_velocity > velocity_threshold
                or block_max_angular > angular_threshold
            ):
                unstable = True

            observed_blocks += 1

        if observed_blocks == 0:
            logging.warning("Unable to observe any Jenga blocks during stability assessment")
            return False

        self._last_stability_metrics = {
            "max_displacement": float(max_displacement),
            "max_velocity": float(max_velocity),
            "max_angular_velocity": float(max_angular_velocity),
        }
        self.metadata["tower_stability_metrics"] = dict(self._last_stability_metrics)

        return not unstable

    def _update_blocks_with_settled_state(self, animation):
        if not animation:
            return

        for block in self.jenga_blocks:
            data = animation.get(block)
            if not data:
                continue

            positions = data.get("position", [])
            quaternions = data.get("quaternion", [])
            if not positions or not quaternions:
                continue

            final_position = tuple(float(x) for x in positions[-1])
            final_quaternion = tuple(float(x) for x in quaternions[-1])

            block.position = final_position
            block.quaternion = final_quaternion
            block.velocity = [0.0, 0.0, 0.0]
            block.angular_velocity = [0.0, 0.0, 0.0]
            block.metadata["initial_position"] = list(final_position)
            block.metadata["initial_quaternion"] = list(final_quaternion)
            block.metadata["initial_velocity"] = [0.0, 0.0, 0.0]
            block.metadata["initial_angular_velocity"] = [0.0, 0.0, 0.0]

            try:
                block_idx = self.simulator.get_obj_idx(block)
                self.simulator._physics_client.resetBaseVelocity(block_idx, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            except Exception as exc:  # noqa: BLE001
                logging.debug("Failed to reset base velocity for %s: %s", block.name, exc)

    def _build_stable_jenga_tower(self):
        max_attempts = max(1, int(self.args.tower_stability_attempts))
        settle_frames = max(1, int(self.args.tower_stability_settle_frames))

        horizontal_jitter = float(self.args.max_horizontal_jitter)
        vertical_jitter = float(self.args.max_vertical_jitter)
        missing_prob = float(np.clip(self.args.missing_block_probability, 0.0, 0.95))

        attempt_log = []
        settle_animation = None
        tower_stable = False

        self.metadata["tower_stability_thresholds"] = {
            "displacement": float(max(self.args.tower_stability_displacement, 0.0)),
            "velocity": float(max(self.args.tower_stability_velocity, 0.0)),
            "angular_velocity": float(max(self.args.tower_stability_angular_velocity, 0.0)),
            "tail_frames": int(max(1, self.args.tower_stability_tail_frames)),
            "settle_frames": settle_frames,
        }

        try:
            for attempt in range(max_attempts):
                if attempt > 0:
                    self._clear_existing_jenga_blocks()

                self._current_horizontal_jitter = horizontal_jitter
                self._current_vertical_jitter = vertical_jitter
                self._current_missing_prob = missing_prob

                self._build_jenga_tower()
                settle_animation = self._simulate_tower_settle(settle_frames)
                is_stable = self._tower_is_stable(settle_animation)

                attempt_entry = {
                    "attempt": attempt + 1,
                    "horizontal_jitter": float(horizontal_jitter),
                    "vertical_jitter": float(vertical_jitter),
                    "missing_block_probability": float(missing_prob),
                    "stable": bool(is_stable),
                    "metrics": dict(self._last_stability_metrics) if self._last_stability_metrics else {},
                }
                attempt_log.append(attempt_entry)

                if is_stable:
                    logging.info("Jenga tower stabilized after %d attempt(s)", attempt + 1)
                    self.metadata["tower_build_attempts"] = attempt + 1
                    tower_stable = True
                    break

                logging.warning(
                    (
                        "Unstable tower on attempt %d (max displacement=%.4fm, velocity=%.4fm/s, "
                        "angular=%.4frad/s); reducing jitter and retrying"
                    ),
                    attempt + 1,
                    self._last_stability_metrics.get("max_displacement", 0.0),
                    self._last_stability_metrics.get("max_velocity", 0.0),
                    self._last_stability_metrics.get("max_angular_velocity", 0.0),
                )

                if attempt == max_attempts - 1:
                    break

                horizontal_jitter = max(horizontal_jitter * 0.6, 5e-5)
                vertical_jitter = max(vertical_jitter * 0.6, 5e-5)
                missing_prob = max(missing_prob * 0.5, 0.0)
        finally:
            for attr in ("_current_horizontal_jitter", "_current_vertical_jitter", "_current_missing_prob"):
                if hasattr(self, attr):
                    delattr(self, attr)

        self.metadata["tower_build_attempt_log"] = attempt_log

        if not tower_stable:
            self._clear_existing_jenga_blocks()
            raise RuntimeError("Unable to construct a stable Jenga tower within the allotted attempts")

        self.metadata["jenga_horizontal_jitter_used"] = float(horizontal_jitter)
        self.metadata["jenga_vertical_jitter_used"] = float(vertical_jitter)
        self.metadata["jenga_missing_block_probability"] = float(missing_prob)

        if settle_animation:
            self._update_blocks_with_settled_state(settle_animation)

    # ------------------------------------------------------------------
    # Velocity selection and application
    # ------------------------------------------------------------------

    def _select_force_block(self):
        if not self.jenga_blocks:
            raise RuntimeError("Jenga tower has not been built before force selection.")

        candidate_blocks = [
            block
            for block in self.jenga_blocks
            if 0 < block.metadata.get("layer_index", 0) < self.jenga_layer_count - 1
        ]
        if not candidate_blocks:
            candidate_blocks = list(self.jenga_blocks)

        layers = np.array([blk.metadata.get("layer_index", 0) for blk in candidate_blocks], dtype=float)
        indices = np.arange(len(candidate_blocks))

        bias = float(np.clip(self.args.force_layer_bias, 0.0, 1.0))
        weights = None
        if bias > 0.0 and self.jenga_layer_count > 1 and layers.size > 0:
            center = (self.jenga_layer_count - 1) / 2.0
            proximity = 1.0 - (np.abs(layers - center) / max(center, 1e-6))
            proximity = np.clip(proximity, 0.0, 1.0)
            weights = (1.0 - bias) + bias * proximity
            total = weights.sum()
            if total > 0:
                weights = weights / total
            else:
                weights = None

        if weights is not None:
            idx = self.rng.choice(indices, p=weights)
        else:
            idx = self.rng.choice(indices)

        block = candidate_blocks[int(idx)]
        logging.info(
            "Selected block %s (layer=%d, index=%d) to apply velocity",
            block.name,
            block.metadata.get("layer_index", -1),
            block.metadata.get("block_index", -1),
        )
        return block

    def _apply_velocity_to_block(self, block, velocity_vector=None, persistent=False, duration=None):
        if block is None:
            raise ValueError("Cannot apply velocity to a None block")

        if velocity_vector is None:
            explicit_velocity = self.args.velocity_magnitude
            if explicit_velocity is not None:
                magnitude = float(explicit_velocity)
            else:
                magnitude = float(self.rng.uniform(self.args.min_velocity, self.args.max_velocity))

            local_axis = Vector((1.0, 0.0, 0.0))
            quat = Quaternion(block.quaternion)
            direction = np.array(quat @ local_axis)
            direction[2] = 0.0
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm

            velocity_vector = direction * magnitude
        else:
            velocity_vector = np.array(velocity_vector, dtype=float)
            magnitude = float(np.linalg.norm(velocity_vector))

        block_idx = self.simulator.get_obj_idx(block)
        velocity_point_world = tuple(block.position)
        if persistent:
            self.simulator.add_persistent_velocity(block_idx, velocity_vector, duration=duration)
        else:
            _, angular_current = self.simulator._physics_client.getBaseVelocity(block_idx)
            self.simulator._physics_client.resetBaseVelocity(block_idx, tuple(velocity_vector), angular_current)

        axis_label = "x" if block.metadata.get("orientation_axis", 0) == 0 else "y"
        self._applied_velocity_vector = np.array(velocity_vector)
        self._applied_velocity_point = np.array(velocity_point_world)
        self._target_block = block
        self._trajectory_description = f"push_along_{axis_label}"

        if magnitude > 0:
            direction_norm = velocity_vector / magnitude
        else:
            direction_norm = np.array([0.0, 0.0, 0.0])

        self.metadata["velocity_magnitude"] = float(magnitude)
        self.metadata["velocity_direction"] = [float(x) for x in direction_norm]
        self.metadata["velocity_point"] = list(velocity_point_world)
        self.metadata["velocity_applied_block"] = block.name
        self.metadata["velocity_applied_layer"] = int(block.metadata.get("layer_index", -1))
        self.metadata["velocity_applied_block_index"] = int(block.metadata.get("block_index", -1))
        self.metadata["velocity_application_mode"] = "push_block"
        self.metadata["min_velocity"] = float(self.args.min_velocity)
        self.metadata["max_velocity"] = float(self.args.max_velocity)
        self.metadata["velocity_layer_bias"] = float(self.args.force_layer_bias)

        logging.info(
            "Setting velocity %.2fm/s on %s with vector %s",
            magnitude,
            block.name,
            velocity_vector,
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _collect_all_object_metadata(self):
        logging.info("Collecting metadata for scene objects")

        if "object_data" not in self.metadata:
            self.metadata["object_data"] = {}

        all_objects = []
        if hasattr(self, "dome") and self.dome:
            all_objects.append(("ground", self.dome))
        for block in self.jenga_blocks:
            all_objects.append(("jenga_block", block))

        for obj_type, obj in all_objects:
            try:
                obj_metadata = get_object_metadata(obj)
                obj_metadata["type"] = obj_type
                obj_metadata["object_name"] = obj.name

                for key, value in obj_metadata.items():
                    if key not in self.metadata["object_data"]:
                        self.metadata["object_data"][key] = []
                    self.metadata["object_data"][key].append(value)
            except Exception as exc:
                logging.error("Error collecting metadata for %s '%s': %s", obj_type, obj.name, exc)

    def compute_scene_bounds(self, objects):
        if not objects:
            return np.array([-1, -1, 0]), np.array([1, 1, 1])

        all_mins = []
        all_maxs = []
        for obj in objects:
            min_corner, max_corner = get_world_object_bounds(obj)
            all_mins.append(min_corner)
            all_maxs.append(max_corner)

        scene_min = np.min(all_mins, axis=0)
        scene_max = np.max(all_maxs, axis=0)
        return scene_min, scene_max

    # ------------------------------------------------------------------
    # Camera setup
    # ------------------------------------------------------------------

    def _setup_jenga_camera(self, objects):
        scene_min, scene_max = self.compute_scene_bounds(objects)
        scene_center = (scene_min + scene_max) / 2.0

        focal_length = self.args.focal_length if self.args.focal_length is not None else 50.0
        sensor_width = self.args.sensor_width if hasattr(self.args, "sensor_width") else 32.0
        if sensor_width < 10.0:
            logging.warning("Sensor width too small; overriding to 32mm for camera setup")
            sensor_width = 32.0

        self.scene.camera = kb.PerspectiveCamera(focal_length=focal_length, sensor_width=sensor_width)

        resolution = tuple(map(int, self.args.resolution.split("x")))
        aspect_ratio = resolution[0] / resolution[1]

        horizontal_fov = 2 * np.arctan(sensor_width / (2 * focal_length))
        vertical_fov = 2 * np.arctan((sensor_width / aspect_ratio) / (2 * focal_length))

        extent = scene_max - scene_min
        horizontal_extent = max(extent[0], extent[1]) * 1.2
        vertical_extent = extent[2] * 1.5 if extent[2] > 0 else JENGA_BLOCK_SIZE[2] * self.jenga_layer_count

        distance_horizontal = horizontal_extent / (2 * np.tan(horizontal_fov / 2))
        distance_vertical = vertical_extent / (2 * np.tan(vertical_fov / 2))
        base_camera_distance = max(distance_horizontal, distance_vertical, 1.5)

        # Get velocity direction for camera alignment
        velocity_direction_xy = self._get_velocity_direction_xy()
        
        # Define composition styles similar to cube deform simulation
        composition_styles = [
            'center',           # Jenga tower in center (original behavior)
            'left_third',       # Jenga tower on left third of frame  
            'right_third',      # Jenga tower on right third of frame
            'upper_left',       # Jenga tower in upper left area
            'upper_right',      # Jenga tower in upper right area
            'lower_center',     # Jenga tower in lower center (more ground visible)
            'lower_left',       # Jenga tower in lower left area
            'lower_right',      # Jenga tower in lower right area
        ]
        
        # Select composition style - avoid problematic styles for target blocks near edges
        if self.args.composition_style in composition_styles:
            composition_style = self.args.composition_style
        else:
            # If we have a target block, be more conservative with composition styles
            if hasattr(self, '_target_block') and self._target_block is not None:
                target_position = np.array(self._target_block.position)
                target_layer = self._target_block.metadata.get("layer_index", 0)
                
                # Avoid lower compositions for blocks in lower layers to prevent cutoff
                safe_styles = composition_styles.copy()
                if target_layer < self.jenga_layer_count * 0.4:  # Lower 40% of tower
                    safe_styles = [style for style in safe_styles if 'lower' not in style]
                    logging.info(f"Target block in lower layer {target_layer}, avoiding lower compositions")
                
                # Avoid edge compositions for blocks that might be cut off
                if target_layer < self.jenga_layer_count * 0.2:  # Bottom 20% of tower
                    safe_styles = [style for style in safe_styles if style in ['center', 'upper_left', 'upper_right']]
                    logging.info(f"Target block in bottom layer {target_layer}, using only safe compositions")
                
                composition_style = self.rng.choice(safe_styles if safe_styles else ['center'])
            else:
                composition_style = self.rng.choice(composition_styles)

        # Calculate camera position aligned with velocity direction (within 60 degrees)
        elevation_angle, azimuth_angle, camera_distance = self._calculate_camera_position_aligned_with_velocity(
            velocity_direction_xy, base_camera_distance, scene_center
        )

        # Calculate base camera position using spherical coordinates
        x = camera_distance * np.cos(elevation_angle) * np.cos(azimuth_angle)
        y = camera_distance * np.cos(elevation_angle) * np.sin(azimuth_angle)
        z = camera_distance * np.sin(elevation_angle)
        base_camera_position = scene_center + np.array([x, y, z])

        # Calculate look-at point based on composition style
        look_at_point = self._calculate_jenga_look_at_point(
            scene_center, composition_style, horizontal_fov, vertical_fov, camera_distance
        )
        
        # If we have a target block, adjust look-at point to ensure it stays visible
        if hasattr(self, '_target_block') and self._target_block is not None:
            target_position = np.array(self._target_block.position)
            # Bias look-at point slightly towards target block to improve visibility
            target_weight = 0.15  # 15% bias towards target block
            look_at_point = look_at_point * (1 - target_weight) + target_position * target_weight
            logging.info(f"Adjusted look-at point towards target block: {target_position}")
        

        # Adjust camera position to ensure tower visibility
        camera_position = self._adjust_camera_for_tower_visibility(
            base_camera_position, look_at_point, scene_center, 
            horizontal_fov, vertical_fov, camera_distance
        )

        self.scene.camera.position = camera_position
        self.scene.camera.look_at(look_at_point)

        view_direction = look_at_point - camera_position
        view_direction = view_direction / np.linalg.norm(view_direction)

        self._composition_style = composition_style
        self._scene_center = look_at_point
        self._camera_position = camera_position
        self._camera_look_direction = view_direction
        self._camera_horizontal_fov = horizontal_fov
        self._camera_vertical_fov = vertical_fov

        self.metadata["camera_position"] = list(camera_position.tolist())
        self.metadata["camera_look_at"] = list(look_at_point.tolist())
        self.metadata["camera_distance"] = float(np.linalg.norm(camera_position - look_at_point))
        self.metadata["camera_composition_style"] = composition_style
        self.metadata["velocity_direction_xy"] = list(velocity_direction_xy)
        
        # Store target block visibility information
        if hasattr(self, '_target_block') and self._target_block is not None:
            target_position = np.array(self._target_block.position)
            final_view_direction = look_at_point - camera_position
            final_view_direction = final_view_direction / np.linalg.norm(final_view_direction)
            
            target_visible = self._is_point_in_frustum(target_position, camera_position, final_view_direction, 
                                                     horizontal_fov, vertical_fov, margin=0.1)
            target_screen_coords = self._get_screen_coordinates(target_position, camera_position, final_view_direction, 
                                                              horizontal_fov, vertical_fov)
            
            self.metadata["target_block_visibility"] = {
                "position": list(target_position),
                "layer_index": int(self._target_block.metadata.get("layer_index", -1)),
                "visible_in_frustum": bool(target_visible),
                "screen_coordinates": list(target_screen_coords) if target_screen_coords is not None else None,
                "distance_from_camera": float(np.linalg.norm(target_position - camera_position))
            }
        
        # Calculate angle between camera view direction and velocity direction on XY plane
        camera_view_xy = np.array([view_direction[0], view_direction[1]])
        camera_view_xy_norm = camera_view_xy / np.linalg.norm(camera_view_xy) if np.linalg.norm(camera_view_xy) > 1e-6 else np.array([1.0, 0.0])
        
        # Ensure both vectors are 2D numpy arrays
        velocity_direction_xy_2d = np.array(velocity_direction_xy[:2])  # Take first 2 components
        camera_view_xy_norm = np.array(camera_view_xy_norm[:2])  # Ensure 2D
        
        # Calculate dot product and angle
        dot_product = np.dot(camera_view_xy_norm, velocity_direction_xy_2d)
        angle_between = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        self.metadata["camera_velocity_angle_degrees"] = float(angle_between)

        logging.info(
            "🎥 JENGA CAMERA SETUP WITH VELOCITY ALIGNMENT")
        logging.info(f"📐 Composition style: {composition_style}")
        logging.info(f"🎯 Velocity direction (XY): {velocity_direction_xy}")
        logging.info(f"📷 Camera position: {camera_position}")
        logging.info(f"👀 Look-at point: {look_at_point}")
        logging.info(f"📏 Camera distance: {np.linalg.norm(camera_position - look_at_point):.2f}m")
        logging.info(f"📐 Elevation: {np.degrees(elevation_angle):.1f}°")
        logging.info(f"🧭 Azimuth: {np.degrees(azimuth_angle):.1f}°")
        logging.info(f"🎯 Angle between camera view and velocity (XY): {angle_between:.1f}° (target: <60°)")
        
        if angle_between > 60.0:
            logging.warning(f"⚠️  Camera-velocity angle {angle_between:.1f}° exceeds 60° target!")
        else:
            logging.info(f"✅ Camera-velocity alignment within target: {angle_between:.1f}° < 60°")

    def _get_velocity_direction_xy(self):
        """Get the velocity direction in the XY plane for the target block."""
        if not hasattr(self, '_target_block') or self._target_block is None:
            # If no target block selected yet, return a default direction
            return np.array([1.0, 0.0])
        
        # Calculate velocity direction the same way as in _apply_velocity_to_block
        block = self._target_block
        local_axis = Vector((1.0, 0.0, 0.0))
        quat = Quaternion(block.quaternion)
        direction = np.array(quat @ local_axis)
        direction[2] = 0.0  # Project to XY plane
        
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            # Return 2D array
            return np.array([1.0, 0.0])
        else:
            direction = direction / norm
            # Return only XY components as 2D array
            return np.array([direction[0], direction[1]])

    def _calculate_camera_position_aligned_with_velocity(self, velocity_direction_xy, base_camera_distance, scene_center):
        """Calculate camera position that is within 60 degrees of the velocity direction on XY plane."""
        
        # Calculate the velocity azimuth angle (direction the block will move)
        velocity_azimuth = np.arctan2(velocity_direction_xy[1], velocity_direction_xy[0])
        
        # Define the range of valid camera azimuth angles (within 60 degrees of velocity direction)
        max_angle_offset = np.radians(60)  # 60 degrees in radians
        min_camera_azimuth = velocity_azimuth - max_angle_offset
        max_camera_azimuth = velocity_azimuth + max_angle_offset
        
        # Apply some variation within the valid range
        azimuth_variation = self.rng.uniform(-max_angle_offset * 0.8, max_angle_offset * 0.8)
        camera_azimuth = velocity_azimuth + azimuth_variation
        
        # Ensure the angle is within [-pi, pi]
        camera_azimuth = np.arctan2(np.sin(camera_azimuth), np.cos(camera_azimuth))
        
        # Choose elevation angle based on user preference or random within reasonable range
        if self.args.camera_elevation_angle is not None:
            elevation_angle = np.radians(self.args.camera_elevation_angle)
        else:
            # Use elevation angles that provide good visibility of the tower and block movement
            elevation_angle = np.radians(self.rng.uniform(15, 45))
        
        # Apply some distance variation for visual diversity
        distance_variation = self.rng.uniform(0.9, 1.3)
        camera_distance = base_camera_distance * distance_variation
        
        logging.info(f"🎯 Velocity azimuth: {np.degrees(velocity_azimuth):.1f}°")
        logging.info(f"📷 Camera azimuth: {np.degrees(camera_azimuth):.1f}° (offset: {np.degrees(azimuth_variation):.1f}°)")
        logging.info(f"📐 Valid azimuth range: [{np.degrees(min_camera_azimuth):.1f}°, {np.degrees(max_camera_azimuth):.1f}°]")
        
        return elevation_angle, camera_azimuth, camera_distance

    def _calculate_jenga_look_at_point(self, scene_center, composition_style, horizontal_fov, vertical_fov, camera_distance):
        """Calculate look-at point based on composition style to vary tower position in frame."""
        
        # Base look-at point starts with scene center
        look_at_point = scene_center.copy()
        
        # Adjust Z to focus on middle of the tower
        look_at_point[2] = max(scene_center[2], self.jenga_base_height + JENGA_BLOCK_SIZE[2] * self.jenga_layer_count / 2)
        
        # Calculate offset distances based on FOV and camera distance
        # These offsets will move the look-at point, which shifts where the tower appears in frame
        horizontal_offset_max = camera_distance * np.tan(horizontal_fov / 4)  # 1/4 of FOV width
        vertical_offset_max = camera_distance * np.tan(vertical_fov / 4)      # 1/4 of FOV height
        
        if composition_style == 'center':
            # No offset - tower stays in center (original behavior)
            pass
            
        elif composition_style == 'left_third':
            # Move look-at point to the right, so tower appears on left third
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            
        elif composition_style == 'right_third':
            # Move look-at point to the left, so tower appears on right third  
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            
        elif composition_style == 'upper_left':
            # Move look-at point right and down, so tower appears upper left
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            look_at_point[2] -= vertical_offset_max * self.rng.uniform(0.5, 1.0)
            
        elif composition_style == 'upper_right':
            # Move look-at point left and down, so tower appears upper right
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            look_at_point[2] -= vertical_offset_max * self.rng.uniform(0.5, 1.0)
            
        elif composition_style == 'lower_center':
            # Move look-at point up, so tower appears in lower center (more ground visible)
            look_at_point[2] += vertical_offset_max * self.rng.uniform(0.2, 0.6)
            
        elif composition_style == 'lower_left':
            # Move look-at point right and up, so tower appears in lower left
            look_at_point[0] -= horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            look_at_point[2] += vertical_offset_max * 0.7# self.rng.uniform(0.2, 0.6)
            
        elif composition_style == 'lower_right':
            # Move look-at point left and up, so tower appears in lower right
            look_at_point[0] += horizontal_offset_max * self.rng.uniform(0.5, 1.3)
            look_at_point[2] += vertical_offset_max * 0.7# self.rng.uniform(0.2, 0.6)
        
        return look_at_point

    def _adjust_camera_for_tower_visibility(self, base_camera_position, look_at_point, scene_center, 
                                          horizontal_fov, vertical_fov, camera_distance):
        """Adjust camera position to ensure the Jenga tower and target block are visible."""
        
        camera_position = base_camera_position.copy()
        
        # Basic check: ensure camera is not too close to the look-at point
        distance_to_lookat = np.linalg.norm(camera_position - look_at_point)
        min_distance = 1.0  # Minimum 1 meter distance
        
        if distance_to_lookat < min_distance:
            # Move camera back along the direction from look-at point to camera
            direction = camera_position - look_at_point
            direction_norm = direction / np.linalg.norm(direction)
            camera_position = look_at_point + direction_norm * min_distance
        
        # If we have a target block, ensure it's visible within the camera frustum
        if hasattr(self, '_target_block') and self._target_block is not None:
            target_position = np.array(self._target_block.position)
            
            # Calculate velocity endpoint for visibility checking
            if hasattr(self, '_applied_velocity_vector') and self._applied_velocity_vector is not None:
                velocity_vector = self._applied_velocity_vector
            else:
                # Use predicted velocity direction based on block orientation
                local_axis = Vector((1.0, 0.0, 0.0))
                quat = Quaternion(self._target_block.quaternion)
                direction = np.array(quat @ local_axis)
                direction[2] = 0.0  # Project to XY plane
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    velocity_vector = np.array([0.1, 0.0, 0.0])
                else:
                    velocity_vector = direction / norm * 0.2  # Default magnitude
            
            velocity_scale = 0.3  # Show velocity vector extending 30cm from block
            velocity_endpoint = target_position + velocity_vector * velocity_scale
            
            # Iteratively adjust camera to ensure target block and velocity vector are visible
            max_iterations = 5
            for iteration in range(max_iterations):
                view_direction = (look_at_point - camera_position)
                view_direction = view_direction / np.linalg.norm(view_direction)
                
                # Check visibility of target block and velocity endpoint
                target_visible = self._is_point_in_frustum(target_position, camera_position, view_direction, 
                                                         horizontal_fov, vertical_fov, margin=0.15)
                velocity_visible = self._is_point_in_frustum(velocity_endpoint, camera_position, view_direction,
                                                           horizontal_fov, vertical_fov, margin=0.15)
                
                if target_visible and velocity_visible:
                    logging.info(f"✅ Target block and velocity vector are visible (iteration {iteration + 1})")
                    break
                
                # If not visible, adjust camera position
                if not target_visible:
                    logging.info(f"🔄 Target block not visible, adjusting camera (iteration {iteration + 1})")
                    camera_position = self._adjust_camera_for_point_visibility(
                        camera_position, look_at_point, target_position, horizontal_fov, vertical_fov
                    )
                
                if not velocity_visible:
                    logging.info(f"🔄 Velocity vector not visible, adjusting camera (iteration {iteration + 1})")
                    camera_position = self._adjust_camera_for_point_visibility(
                        camera_position, look_at_point, velocity_endpoint, horizontal_fov, vertical_fov
                    )
                
                # Ensure minimum distance is maintained after adjustments
                distance_to_lookat = np.linalg.norm(camera_position - look_at_point)
                if distance_to_lookat < min_distance:
                    direction = camera_position - look_at_point
                    direction_norm = direction / np.linalg.norm(direction)
                    camera_position = look_at_point + direction_norm * min_distance
            
            # Final visibility check and warning if still not visible
            final_view_direction = (look_at_point - camera_position)
            final_view_direction = final_view_direction / np.linalg.norm(final_view_direction)
            
            final_target_visible = self._is_point_in_frustum(target_position, camera_position, final_view_direction, 
                                                           horizontal_fov, vertical_fov, margin=0.15)
            final_velocity_visible = self._is_point_in_frustum(velocity_endpoint, camera_position, final_view_direction,
                                                             horizontal_fov, vertical_fov, margin=0.15)
            
            if not final_target_visible:
                logging.warning("⚠️  Target block may still be partially cut off from frame")
            if not final_velocity_visible:
                logging.warning("⚠️  Velocity vector may still be partially cut off from frame")
                
        return camera_position

    def _is_point_in_frustum(self, point, camera_position, view_direction, horizontal_fov, vertical_fov, margin=0.0):
        """Check if a point is within the camera's view frustum with optional margin."""
        point = np.array(point)
        camera_position = np.array(camera_position)
        
        # Vector from camera to point
        to_point = point - camera_position
        distance = np.linalg.norm(to_point)
        
        if distance < 0.01:  # Too close to camera
            return True
            
        to_point_normalized = to_point / distance
        
        # Check if point is in front of camera
        forward_dot = np.dot(to_point_normalized, view_direction)
        if forward_dot < 0.1:  # Behind camera or too close to camera plane
            return False
        
        # Create camera coordinate system
        world_up = np.array([0, 0, 1])
        right = np.cross(view_direction, world_up)
        if np.linalg.norm(right) < 0.01:
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)
        
        # Project point onto camera's right and up axes
        proj_right = np.dot(to_point, right)
        proj_up = np.dot(to_point, up)
        proj_forward = np.dot(to_point, view_direction)
        
        if proj_forward <= 0:
            return False
        
        # Calculate angles from camera center to point
        angle_right = np.arctan2(abs(proj_right), proj_forward)
        angle_up = np.arctan2(abs(proj_up), proj_forward)
        
        # Apply margin to FOV
        effective_horizontal_fov = horizontal_fov * (1 + margin)
        effective_vertical_fov = vertical_fov * (1 + margin)
        
        # Check if within frustum bounds
        within_horizontal = angle_right <= effective_horizontal_fov / 2
        within_vertical = angle_up <= effective_vertical_fov / 2
        
        return within_horizontal and within_vertical

    def _adjust_camera_for_point_visibility(self, camera_position, look_at_point, target_point, horizontal_fov, vertical_fov):
        """Adjust camera position to ensure a specific point is visible."""
        camera_position = np.array(camera_position)
        look_at_point = np.array(look_at_point)
        target_point = np.array(target_point)
        
        # Calculate current view direction
        view_direction = look_at_point - camera_position
        view_direction = view_direction / np.linalg.norm(view_direction)
        
        # Calculate vector from camera to target point
        to_target = target_point - camera_position
        distance_to_target = np.linalg.norm(to_target)
        
        if distance_to_target < 0.01:
            return camera_position
            
        to_target_normalized = to_target / distance_to_target
        
        # Create camera coordinate system
        world_up = np.array([0, 0, 1])
        right = np.cross(view_direction, world_up)
        if np.linalg.norm(right) < 0.01:
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)
        
        # Project target point onto camera axes
        proj_right = np.dot(to_target, right)
        proj_up = np.dot(to_target, up)
        proj_forward = np.dot(to_target, view_direction)
        
        if proj_forward <= 0:
            # Target is behind camera, move camera back
            camera_position = camera_position - view_direction * 0.5
            return camera_position
        
        # Calculate angles
        angle_right = np.arctan2(abs(proj_right), proj_forward)
        angle_up = np.arctan2(abs(proj_up), proj_forward)
        
        # Check if adjustments are needed
        max_horizontal_angle = horizontal_fov / 2 * 0.85  # Keep within 85% of FOV
        max_vertical_angle = vertical_fov / 2 * 0.85
        
        adjustment_needed = False
        adjustment_vector = np.array([0.0, 0.0, 0.0])
        
        if angle_right > max_horizontal_angle:
            # Move camera in the direction that reduces the horizontal angle
            sign = 1 if proj_right > 0 else -1
            adjustment_vector += right * sign * 0.2
            adjustment_needed = True
            
        if angle_up > max_vertical_angle:
            # Move camera in the direction that reduces the vertical angle
            sign = 1 if proj_up > 0 else -1
            adjustment_vector += up * sign * 0.2
            adjustment_needed = True
        
        if adjustment_needed:
            camera_position = camera_position + adjustment_vector
        else:
            # If point is too close to edge, move camera back slightly
            camera_position = camera_position - view_direction * 0.1
        
        return camera_position

    def _get_screen_coordinates(self, point, camera_position, view_direction, horizontal_fov, vertical_fov):
        """Get normalized screen coordinates (-1 to 1) for a point."""
        point = np.array(point)
        camera_position = np.array(camera_position)
        
        # Vector from camera to point
        to_point = point - camera_position
        distance = np.linalg.norm(to_point)
        
        if distance < 0.01:
            return None
            
        to_point_normalized = to_point / distance
        
        # Check if point is in front of camera
        forward_dot = np.dot(to_point_normalized, view_direction)
        if forward_dot < 0.1:
            return None
        
        # Create camera coordinate system
        world_up = np.array([0, 0, 1])
        right = np.cross(view_direction, world_up)
        if np.linalg.norm(right) < 0.01:
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, view_direction)
        up = up / np.linalg.norm(up)
        
        # Project point onto camera axes
        proj_right = np.dot(to_point, right)
        proj_up = np.dot(to_point, up)
        proj_forward = np.dot(to_point, view_direction)
        
        if proj_forward <= 0:
            return None
        
        # Calculate normalized screen coordinates
        # Screen coordinates range from -1 to 1
        screen_x = np.arctan2(proj_right, proj_forward) / (horizontal_fov / 2)
        screen_y = np.arctan2(proj_up, proj_forward) / (vertical_fov / 2)
        
        return np.array([screen_x, screen_y])

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _is_object_in_camera_frustum(self, obj_position, margin=0.5):
        if not hasattr(self, "scene") or not hasattr(self.scene, "camera"):
            return True

        camera = self.scene.camera
        cam_pos = np.array(camera.position)
        obj_pos = np.array(obj_position)
        view_vec = obj_pos - cam_pos
        view_distance = np.linalg.norm(view_vec)
        if view_distance < 0.01:
            return True

        view_direction = view_vec / view_distance
        camera_forward = getattr(self, "_camera_look_direction", view_direction)

        forward_dot = np.dot(view_direction, camera_forward)
        if forward_dot < 0.1:
            return False

        horizontal_fov = getattr(self, "_camera_horizontal_fov", np.radians(60)) * (1 + margin)
        vertical_fov = getattr(self, "_camera_vertical_fov", np.radians(45)) * (1 + margin)

        world_up = np.array([0, 0, 1])
        right = np.cross(camera_forward, world_up)
        if np.linalg.norm(right) < 0.01:
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, camera_forward)
        up = up / np.linalg.norm(up)

        obj_relative = obj_pos - cam_pos
        proj_right = np.dot(obj_relative, right)
        proj_up = np.dot(obj_relative, up)
        proj_forward = np.dot(obj_relative, camera_forward)
        if proj_forward <= 0:
            return False

        angle_right = np.arctan2(abs(proj_right), proj_forward)
        angle_up = np.arctan2(abs(proj_up), proj_forward)
        return angle_right <= horizontal_fov / 2 and angle_up <= vertical_fov / 2

    def _detect_motion_settled(self, animation_data):
        num_frames = len(next(iter(animation_data.values()))["velocity"])
        settle_counter = 0
        visible_objects = set()

        for obj in animation_data:
            for f in range(num_frames):
                if self._is_object_in_camera_frustum(animation_data[obj]["position"][f]):
                    visible_objects.add(obj)
                    break

        if not visible_objects:
            return None

        for f in range(num_frames):
            moving = False
            for obj in visible_objects:
                velocity = animation_data[obj]["velocity"][f]
                angular_velocity = animation_data[obj]["angular_velocity"][f]
                if (
                    np.linalg.norm(velocity) > self.args.velocity_threshold
                    or np.linalg.norm(angular_velocity) > self.args.angular_velocity_threshold
                ):
                    moving = True
                    break

            if not moving:
                settle_counter += 1
                if settle_counter >= self.args.settle_frames:
                    settle_frame = f - self.args.settle_frames + 1
                    logging.info("Motion settled at frame %d", settle_frame)
                    return settle_frame
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

    # ------------------------------------------------------------------
    # Force visualization helpers
    # ------------------------------------------------------------------

    def debug_pause_at_key_points(self):
        if hasattr(self.simulator, "pause_for_inspection"):
            self.simulator.pause_for_inspection("Scene setup complete. Inspect Jenga tower and camera angle.")

    def add_velocity_application(self, object_name, velocity_point_world, velocity_vector_world, frame=0):
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
    # Main orchestration
    # ------------------------------------------------------------------

    def run(self):
        self._setup_background_and_plane()

        if self.args.debug_gui:
            self.debug_pause_at_key_points()

        self._build_stable_jenga_tower()

        if self.args.debug_gui:
            self.debug_pause_at_key_points()

        self._collect_all_object_metadata()
        
        # Select target block before camera setup so camera can align with velocity direction
        target_block = self._select_force_block()
        self._target_block = target_block  # Store for camera setup
        
        self._setup_jenga_camera(self.jenga_blocks)

        if self.args.debug_gui:
            self.debug_pause_at_key_points()
        
        logging.info("Running physics simulation for %d frames", self.args.frame_end + 1)
        simulation_frame_start = 0
        simulation_frame_end = self.args.frame_end + 1
        velocity_duration = int(0.6 * self.args.frame_rate)
        self._apply_velocity_to_block(target_block, persistent=True, duration=velocity_duration)
        try:
            anim_data, _ = self.simulator.run(
                frame_start=simulation_frame_start,
                frame_end=simulation_frame_end,
            )
        finally:
            self.simulator.clear_persistent_velocities()

        if target_block in anim_data:
            positions = anim_data[target_block]["position"]
            if positions:
                velocity_point_world = np.array(self._applied_velocity_point)
                self.add_velocity_application(
                    object_name=target_block.name,
                    velocity_point_world=velocity_point_world,
                    velocity_vector_world=self._applied_velocity_vector,
                    frame=0,
                )
        else:
            logging.warning("Target block not found in animation data")

        picklable = make_picklable(anim_data)
        with open(os.path.join(self.output_dir, "animation_data.pkl"), "wb") as f:
            pkl.dump(picklable, f)

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
                
                # Create velocity-annotated version for first frame if debug_gui is enabled
                if i == 0 and self.applied_velocities:
                    try:
                        annotated_image, velocity_viz_metadata = self.create_velocity_annotated_image(
                            image_path, 
                            output_path=os.path.join(self.output_dir, f"velocity_annotated_{i:05d}.jpg")
                        )
                        
                        if velocity_viz_metadata:
                            # Store velocity visualization metadata
                            self.metadata["applied_velocities_image"] = velocity_viz_metadata
                            logging.info("🎯 Created velocity-annotated first frame for debug visualization")
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

        if self.applied_velocities:
            self.metadata["applied_velocities_simulator"] = convert_numpy_types(self.applied_velocities)

        self.metadata["velocity_profile"] = {
            "composition_style": getattr(self, "_composition_style", "unknown"),
            "velocity_vector": convert_numpy_types(self._applied_velocity_vector),
            "velocity_magnitude": float(np.linalg.norm(self._applied_velocity_vector)),
            "trajectory_description": self._trajectory_description,
            "velocity_point": convert_numpy_types(self._applied_velocity_point),
        }

        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(convert_numpy_types(self.metadata), f, indent=4)

        shutil.rmtree(self.scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    sim = JengaForceSimulation(args)
    try:
        sim.run()
    except TimeoutException:
        logging.error("Simulation timed out")
        sys.exit(1)
