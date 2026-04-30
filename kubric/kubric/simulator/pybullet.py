# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=function-redefined

import functools
import inspect
import logging
import pathlib
import sys
import tempfile
from typing import Dict, List, Optional, Tuple, Union

from kubric import core
from kubric.redirect_io import RedirectStream
import tensorflow as tf
import numpy as np
import signal
import time
        
# --- hides the "pybullet build time: May 26 2021 18:52:36" message on import
with RedirectStream(stream=sys.stderr):
  import pybullet as pb

logger = logging.getLogger(__name__)


class _BulletClient:

  def __init__(self, connection_mode: int):
    self._client = pb.connect(connection_mode)

  @property
  def client(self):
    return self._client

  def __del__(self):
    if self.client >= 0:
      try:
        self.disconnect()
      except pb.error:
        pass

  def __getattr__(self, name):
    attribute = getattr(pb, name)
    if inspect.isbuiltin(attribute):
      attribute = functools.partial(attribute, physicsClientId=self.client)
    return attribute


class PyBullet(core.View):
  """Adds physics simulation on top of kb.Scene using PyBullet."""

  def __init__(self, scene: core.Scene, scratch_dir=tempfile.mkdtemp()):
    self.scratch_dir = scratch_dir
    self._physics_client = _BulletClient(pb.DIRECT)  # pb.GUI
    self._persistent_forces: List[dict] = []
    self._persistent_velocities: List[dict] = []

    # --- Set some parameters to fix the sticky-walls problem; see
    # https://github.com/bulletphysics/bullet3/issues/3094
    self._physics_client.setPhysicsEngineParameter(restitutionVelocityThreshold=0.,
                                 warmStartingFactor=0.,
                                 useSplitImpulse=True,
                                 contactSlop=0.,
                                 enableConeFriction=False,
                                 deterministicOverlappingPairs=True)
    # TODO: setTimeStep if scene.step_rate != 240 Hz
    super().__init__(
        scene,
        scene_observers={
            "gravity": [
                lambda change: self._physics_client.setGravity(*change.new)
            ],
        })

  @property
  def physics_client(self):
    return self._physics_client.client

  @functools.singledispatchmethod
  def add_asset(self, asset: core.Asset) -> Optional[int]:
    raise NotImplementedError(f"Cannot add {asset!r}")

  def remove_asset(self, asset: core.Asset) -> None:
    if self in asset.linked_objects:
      self._physics_client.removeBody(asset.linked_objects[self])
    # TODO(klausg): unobserve

  @add_asset.register(core.Camera)
  def _add_object(self, obj: core.Camera) -> None:
    logger.debug("Ignored camera %s", obj)

  @add_asset.register(core.Material)
  def _add_object(self, obj: core.Material) -> None:
    logger.debug("Ignored material %s", obj)

  @add_asset.register(core.Light)
  def _add_object(self, obj: core.Light) -> None:
    logger.debug("Ignored light %s", obj)

  @add_asset.register(core.Cube)
  def _add_object(self, obj: core.Cube) -> Optional[int]:
    collision_idx = self._physics_client.createCollisionShape(
        pb.GEOM_BOX, halfExtents=obj.scale)
    visual_idx = -1
    mass = 0 if obj.static else obj.mass
    # useMaximalCoordinates and contactProcessingThreshold are required to
    # fix the sticky walls issue;
    # see https://github.com/bulletphysics/bullet3/issues/3094
    box_idx = self._physics_client.createMultiBody(
        mass,
        collision_idx,
        visual_idx,
        obj.position,
        wxyz2xyzw(obj.quaternion),
        useMaximalCoordinates=True)
    self._physics_client.changeDynamics(
        box_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, box_idx, self._physics_client)

    return box_idx

  @add_asset.register(core.Sphere)
  def _add_object(self, obj: core.Sphere) -> Optional[int]:
    radius = obj.scale[0]
    assert radius == obj.scale[1] == obj.scale[2], obj.scale  # only uniform scaling
    collision_idx = self._physics_client.createCollisionShape(
        pb.GEOM_SPHERE, radius=radius)
    visual_idx = -1
    mass = 0 if obj.static else obj.mass
    # useMaximalCoordinates and contactProcessingThreshold are required to
    # fix the sticky walls issue;
    # see https://github.com/bulletphysics/bullet3/issues/3094
    sphere_idx = self._physics_client.createMultiBody(
        mass,
        collision_idx,
        visual_idx,
        obj.position,
        wxyz2xyzw(obj.quaternion),
        useMaximalCoordinates=True)
    self._physics_client.changeDynamics(
        sphere_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, sphere_idx, self._physics_client)

    return sphere_idx

  @add_asset.register(core.FileBasedObject)
  def _add_object(self, obj: core.FileBasedObject) -> Optional[int]:
    # TODO: support other file-formats
    if obj.simulation_filename is None:
      return None  # if there is no simulation file, then ignore this object
    path = pathlib.Path(obj.simulation_filename).resolve()
    logger.debug("Loading '%s' in the simulator", path)

    if not path.exists():
      raise IOError(f"File '{path}' does not exist.")

    scale = obj.scale[0]
    assert obj.scale[1] == obj.scale[2] == scale, "Pybullet does not support non-uniform scaling"

    # useMaximalCoordinates and contactProcessingThreshold are required to
    # fix the sticky walls issue;
    # see https://github.com/bulletphysics/bullet3/issues/3094
    if path.suffix == ".urdf":
      obj_idx = self._physics_client.loadURDF(
          str(path),
          useFixedBase=obj.static,
          globalScaling=scale,
          useMaximalCoordinates=True)
      
      # Read the URDF xml file to get the origin offset
      import xml.etree.ElementTree as ET
      try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        # Find the <origin> tag under <link>/<inertial>
        origin_elem = root.find(".//link/inertial/origin")
        if origin_elem is not None and "xyz" in origin_elem.attrib:
          xyz_str = origin_elem.attrib["xyz"]
          xyz = tuple(float(x) for x in xyz_str.strip().split())
          obj.urdf_origin_offset = xyz
      except Exception as e:
        logger.warning(f"Could not parse URDF origin offset from '{path}': {e}")
      
    else:
      raise IOError(
          "Unsupported format '{path.suffix}' of file '{path}'")

    if obj_idx < 0:
      raise IOError(f"Failed to load '{path}'")

    self._physics_client.changeDynamics(
        obj_idx, -1, contactProcessingThreshold=0)

    register_physical_object_setters(obj, obj_idx, self._physics_client)
    return obj_idx

  @add_asset.register(core.SoftBody)
  def _add_soft_body(self, obj: core.SoftBody) -> Optional[int]:
    """Add a soft body to the physics simulation."""
    if obj.simulation_filename is None:
      return None  # if there is no simulation file, then ignore this object
    
    path = pathlib.Path(obj.simulation_filename).resolve()
    logger.debug("Loading soft body '%s' in the simulator", path)

    if not path.exists():
      raise IOError(f"File '{path}' does not exist.")

    # Check for supported soft body formats
    if path.suffix not in [".vtk", ".obj"]:
      raise IOError(f"Unsupported soft body format '{path.suffix}'. Supported formats: .vtk, .obj")

    # Check vertex count to prevent loading overly complex meshes
    nr_vertices = obj.metadata.get('tetrahedral_vertices', 0)
    nr_cells = obj.metadata.get('tetrahedral_cells', 0)
    
    scale = obj.scale[0] 
    assert obj.scale[1] == obj.scale[2] == scale, "PyBullet does not support non-uniform scaling for soft bodies"

    # Print the number of vertices in the soft body
    logger.debug(f"Loading softbody {obj.asset_id} with {nr_vertices} vertices and {nr_cells} cells")

    # Try to load the soft body with timeout and error handling
    initial_position = obj.position
    initial_orientation = wxyz2xyzw(obj.quaternion)
    
    try:
      obj_idx = -1
      if path.suffix == ".vtk":
        logger.info(f"Attempting to load VTK soft body: {path}")
        logger.info(f"Parameters: mass={obj.mass}, friction={obj.friction}, scale={scale}")

        obj_idx = self._physics_client.loadSoftBody(
            str(path),
            basePosition=initial_position,
            baseOrientation=initial_orientation,
            scale=scale,
            mass=obj.mass,
            frictionCoeff=obj.friction,
            useNeoHookean=obj.use_neo_hookean,
            NeoHookeanMu=obj.neo_hookean_mu if obj.use_neo_hookean else 100.0,
            NeoHookeanLambda=obj.neo_hookean_lambda if obj.use_neo_hookean else 100.0,
            NeoHookeanDamping=obj.neo_hookean_damping if obj.use_neo_hookean else 0.01,
            useBendingSprings=obj.use_bending_constraints,
            useMassSpring=obj.use_mass_spring,
            springElasticStiffness=obj.spring_elastic_stiffness,
            springDampingStiffness=obj.spring_damping_stiffness,
            springBendingStiffness=obj.spring_bending_stiffness)  # Improved contact handling
      
        # Load tet to tri mapping
        tri_to_tet_mapping_path = pathlib.Path(obj.tri_to_tet_mapping_filename).resolve()
        if tri_to_tet_mapping_path.exists():
          with open(tri_to_tet_mapping_path, "r") as f:
            # Ignore lines that start with '#'
            tri_to_tet_mapping = [list(map(int, line.split())) for line in f.readlines() if not line.startswith('#')]
          obj.tri_to_tet_mapping = tri_to_tet_mapping
          logger.info(f"Loaded tri-to-tet mapping with {len(tri_to_tet_mapping)} surface vertices")
        else:
          logger.warning(f"No tri to tet mapping file found at {tri_to_tet_mapping_path}")
          obj.tri_to_tet_mapping = None
      else:
        logger.info(f"Attempting to load OBJ soft body: {path}")
        obj_idx = self._physics_client.loadSoftBody(
              str(path),
              basePosition=initial_position,
              baseOrientation=initial_orientation,
              scale=scale,
              mass=obj.mass,
              useNeoHookean=obj.use_neo_hookean,
              NeoHookeanMu=obj.neo_hookean_mu if obj.use_neo_hookean else 100.0,
              NeoHookeanLambda=obj.neo_hookean_lambda if obj.use_neo_hookean else 100.0,
              NeoHookeanDamping=obj.neo_hookean_damping if obj.use_neo_hookean else 0.01,
              useBendingSprings=obj.use_bending_constraints,
              useMassSpring=obj.use_mass_spring,
              springElasticStiffness=obj.spring_elastic_stiffness,
              springDampingStiffness=obj.spring_damping_stiffness,
              springBendingStiffness=obj.spring_bending_stiffness)
              
    except Exception as e:
      logger.error(f"Failed to load soft body '{path}' - PyBullet returned invalid index")
      logger.error(f"Error: {e}")
      logger.error(f"Error type: {type(e).__name__}")
      import traceback
      logger.error(f"Traceback: {traceback.format_exc()}")
      return None

    if obj_idx < 0:
      logger.error(f"Failed to load soft body '{path}' - PyBullet returned invalid index {obj_idx}")
      return None

    logger.info(f"Successfully loaded soft body with index {obj_idx}")

    # Apply additional configuration only if loading succeeded
    self._physics_client.changeDynamics(obj_idx, -1, 
                                        activationState=2)  # Keep active

    # Register soft body specific setters
    register_soft_body_setters(obj, obj_idx, self._physics_client)
          
    return obj_idx

  def check_overlap(self, obj: core.PhysicalObject) -> bool:
    obj_idx = obj.linked_objects[self]

    body_ids = [
        self._physics_client.getBodyUniqueId(i)
        for i in range(self._physics_client.getNumBodies())
    ]
    for body_id in body_ids:
      if body_id == obj_idx:
        continue
      overlap_points = self._physics_client.getClosestPoints(
          obj_idx, body_id, distance=0)
      if overlap_points:
        return True
    return False
  
  def check_aabbox_overlap(self, obj: core.PhysicalObject) -> bool:
    obj_idx = obj.linked_objects[self]

    # Check if the object bounding box overlaps with any other object's bounding box
    obj_aabbox = obj.aabbox
    overlapping_objects = self._physics_client.getOverlappingObjects(obj_aabbox[0], obj_aabbox[1])
    if len(overlapping_objects) > 0:
      return True
    
    return False

  def get_position_and_rotation(self, obj_idx: int):
    pos, quat = self._physics_client.getBasePositionAndOrientation(obj_idx)
    return pos, xyzw2wxyz(quat)  # convert quaternion format

  def get_velocities(self, obj_idx: int):
    velocity, angular_velocity = self._physics_client.getBaseVelocity(obj_idx)
    return velocity, angular_velocity

  def save_state(self, path: Union[pathlib.Path, str] = "scene.bullet"):
    """Receives a folder path as input."""
    assert self.scratch_dir is not None
    # first store in a temporary file and then copy, to support remote paths
    self._physics_client.saveBullet(str(self.scratch_dir / "scene.bullet"))
    tf.io.gfile.copy(self.scratch_dir / "scene.bullet", path, overwrite=True)

  def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
    """Set the gravity vector for the physics simulation.
    
    Args:
      gravity: A tuple of (x, y, z) components of the gravity vector.
    """
    self._physics_client.setGravity(*gravity)
    # Update the scene's gravity to maintain consistency
    self.scene.gravity = gravity

  def run(self,
        frame_start: int = 0,
        frame_end: Optional[int] = None
) -> Tuple[Dict[core.PhysicalObject, Dict[str, list]], List[dict]]:
    """
    Run the physics simulation with proper soft body coordinate handling.
    """

    frame_end = self.scene.frame_end if frame_end is None else frame_end
    steps_per_frame = self.scene.step_rate // self.scene.frame_rate
    max_step = (frame_end - frame_start + 1) * steps_per_frame

    obj_idxs = [
        self._physics_client.getBodyUniqueId(i)
        for i in range(self._physics_client.getNumBodies())
    ]
    
    # Regular animation data for all bodies (including soft bodies' base transforms)
    animation = {obj_id: {"position": [], "quaternion": [], "velocity": [], "angular_velocity": []}
                for obj_id in obj_idxs}
    
    # Track soft body deformation data separately
    soft_body_animation = {}
    soft_body_assets = {}
    
    # Identify soft bodies in the scene
    for asset in self.scene.assets:
      if isinstance(asset, core.SoftBody) and asset.linked_objects.get(self) is not None:
        soft_body_idx = asset.linked_objects[self]
        soft_body_assets[soft_body_idx] = asset
        soft_body_animation[soft_body_idx] = {"vertex_positions": []}

    collisions = []
    for current_step in range(max_step):

      contact_points = self._physics_client.getContactPoints()
      for collision in contact_points:
        (contact_flag,
        body_a, body_b,
        link_a, link_b,
        position_a, position_b, contact_normal_b,
        contact_distance, normal_force,
        lateral_friction1, lateral_friction_dir1,
        lateral_friction2, lateral_friction_dir2) = collision
        del link_a, link_b, contact_flag, contact_distance, position_a
        del lateral_friction1, lateral_friction2
        del lateral_friction_dir1, lateral_friction_dir2
        if normal_force > 1e-6:
          collisions.append({
              "instances": (self._obj_idx_to_asset(body_b), self._obj_idx_to_asset(body_a)),
              "position": position_b,
              "contact_normal": contact_normal_b,
              "frame": current_step / steps_per_frame,
              "force": normal_force,
          })

      # Store animation data at frame intervals
      if current_step % steps_per_frame == 0:
        
        # Handle ALL bodies (both rigid and soft) for base transforms
        start_time = time.time()
        for obj_idx in obj_idxs:
          position, quaternion = self.get_position_and_rotation(obj_idx)
          velocity, angular_velocity = self.get_velocities(obj_idx)

          animation[obj_idx]["position"].append(position)
          animation[obj_idx]["quaternion"].append(quaternion)
          animation[obj_idx]["velocity"].append(velocity)
          animation[obj_idx]["angular_velocity"].append(angular_velocity)
        base_transform_time = time.time() - start_time
        logger.debug(f"Base transform processing took {base_transform_time:.4f}s for {len(obj_idxs)} objects")
        
        # Handle soft body deformation (in addition to base transforms)
        start_time = time.time()
        for soft_body_idx, asset in soft_body_assets.items():
          # Use surface positions if tri-to-tet mapping is available
          soft_body_data = self.get_surface_positions_from_tet(soft_body_idx, asset)
          if soft_body_data["num_vertices"] > 0:
            soft_body_animation[soft_body_idx]["vertex_positions"].append(
                soft_body_data["positions"].copy())
            logger.debug(f"Stored {soft_body_data['num_vertices']} vertices for soft body {soft_body_idx}")
        soft_body_time = time.time() - start_time
        logger.debug(f"Soft body deformation processing took {soft_body_time:.4f}s for {len(soft_body_assets)} soft bodies")
      
      start_time = time.time()
      if self._persistent_forces:
        for persistent in self._persistent_forces:
          self._physics_client.applyExternalForce(
              persistent["obj_idx"],
              persistent["link_idx"],
              persistent["force"],
              persistent["point"],
              persistent["frame"],
          )
      if self._persistent_velocities:
        for velocity in self._persistent_velocities:
          angular = velocity["angular"]
          duration = velocity["duration"]
          if angular is None:
            _, angular_current = self._physics_client.getBaseVelocity(velocity["obj_idx"])
            angular = angular_current
          if duration is not None and current_step >= (duration - frame_start)*steps_per_frame:
            continue
          self._physics_client.resetBaseVelocity(
              velocity["obj_idx"],
              velocity["linear"],
              angular,
          )
      self._physics_client.stepSimulation()
      logger.debug(f"Total simulation step took {time.time() - start_time:.4f}s")

    # Convert animation indices back to assets
    start_time = time.time()
    animation = {asset: animation[asset.linked_objects[self]] for asset in self.scene.assets
                if asset.linked_objects.get(self) in obj_idxs}
    animation_conversion_time = time.time() - start_time
    logger.debug(f"Animation index conversion took {animation_conversion_time:.4f}s")

    # --- Transfer simulation to renderer keyframes
    start_time = time.time()
    if frame_start >= 0:
      for obj in animation.keys():
        # ALL objects (including soft bodies) get their base transform keyframes
        for frame_id in range(frame_end - frame_start + 1):
          # Get the original position and rotation from PyBullet
          pybullet_position = animation[obj]["position"][frame_id]
          pybullet_quaternion = animation[obj]["quaternion"][frame_id]
          
          # Apply URDF origin compensation for FileBasedObjects (handles both position and rotation)
          adjusted_position = self._compensate_urdf_origin_offset(obj, pybullet_position, pybullet_quaternion)
          
          obj.position = adjusted_position
          obj.quaternion = pybullet_quaternion
          obj.velocity = animation[obj]["velocity"][frame_id]
          obj.angular_velocity = animation[obj]["angular_velocity"][frame_id]
          obj.keyframe_insert("position", frame_id + frame_start)
          obj.keyframe_insert("quaternion", frame_id + frame_start)
          obj.keyframe_insert("velocity", frame_id + frame_start)
          obj.keyframe_insert("angular_velocity", frame_id + frame_start)
    keyframe_transfer_time = time.time() - start_time
    logger.debug(f"Keyframe transfer took {keyframe_transfer_time:.4f}s for {len(animation)} objects")

    # --- Transfer soft body vertex deformation keyframes (additional to base transform)
    start_time = time.time()
    if frame_start >= 0:
      for soft_body_idx, asset in soft_body_assets.items():
        vertex_positions_over_time = soft_body_animation[soft_body_idx]["vertex_positions"]
        if vertex_positions_over_time:
          # Store vertex positions in the asset for the renderer to access
          asset.vertex_positions_animation = vertex_positions_over_time
          
          # Create keyframes for mesh deformation
          for frame_id in range(frame_end - frame_start + 1):
            if frame_id < len(vertex_positions_over_time):
              asset.current_vertex_positions = vertex_positions_over_time[frame_id]
              # Insert keyframe to notify renderer of mesh deformation
              asset.keyframe_insert("vertex_positions", frame_id + frame_start)
              logger.debug(f"Inserted vertex keyframe for soft body {asset.uid} at frame {frame_id + frame_start}")
    soft_body_keyframe_time = time.time() - start_time
    logger.debug(f"Soft body keyframe transfer took {soft_body_keyframe_time:.4f}s for {len(soft_body_assets)} soft bodies")

    # Update animation data with soft body animation data
    for soft_body_idx, soft_asset in soft_body_assets.items():
      for key, value in soft_body_animation[soft_body_idx].items():
        animation[soft_asset][key] = value

    logger.info(f"Completed simulation with {len(collisions)} collision events")
    logger.info(f"Processed {len(soft_body_assets)} soft bodies")
    
    return animation, collisions

  def _compensate_urdf_origin_offset(self, obj, pybullet_position, pybullet_quaternion):
    """Compensate for URDF origin offsets when transferring positions to renderer.
    
    When an object rotates in PyBullet around its physics center (URDF origin),
    but the visual mesh needs to rotate around its mesh center (OBJ origin),
    we need to account for the rotated offset vector.
    
    Args:
      obj: The Kubric object
      pybullet_position: Position from PyBullet simulation
      pybullet_quaternion: Quaternion from PyBullet simulation (w, x, y, z)
      
    Returns:
      Adjusted position for rendering
    """
    # Only apply compensation to FileBasedObjects with URDF files
    if not isinstance(obj, core.FileBasedObject):
      return pybullet_position
      
    if obj.simulation_filename is None:
      return pybullet_position
      
    # path = pathlib.Path(obj.simulation_filename).resolve()
    
    # Get URDF origin offset for known objects
    urdf_origin_offset = self._get_urdf_origin_offset(obj)
    
    if urdf_origin_offset is not None:
      # Convert quaternion (w, x, y, z) to rotation matrix
      w, x, y, z = pybullet_quaternion
      
      # Rotation matrix from quaternion
      rotation_matrix = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
      ])
      
      # Rotate the offset vector by the current rotation
      offset_vector = np.array(urdf_origin_offset)
      rotated_offset = rotation_matrix @ offset_vector
      
      # Subtract the rotated offset to get the correct rendering position
      adjusted_position = (
        pybullet_position[0] - rotated_offset[0],
        pybullet_position[1] - rotated_offset[1], 
        pybullet_position[2] - rotated_offset[2]
      )
      
      logger.debug(f"Applied URDF origin compensation for {obj.name}: "
                  f"PyBullet pos {pybullet_position}, offset {urdf_origin_offset}, "
                  f"rotated offset {rotated_offset} -> Render pos {adjusted_position}")
      return adjusted_position
    
    return pybullet_position
  
  def _get_urdf_origin_offset(self, obj):
    """Get the URDF origin offset for known objects.
    
    Args:
      obj: The Kubric FileBasedObject
      
    Returns:
      Tuple of (x, y, z) origin offset or None if no offset needed
    """
    # Check if this is the brick object with known origin offset
    if (obj.simulation_filename and hasattr(obj, 'urdf_origin_offset') and obj.urdf_origin_offset is not None):
      return obj.urdf_origin_offset
    
    # TODO: Add other objects with URDF origin offsets here as needed
    # Example:
    # if "other_object.urdf" in str(obj.simulation_filename):
    #   return (offset_x, offset_y, offset_z)
    
    return None


  def _obj_idx_to_asset(self, idx):
    assets = [asset for asset in self.scene.assets if asset.linked_objects.get(self) == idx]
    if len(assets) == 1:
      return assets[0]
    elif len(assets) == 0:
      return None
    else:
      raise RuntimeError("Multiple assets linked to same pybullet object. That should never happen")
    
  def get_obj_idx(self, asset: core.PhysicalObject) -> int:
    """Get the pybullet object index for a Kubric asset."""
    return asset.linked_objects[self]

  def apply_force(self, obj_idx: int, force: Tuple[float, float, float], point: Tuple[float, float, float] = [0, 0, 0], frame: int = pb.LINK_FRAME) -> None:
    """Apply a linear force to an object.

    Args:
      obj_idx: The pybullet object index to apply force to
      force: A tuple of (x, y, z) components of the force vector
      point: A tuple of (x, y, z) coordinates where the force should be applied
    """
    self._physics_client.applyExternalForce(obj_idx, -1, force, point, frame)

  def add_persistent_force(self, obj_idx: int, force: Tuple[float, float, float],
                           point: Tuple[float, float, float] = (0., 0., 0.),
                           frame: int = pb.LINK_FRAME, link_idx: int = -1) -> None:
    """Register a force that should be applied on every simulation substep."""
    self._persistent_forces.append({
        "obj_idx": obj_idx,
        "link_idx": link_idx,
        "force": tuple(force),
        "point": tuple(point),
        "frame": frame,
    })

  def clear_persistent_forces(self) -> None:
    """Remove all persistent forces before the next run."""
    self._persistent_forces.clear()

  def add_persistent_velocity(self, obj_idx: int,
                               linear_velocity: Tuple[float, float, float],
                               angular_velocity: Optional[Tuple[float, float, float]] = None, 
                               duration: Optional[float] = None) -> None:
    """Register a velocity that should be enforced on every simulation substep.
    
    Args:
      obj_idx: The pybullet object index to apply velocity to
      linear_velocity: A tuple of (x, y, z) components of the linear velocity vector
      angular_velocity: A tuple of (x, y, z) components of the angular velocity vector
      duration: The duration of the velocity application in frames (None for infinite duration)
    """
    self._persistent_velocities.append({
        "obj_idx": obj_idx,
        "linear": tuple(linear_velocity),
        "angular": tuple(angular_velocity) if angular_velocity is not None else None,
        "duration": duration,
    })

  def clear_persistent_velocities(self) -> None:
    """Remove all persistent velocity constraints."""
    self._persistent_velocities.clear()

  def apply_torque(self, obj_idx: int, torque: Tuple[float, float, float], frame: int = pb.LINK_FRAME) -> None:
    """Apply a torque (rotational force) to an object.
    
    Args:
      obj_idx: The pybullet object index to apply torque to
      torque: A tuple of (x, y, z) components of the torque vector
      frame: The frame to apply the torque in (pb.WORLD_FRAME or pb.LINK_FRAME)
    """
    self._physics_client.applyExternalTorque(obj_idx, -1, torque, frame)

  def apply_force_at_point(self, obj_idx: int, force: Tuple[float, float, float], 
                          point: Tuple[float, float, float]) -> None:
    """Apply a force at a specific point on the object.
    
    Args:
      obj_idx: The pybullet object index to apply force to
      force: A tuple of (x, y, z) components of the force vector
      point: A tuple of (x, y, z) coordinates where the force should be applied
    """
    self._physics_client.applyExternalForce(obj_idx, -1, force, point, pb.WORLD_FRAME)

  def apply_soft_body_force(self, obj_idx: int, node_index: int, 
                           force: Tuple[float, float, float]) -> None:
    """Apply a force to a specific node of a soft body.
    
    Args:
      obj_idx: The pybullet soft body object index
      node_index: Index of the soft body node to apply force to
      force: A tuple of (x, y, z) components of the force vector
    """
    info = self._physics_client.getSoftBodyData(obj_idx)
    if node_index >= 0 and node_index < len(info[1]):  # info[1] contains node positions
      self._physics_client.addUserData(obj_idx, "soft_force", 
                                      {"node": node_index, "force": force})

  def create_soft_body_anchor(self, soft_body_idx: int, node_index: int, 
                             rigid_body_idx: int = -1, 
                             anchor_position: Optional[Tuple[float, float, float]] = None) -> int:
    """Create an anchor between a soft body node and a rigid body or world position.
    
    Args:
      soft_body_idx: The pybullet soft body object index
      node_index: Index of the soft body node to anchor
      rigid_body_idx: Index of rigid body to anchor to (-1 for world anchor)
      anchor_position: Position to anchor to (for world anchors)
      
    Returns:
      Anchor constraint ID
    """
    if rigid_body_idx == -1 and anchor_position is not None:
      # Create world anchor
      anchor_id = self._physics_client.createSoftBodyAnchor(
          soft_body_idx, node_index, -1, anchor_position)
    elif rigid_body_idx >= 0:
      # Create anchor to rigid body
      anchor_id = self._physics_client.createSoftBodyAnchor(
          soft_body_idx, node_index, rigid_body_idx, [0, 0, 0])
    else:
      raise ValueError("Must specify either rigid_body_idx >= 0 or anchor_position")
    
    return anchor_id

  def remove_soft_body_anchor(self, anchor_id: int) -> None:
    """Remove a soft body anchor constraint.
    
    Args:
      anchor_id: The anchor constraint ID to remove
    """
    self._physics_client.removeConstraint(anchor_id)

  # Fixed get_soft_body_data method for pybullet.py
  def get_soft_body_data(self, body_id: int) -> Dict[str, np.ndarray]:
    """Get the current state of a soft body including vertex positions.
    
    Args:
      body_id: The PyBullet body ID of the soft body
      
    Returns:
      Dictionary containing:
        - positions: Array of vertex positions relative to initial state
        - num_vertices: Number of vertices in the soft body
    """
    # Use getMeshData to get current vertex positions
    kwargs = {}
    if hasattr(pb, "MESH_DATA_SIMULATION_MESH"):
      kwargs["flags"] = pb.MESH_DATA_SIMULATION_MESH
    
    num_vertices, mesh_data = self._physics_client.getMeshData(body_id, **kwargs)
    
    if num_vertices > 0 and mesh_data:
      # Convert mesh data to numpy array of positions (world coordinates)
      world_positions = np.array(mesh_data).reshape((-1, 3))
      
      # KEY FIX: For soft bodies, we need to return positions relative to 
      # the soft body's INITIAL/REST state, not transformed coordinates.
      # The base transform will be handled separately by the normal object transform system.
      
      # Get the current base position and orientation
      try:
        base_pos, base_orn = self._physics_client.getBasePositionAndOrientation(body_id)
        base_pos = np.array(base_pos)
        
        # Convert quaternion to rotation matrix  
        x, y, z, w = base_orn
        rotation_matrix = np.array([
          [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
          [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
          [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Transform world positions to local object space
        # This gives us the deformed shape relative to the object's coordinate system
        inverse_rotation = rotation_matrix.T
        local_positions = np.array([
          inverse_rotation @ (pos - base_pos) for pos in world_positions
        ])
        
        return {
          "positions": local_positions,  # Local deformed positions
          "num_vertices": num_vertices
        }
        
      except Exception as e:
        logger.error(f"Failed to transform soft body coordinates for {body_id}: {e}")
        # Fallback: return world positions and let Blender handle it
        return {
          "positions": world_positions,
          "num_vertices": num_vertices
        }
    else:
      logger.warning(f"No mesh data available for soft body {body_id}")
      return {"positions": np.array([]), "num_vertices": 0}

  def get_surface_positions_from_tet(self, body_id: int, asset: core.SoftBody) -> Dict[str, np.ndarray]:
    """Get surface mesh positions from tetrahedral mesh using tri-to-tet mapping.
    
    SIMPLE APPROACH: Just use the same coordinate transformation as OBJ files.
    """
    sim_fname = pathlib.Path(asset.simulation_filename).resolve()
    
    soft_body_data = self.get_soft_body_data(body_id)
    # For OBJ files, use the standard method
    if sim_fname.suffix == ".obj" or not hasattr(asset, 'tri_to_tet_mapping') or asset.tri_to_tet_mapping is None:
        return soft_body_data

    # For VTK files: Get the tetrahedral data and extract surface vertices
    try:
        # NOW apply the tri-to-tet mapping to get surface positions
        mapping = asset.tri_to_tet_mapping
        mapping_array = np.array(mapping)
        tet_ids = mapping_array[:, 1]
        
        # Extract surface positions from the local tetrahedral positions
        # local_surface_positions = local_tet_positions[tet_ids]
        local_surface_positions = soft_body_data["positions"][tet_ids]
        
        # Debug: Log the centers to verify correctness
        tet_center = np.mean(soft_body_data["positions"], axis=0)
        surface_center = np.mean(local_surface_positions, axis=0)
        logger.debug(f"Local tet center: {tet_center}")
        logger.debug(f"Local surface center: {surface_center}")
        
        logger.debug(f"Mapped {len(local_surface_positions)} surface vertices from {len(soft_body_data['positions'])} tet vertices")
        
        return {
            "positions": local_surface_positions,
            "num_vertices": len(local_surface_positions)
        }
        
    except Exception as e:
        logger.error(f"Failed to map tetrahedral to surface positions: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to standard method
        return self.get_soft_body_data(body_id)
      
  def resetSimulationToDeformableWorld(self):
    """Reset the simulation to a deformable world.
    """
    self._physics_client.resetSimulation(self._physics_client.RESET_USE_DEFORMABLE_WORLD)

def xyzw2wxyz(xyzw):
  """Convert quaternions from XYZW format to WXYZ."""
  x, y, z, w = xyzw
  return w, x, y, z


def wxyz2xyzw(wxyz):
  """Convert quaternions from WXYZ format to XYZW."""
  w, x, y, z = wxyz
  return x, y, z, w


def register_physical_object_setters(obj: core.PhysicalObject, obj_idx,
                                     physics_client: _BulletClient):
  assert isinstance(obj, core.PhysicalObject), f"{obj!r} is not a PhysicalObject"

  def setter(object_idx, func):
    def _callable(change):
      return func(object_idx, change.new, change.owner, physics_client)
    return _callable

  obj.observe(setter(obj_idx, set_position), "position")
  obj.observe(setter(obj_idx, set_quaternion), "quaternion")
  # TODO Pybullet does not support rescaling. So we should warn if scale is changed
  obj.observe(setter(obj_idx, set_velocity), "velocity")
  obj.observe(setter(obj_idx, set_angular_velocity), "angular_velocity")
  obj.observe(setter(obj_idx, set_friction), "friction")
  obj.observe(setter(obj_idx, set_rolling_friction), "rolling_friction")
  obj.observe(setter(obj_idx, set_spinning_friction), "spinning_friction")
  obj.observe(setter(obj_idx, set_restitution), "restitution")
  obj.observe(setter(obj_idx, set_mass), "mass")
  obj.observe(setter(obj_idx, set_static), "static")


def set_position(object_idx, position, asset, physics_client: _BulletClient):  # pylint: disable=unused-argument
  # reuse existing quaternion
  _, quaternion = physics_client.getBasePositionAndOrientation(object_idx)
  # resetBasePositionAndOrientation zeroes out velocities, but we wish to conserve them
  velocity, angular_velocity = physics_client.getBaseVelocity(object_idx)
  physics_client.resetBasePositionAndOrientation(object_idx, position, quaternion)
  physics_client.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_quaternion(object_idx, quaternion, asset, physics_client: _BulletClient):  # pylint: disable=unused-argument
  quaternion = wxyz2xyzw(quaternion)  # convert quaternion format
  # reuse existing position
  position, _ = physics_client.getBasePositionAndOrientation(object_idx)
  # resetBasePositionAndOrientation zeroes out velocities, but we wish to conserve them
  velocity, angular_velocity = physics_client.getBaseVelocity(object_idx)
  physics_client.resetBasePositionAndOrientation(object_idx, position,
                                                 quaternion)
  physics_client.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_velocity(object_idx, velocity, asset, physics_client: _BulletClient):  # pylint: disable=unused-argument
  _, angular_velocity = physics_client.getBaseVelocity(object_idx)  # reuse existing angular velocity
  physics_client.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_angular_velocity(object_idx, angular_velocity, asset,
                         physics_client: _BulletClient):  # pylint: disable=unused-argument
  velocity, _ = physics_client.getBaseVelocity(object_idx)  # reuse existing velocity
  physics_client.resetBaseVelocity(object_idx, velocity, angular_velocity)


def set_mass(object_idx, mass: float, asset, physics_client: _BulletClient):
  if mass < 0:
    raise ValueError(f"mass cannot be negative ({mass})")
  if not asset.static:
    physics_client.changeDynamics(object_idx, -1, mass=mass)


def set_static(object_idx, is_static, asset, physics_client: _BulletClient):
  if is_static:
    physics_client.changeDynamics(object_idx, -1, mass=0.)
  else:
    physics_client.changeDynamics(object_idx, -1, mass=asset.mass)


def set_friction(object_idx, friction: float, asset, physics_client: _BulletClient):  # pylint: disable=unused-argument
  if friction < 0:
    raise ValueError("friction cannot be negative ({friction})")
  physics_client.changeDynamics(object_idx, -1, lateralFriction=friction)


def set_restitution(object_idx, restitution: float, asset, physics_client: _BulletClient):  # pylint: disable=unused-argument
  if restitution < 0:
    raise ValueError("restitution cannot be negative ({restitution})")
  if restitution > 1:
    raise ValueError("restitution should be below 1.0 ({restitution})")
  physics_client.changeDynamics(object_idx, -1, restitution=restitution)


def register_soft_body_setters(obj: core.SoftBody, obj_idx,
                              physics_client: _BulletClient):
  assert isinstance(obj, core.SoftBody), f"{obj!r} is not a SoftBody"

  def setter(object_idx, func):
    def _callable(change):
      return func(object_idx, change.new, change.owner, physics_client)
    return _callable

  # obj.observe(setter(obj_idx, set_neo_hookean), "use_neo_hookean")
  # obj.observe(setter(obj_idx, set_neo_hookean_mu), "neo_hookean_mu")
  # obj.observe(setter(obj_idx, set_neo_hookean_lambda), "neo_hookean_lambda")
  # obj.observe(setter(obj_idx, set_neo_hookean_damping), "neo_hookean_damping")
  # obj.observe(setter(obj_idx, set_use_bending_constraints), "use_bending_constraints")
  # obj.observe(setter(obj_idx, set_use_mass_spring), "use_mass_spring")
  # obj.observe(setter(obj_idx, set_spring_elastic_stiffness), "spring_elastic_stiffness")
  # obj.observe(setter(obj_idx, set_spring_damping_stiffness), "spring_damping_stiffness")
  # obj.observe(setter(obj_idx, set_spring_bending_stiffness), "spring_bending_stiffness")
  # obj.observe(setter(obj_idx, set_self_collision), "self_collision")
  obj.observe(setter(obj_idx, set_collision_margin), "collision_margin")
  obj.observe(setter(obj_idx, set_cluster_count), "cluster_count")


def set_neo_hookean(object_idx, use_neo_hookean, asset, physics_client: _BulletClient):
  if use_neo_hookean:
    physics_client.changeDynamics(object_idx, -1, useNeoHookean=1)
  else:
    physics_client.changeDynamics(object_idx, -1, useNeoHookean=0)


def set_neo_hookean_mu(object_idx, neo_hookean_mu, asset, physics_client: _BulletClient):
  if neo_hookean_mu < 0:
    raise ValueError(f"neo_hookean_mu cannot be negative ({neo_hookean_mu})")
  physics_client.changeDynamics(object_idx, -1, NeoHookeanMu=neo_hookean_mu)


def set_neo_hookean_lambda(object_idx, neo_hookean_lambda, asset, physics_client: _BulletClient):
  if neo_hookean_lambda < 0:
    raise ValueError(f"neo_hookean_lambda cannot be negative ({neo_hookean_lambda})")
  physics_client.changeDynamics(object_idx, -1, NeoHookeanLambda=neo_hookean_lambda)


def set_neo_hookean_damping(object_idx, neo_hookean_damping, asset, physics_client: _BulletClient):
  if neo_hookean_damping < 0:
    raise ValueError(f"neo_hookean_damping cannot be negative ({neo_hookean_damping})")
  physics_client.changeDynamics(object_idx, -1, NeoHookeanDamping=neo_hookean_damping)


def set_use_bending_constraints(object_idx, use_bending_constraints, asset, physics_client: _BulletClient):
  if use_bending_constraints:
    physics_client.changeDynamics(object_idx, -1, useBendingSprings=1)
  else:
    physics_client.changeDynamics(object_idx, -1, useBendingSprings=0)


def set_use_mass_spring(object_idx, use_mass_spring, asset, physics_client: _BulletClient):
  if use_mass_spring:
    physics_client.changeDynamics(object_idx, -1, useMassSpring=1)
  else:
    physics_client.changeDynamics(object_idx, -1, useMassSpring=0)


def set_spring_elastic_stiffness(object_idx, spring_elastic_stiffness, asset, physics_client: _BulletClient):
  if spring_elastic_stiffness < 0:
    raise ValueError(f"spring_elastic_stiffness cannot be negative ({spring_elastic_stiffness})")
  physics_client.changeDynamics(object_idx, -1, springElasticStiffness=spring_elastic_stiffness)


def set_spring_damping_stiffness(object_idx, spring_damping_stiffness, asset, physics_client: _BulletClient):
  if spring_damping_stiffness < 0:
    raise ValueError(f"spring_damping_stiffness cannot be negative ({spring_damping_stiffness})")
  physics_client.changeDynamics(object_idx, -1, springDampingStiffness=spring_damping_stiffness)


def set_spring_bending_stiffness(object_idx, spring_bending_stiffness, asset, physics_client: _BulletClient):
  if spring_bending_stiffness < 0:
    raise ValueError(f"spring_bending_stiffness cannot be negative ({spring_bending_stiffness})")
  physics_client.changeDynamics(object_idx, -1, springBendingStiffness=spring_bending_stiffness)


def set_self_collision(object_idx, self_collision, asset, physics_client: _BulletClient):
  if self_collision:
    physics_client.changeDynamics(object_idx, -1, useSelfCollision=1)
  else:
    physics_client.changeDynamics(object_idx, -1, useSelfCollision=0)


def set_collision_margin(object_idx, collision_margin, asset, physics_client: _BulletClient):
  if collision_margin < 0:
    raise ValueError(f"collision_margin cannot be negative ({collision_margin})")
  physics_client.changeDynamics(object_idx, -1, collisionMargin=collision_margin)


def set_cluster_count(object_idx, cluster_count, asset, physics_client: _BulletClient):
  if cluster_count < 0:
    raise ValueError(f"cluster_count cannot be negative ({cluster_count})")
  physics_client.changeDynamics(object_idx, -1, activationState=2 if cluster_count > 0 else 0)


def set_rolling_friction(object_idx, rolling_friction, asset, physics_client: _BulletClient):
  if rolling_friction < 0:
    raise ValueError(f"rolling_friction cannot be negative ({rolling_friction})")
  physics_client.changeDynamics(object_idx, -1, rollingFriction=rolling_friction)


def set_spinning_friction(object_idx, spinning_friction, asset, physics_client: _BulletClient):
  if spinning_friction < 0:
    raise ValueError(f"spinning_friction cannot be negative ({spinning_friction})")
  physics_client.changeDynamics(object_idx, -1, spinningFriction=spinning_friction)
