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

import collections
from contextlib import redirect_stdout
import functools
import io
import logging
import os
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence, Union

import kubric as kb
from kubric import core
from kubric import file_io
from kubric.core.assets import UndefinedAsset
from kubric.file_io import PathLike
from kubric.redirect_io import RedirectStream
from kubric.renderer import blender_utils
from kubric.safeimport.bpy import bpy
import numpy as np
import tensorflow as tf
import time

logger = logging.getLogger(__name__)


def add_top_level_empty_parent(name: str = "Empty") -> bpy.types.Object:
  """Adds an empty parent to scene and makes it the parent of all objects.

  Args:
    name: The name of the empty parent.

  Returns:
    The newly created empty parent.
  """
  parent_obj = bpy.data.objects.new(name, None)
  parent_obj.rotation_mode = "QUATERNION"
  bpy.context.scene.collection.objects.link(parent_obj)
  for obj in bpy.context.scene.objects:
    if obj != parent_obj and obj.parent is None:
      obj.parent = parent_obj
  return parent_obj


# noinspection PyUnresolvedReferences
class Blender(core.View):
  """ An implementation of a rendering backend in Blender/Cycles."""

  def __init__(self,
               scene: core.Scene,
               scratch_dir=None,
               adaptive_sampling=False,
               use_denoising=True,
               samples_per_pixel=128,
               background_transparency=False,
               verbose: bool = False,
               custom_scene: Optional[str] = None,
               motion_blur: Optional[float] = None,
               default_layers: Optional[Sequence[str]] = None,
               aux_layers: Optional[Sequence[str]] = None,
               ):
    """
    Args:
      scene: the kubric scene this class will observe
      scratch_dir: Blender always writes the rendered images to disk. The scratch_dir is the
        (temporary) directory used for that. The results are read into memory by kubric,
        immediately after the rendering is done, so the contents of this directory can
        be discarded afterwards.
      adaptive_sampling: Adjust the number of rays cast based on the complexity of the patch
        (see https://docs.blender.org/manual/en/latest/render/cycles/render_settings/sampling.html)
      use_denoising: Use the blender denoiser to improve the image quality.
        (see https://docs.blender.org/manual/en/latest/render/layers/denoising.html#denoising)
      samples_per_pixel: Number of rays cast per pixel
        (see https://docs.blender.org/manual/en/latest/render/cycles/render_settings/sampling.html)
      background_transparency: Render the background transparent.
        (see https://docs.blender.org/manual/en/latest/render/cycles/render_settings/film.html)
      verbose: when False, blender stdout is redirected to stdnull
      custom_scene: By default (None) Blender is initialized with an empty scene.
        If this argument is set to the path for a `.blend` file, then that scene is loaded instead.
        Note that this scene only affects the rendering output. It is not accessible from Kubric and
        not taken into account by the simulator.
    """
    self.scratch_dir = tempfile.mkdtemp() if scratch_dir is None else scratch_dir
    self.ambient_node = None
    self.ambient_hdri_node = None
    self.illum_mapping_node = None
    self.bg_node = None
    self.bg_hdri_node = None
    self.bg_mapping_node = None
    self.verbose = verbose

    # blender has a default scene on load, so we clear everything first
    self.clear_and_reset_blender_scene(self.verbose, custom_scene=custom_scene)
    self.blender_scene = bpy.context.scene

    # the ray-tracing engine is set here because it affects the availability of some features
    bpy.context.scene.render.engine = "CYCLES"
    self.use_gpu = os.getenv("KUBRIC_USE_GPU", "False").lower() in ("true", "1", "t")

    if aux_layers is None:
      aux_layers = ("UV", "Normal", "CryptoObject00", "ObjectCoordinates")
    if default_layers is None:
      default_layers = ("Image", "Depth")

    normal_pass, optical_flow_pass, segmentation_pass, uv_pass, depth_pass = False, False, False, False, False
    normal_pass = True if "Normal" in default_layers or "Normal" in aux_layers else False
    optical_flow_pass = True if "ForwardFlow" in default_layers or "ForwardFlow" in aux_layers else False
    segmentation_pass = True if "CryptoObject00" in default_layers or "CryptoObject00" in aux_layers else False
    uv_pass = True if "UV" in default_layers or "UV" in aux_layers else False
    depth_pass = True if "Depth" in default_layers or "Depth" in aux_layers else False
    blender_utils.activate_render_passes(normal=normal_pass, optical_flow=optical_flow_pass, segmentation=segmentation_pass, uv=uv_pass, depth=depth_pass)
    self._setup_scene_shading()

    self.adaptive_sampling = adaptive_sampling  # speeds up rendering
    self.use_denoising = use_denoising  # improves the output quality
    self.samples_per_pixel = samples_per_pixel
    self.background_transparency = background_transparency

    logging.info(f"Rendering with default layers: {default_layers}")
    logging.info(f"Rendering with aux layers: {aux_layers}")
    self.exr_output_node = blender_utils.set_up_exr_output_node(motion_blur=motion_blur, aux_layers=tuple(aux_layers), default_layers=tuple(default_layers))

    self.post_processors = {
        "backward_flow": blender_utils.process_backward_flow,
        "forward_flow": blender_utils.process_forward_flow,
        "depth": blender_utils.process_depth,
        "z": blender_utils.process_z,
        "uv": blender_utils.process_uv,
        "normal": blender_utils.process_normal,
        "object_coordinates": blender_utils.process_object_coordinates,
        "segmentation": blender_utils.process_segementation,
        "rgb": blender_utils.process_rgb,
        "rgba": blender_utils.process_rgba,
    }

    super().__init__(scene, scene_observers={
        "frame_start": [AttributeSetter(self.blender_scene, "frame_start")],
        "frame_end": [AttributeSetter(self.blender_scene, "frame_end")],
        "frame_rate": [AttributeSetter(self.blender_scene.render, "fps")],
        "resolution": [AttributeSetter(self.blender_scene.render, "resolution_x",
                                       converter=lambda x: x[0]),
                       AttributeSetter(self.blender_scene.render, "resolution_y",
                                       converter=lambda x: x[1])],
        "camera": [AttributeSetter(self.blender_scene, "camera",
                                   converter=self._convert_to_blender_object)],
        "ambient_illumination": [lambda change: self._set_ambient_light_color(change.new)],
        "background": [lambda change: self._set_background_color(change.new)],
    })

  @property
  def scratch_dir(self) -> Union[PathLike, None]:
    return self._scratch_dir

  @scratch_dir.setter
  def scratch_dir(self, value: Union[PathLike, None]):
    if value is None:
      self._scratch_dir = None
    else:
      self._scratch_dir = kb.as_path(value)
      self._scratch_dir.mkdir(parents=True, exist_ok=True)

  @property
  def adaptive_sampling(self) -> bool:
    return self.blender_scene.cycles.use_adaptive_sampling

  @adaptive_sampling.setter
  def adaptive_sampling(self, value: bool):
    self.blender_scene.cycles.use_adaptive_sampling = value

  @property
  def use_denoising(self) -> bool:
    return self.blender_scene.cycles.use_denoising

  @use_denoising.setter
  def use_denoising(self, value: bool):
    self.blender_scene.cycles.use_denoising = value
    if bpy.app.version < (3, 0, 0):
      # NLM is removed since Blender 3. TODO: check if denoising still works
      self.blender_scene.cycles.denoiser = "NLM"

  @property
  def samples_per_pixel(self) -> int:
    return self.blender_scene.cycles.samples

  @samples_per_pixel.setter
  def samples_per_pixel(self, nr: int):
    self.blender_scene.cycles.samples = nr

  @property
  def background_transparency(self) -> bool:
    return self.blender_scene.render.film_transparent

  @background_transparency.setter
  def background_transparency(self, value: bool):
    self.blender_scene.render.film_transparent = value

  @property
  def use_gpu(self) -> bool:
    return self.blender_scene.cycles.device == "GPU"

  @use_gpu.setter
  def use_gpu(self, value: bool):
    self.blender_scene.cycles.device = "GPU" if value else "CPU"
    if value:
      # Optimize CPU usage during GPU rendering
      self._optimize_cpu_usage_for_gpu_rendering()
      # call get_devices() to let Blender detect GPU devices
      cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
      
      # Set compute device type to CUDA for better GPU support
      if hasattr(cycles_prefs, 'compute_device_type'):
        for device_type in ['CUDA', 'OPTIX', 'OPENCL', 'HIP']:
          try:
            cycles_prefs.compute_device_type = device_type
            logger.info(f"Set compute device type to: {device_type}")
            break
          except:
            continue
      
      cycles_prefs.get_devices()
      
      # Check if CUDA_VISIBLE_DEVICES is set to restrict to specific GPU
      visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
      if visible_devices is not None:
        try:
          # Parse visible devices (could be single number or comma-separated)
          if ',' in visible_devices:
            allowed_gpu_ids = [int(x.strip()) for x in visible_devices.split(',')]
          else:
            allowed_gpu_ids = [int(visible_devices)]
          
          logger.info(f"CUDA_VISIBLE_DEVICES set to: {visible_devices}")
          logger.info(f"Will only use GPUs with IDs: {allowed_gpu_ids}")
          
          # Find and enable only the specified GPU devices
          gpu_devices = [d for d in cycles_prefs.devices 
                        if d.type in ['CUDA', 'OPTIX', 'OPENCL', 'HIP', 'METAL']]
          
          enabled_count = 0
          for device in cycles_prefs.devices:
            if device.type == 'CPU':
              device.use = False  # Always disable CPU when using GPU
            elif device.type in ['CUDA', 'OPTIX', 'OPENCL', 'HIP', 'METAL']:
              # For CUDA devices, check if the device ID matches allowed ones
              # This is a heuristic since Blender doesn't directly expose GPU IDs
              # We enable the first N GPUs where N is the length of allowed_gpu_ids
              if enabled_count < len(allowed_gpu_ids):
                device.use = True
                enabled_count += 1
                logger.info(f"Enabled GPU device: {device.name} ({device.type})")
              else:
                device.use = False
            else:
              device.use = False
          
          if enabled_count == 0:
            logger.warning("No GPU devices enabled! Falling back to CPU rendering.")
            self.blender_scene.cycles.device = "CPU"
          else:
            logger.info(f"Successfully restricted to {enabled_count} GPU(s)")
            
        except (ValueError, TypeError) as e:
          logger.warning(f"Could not parse CUDA_VISIBLE_DEVICES '{visible_devices}': {e}")
          # Fall back to enabling all GPUs
          self._enable_all_gpus(cycles_prefs)
      else:
        # No restriction, enable all GPUs (original behavior)
        logger.info("No CUDA_VISIBLE_DEVICES restriction, enabling all GPUs")
        self._enable_all_gpus(cycles_prefs)

  def _enable_all_gpus(self, cycles_prefs):
    """Helper method to enable all available GPU devices."""
    # Find and enable GPU devices
    gpu_devices = [d for d in cycles_prefs.devices 
                  if d.type in ['CUDA', 'OPTIX', 'OPENCL', 'HIP', 'METAL']]
    
    if gpu_devices:
      # Enable all GPU devices
      for device in gpu_devices:
        device.use = True
        logger.info(f"Enabled GPU device: {device.name} ({device.type})")
        
      # Disable CPU devices to force GPU usage
      cpu_devices = [d for d in cycles_prefs.devices if d.type == 'CPU']
      for device in cpu_devices:
        device.use = False
    else:
      logger.warning("No GPU devices found! Falling back to CPU rendering.")
      logger.warning("Check if nvidia-smi works and Blender has CUDA support.")
      
    # Log final device list
    devices_used = [d.name for d in cycles_prefs.devices if d.use]
    logger.info("Using the following Device(s): %s", devices_used)

  def _optimize_cpu_usage_for_gpu_rendering(self):
    """Optimize CPU usage when using GPU rendering."""
    try:
      import bpy
      
      # Reduce CPU overhead during GPU rendering
      scene = self.blender_scene
      
      # Optimize render settings
      scene.render.use_persistent_data = True  # Reduce scene rebuilding
      scene.cycles.use_animated_seed = False   # Avoid per-frame seed calculation
      
      # Optimize viewport/preview settings to reduce CPU load
      if hasattr(scene.cycles, 'preview_samples'):
        scene.cycles.preview_samples = 8  # Reduce preview quality for speed
      
      # Optimize memory settings
      scene.cycles.device = 'GPU'
      if hasattr(scene.cycles, 'use_auto_tile'):
        scene.cycles.use_auto_tile = True  # Let Blender optimize tile sizes
      
      # Reduce CPU-intensive features during rendering
      scene.cycles.use_denoising = True  # Use GPU denoising instead of CPU post-processing
      
      logger.info("Applied CPU optimization settings for GPU rendering")
      
    except Exception as e:
      logger.warning(f"Could not apply CPU optimizations: {e}")

  def set_exr_output_path(self, path_prefix: Optional[PathLike]):
    """Set the target path prefix for EXR output.

    The final filename for a frame will be "{path_prefix}{frame_nr:04d}.exr".
    If path_prefix is None then EXR output is disabled.
    """
    if path_prefix is None:
      self.exr_output_node.mute = True
    else:
      self.exr_output_node.mute = False
      self.exr_output_node.base_path = str(path_prefix)

  def save_state(self, path: PathLike, pack_textures: bool = True):
    """Saves the '.blend' blender file to disk.

    If a file with the same path exists, it is overwritten.
    """
    # first write to a temporary file, and later copy
    # (because blender cannot write to gcs buckets etc.)
    tmp_path = self.scratch_dir / "scene.blend"
    # ensure file does NOT exist (as otherwise "scene.blend1" is created instead of "scene.blend")
    kb.as_path(tmp_path).unlink(missing_ok=True)

    # --- ensure directory exists
    parent = kb.as_path(tmp_path).parent
    if not parent.exists():
      parent.mkdir(parents=True)

    # --- save the file; see https://github.com/google-research/kubric/issues/96
    with RedirectStream(stream=sys.stdout, disabled=self.verbose):
      with io.StringIO() as fstdout:  # < scratch stdout buffer
        with redirect_stdout(fstdout):  # < also suppresses python stdout
          if pack_textures:
            bpy.ops.file.pack_all()
          bpy.ops.wm.save_mainfile(filepath=str(tmp_path))
        if self.verbose:
          print(fstdout.getvalue())

    # copy to target path
    path = kb.as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    logger.info("Saving '%s'", path)
    tf.io.gfile.copy(tmp_path, path, overwrite=True)

  # def render(self,
  #            frames: Optional[Sequence[int]] = None,
  #            ignore_missing_textures: bool = False,
  #            return_layers: Sequence[str] = ("rgba", "backward_flow",
  #                                            "forward_flow", "depth",
  #                                            "normal", "object_coordinates",
  #                                            "segmentation"),
  #            ) -> Dict[str, np.ndarray]:
  #   """Renders all frames (or a subset) of the animation and returns images as a dict of arrays.

  #   Args:
  #     frames: list of frames to render (defaults to range(scene.frame_start, scene.frame_end+1)).
  #     ignore_missing_textures: if False then raise a RuntimeError when missing textures are
  #       detected. Otherwise, proceed to render (with purple color instead of missing texture).
  #     return_layers: list of layers to return. For possible values refer to
  #       the Blender.post_processors dict. Defaults to ("backward_flow",
  #       "forward_flow", "depth", "normal", "object_coordinates", "segmentation").

  #   Returns:
  #     A dictionary with one entry for each return layer. By default:
  #       - "rgba": shape = (nr_frames, height, width, 4)
  #       - "segmentation": shape = (nr_frames, height, width, 1) (int)
  #       - "backward_flow": shape = (nr_frames, height, width, 2)
  #       - "forward_flow": shape = (nr_frames, height, width, 2)
  #       - "depth": shape = (nr_frames, height, width, 1)
  #       - "object_coordinates": shape = (nr_frames, height, width, 3) (uint16)
  #       - "normal": shape = (nr_frames, height, width, 3) (uint16)
  #   """
  #   logger.info("Using scratch rendering folder: '%s'", self.scratch_dir)
  #   if not ignore_missing_textures:
  #     self._check_missing_textures()
  #   self.set_exr_output_path(self.scratch_dir / "exr" / "frame_")
  #   # --- starts rendering
  #   if frames is None:
  #     frames = range(self.scene.frame_start, self.scene.frame_end + 1)
  #   with RedirectStream(stream=sys.stdout, disabled=self.verbose):
  #     for frame_nr in frames:
  #       start_time = time.time()
  #       bpy.context.scene.frame_set(frame_nr)
  #       # When writing still images Blender doesn't append the frame number to the png path.
  #       # (but for exr it does, so we only adjust the png path)
  #       bpy.context.scene.render.filepath = str(
  #           self.scratch_dir / "images" / f"frame_{frame_nr:04d}.png")
  #       bpy.ops.render.render(animation=False, write_still=True)
  #       render_time = time.time() - start_time
  #       logger.info("Rendered frame '%s' in %.2f seconds", bpy.context.scene.render.filepath, render_time)

  #   # --- post process the rendered frames
  #   return self.postprocess(self.scratch_dir, return_layers=return_layers)

  def _check_missing_textures(self):
    missing_textures = sorted({img.filepath for img in bpy.data.images
            if tuple(img.size) == (0, 0) and img.filepath})
    if missing_textures:
      raise RuntimeError(f"Missing textures: {missing_textures}")

  def render_still(
      self,
      frame: Optional[int] = None,
      ignore_missing_textures: bool = False,
      return_layers: Sequence[str] = ("rgba", "backward_flow", "forward_flow",
                                      "depth", "normal", "object_coordinates",
                                      "segmentation"),
  ):
    """Render a single frame (first frame by default).

    Args:
    frame: Which frame to render (defaults to scene.frame_start).
    ignore_missing_textures: if False then raise a RuntimeError when missing textures are
      detected. Otherwise, proceed to render (with purple color instead of missing texture).
    return_layers: list of layers to return. For possible values refer to
      the Blender.post_processors dict. Defaults to ("backward_flow",
      "forward_flow", "depth", "normal", "object_coordinates", "segmentation").
    Returns:
    A dictionary with one entry for each return layer. By default:
        - "rgba": shape = (height, width, 4)
        - "segmentation": shape = (height, width, 1) (int)
        - "backward_flow": shape = (height, width, 2) (float32)
        - "forward_flow": shape = (height, width, 2) (float32)
        - "depth": shape = (height, width, 1) (float32)
        - "object_coordinates": shape = (height, width, 3) (uint16)
        - "normal": shape = (height, width, 3) (uint16)
    """
    frame = self.scene.frame_start if frame is None else frame

    result = self.render(frames=[frame],
                         ignore_missing_textures=ignore_missing_textures,
                         return_layers=return_layers)
    return {k: v[0] for k, v in result.items()}

  def postprocess(
      self,
      from_dir: PathLike,
      return_layers: Sequence[str]):

    from_dir = kb.as_path(from_dir)
    # --- collect all layers for all frames
    data_stack = collections.defaultdict(list)
    exr_frames = sorted((from_dir / "exr").glob("*.exr"))
    png_frames = [from_dir / "images" / (exr_filename.stem + ".png")
                  for exr_filename in exr_frames]

    for exr_filename, png_filename in zip(exr_frames, png_frames):
      source_layers = blender_utils.get_render_layers_from_exr(exr_filename)
      # Use the contrast-normalized PNG instead of the EXR for RGBA.
      source_layers["rgba"] = file_io.read_png(png_filename)

      for key in return_layers:
        post_processor = self.post_processors[key]
        data_stack[key].append(post_processor(source_layers, self.scene))

    return {key: np.stack(data_stack[key], axis=0)
            for key in data_stack}

  @staticmethod
  def clear_and_reset_blender_scene(verbose: bool = False, custom_scene: str = None):
    """ Resets Blender to an entirely empty scene (or a custom one)."""
    with RedirectStream(stream=sys.stdout, disabled=verbose):
      bpy.ops.wm.read_factory_settings(use_empty=True)
      if custom_scene is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
      else:
        logger.info("Loading scene from '%s'", custom_scene)
        bpy.ops.wm.open_mainfile(filepath=custom_scene)

  @functools.singledispatchmethod
  def add_asset(self, asset: core.Asset) -> Any:
    raise NotImplementedError(f"Cannot add {asset!r}")

  def remove_asset(self, asset: core.Asset) -> None:
    if self in asset.linked_objects:
      blender_obj = asset.linked_objects[self]
      try:
        if isinstance(blender_obj, bpy.types.Object):
          bpy.data.objects.remove(blender_obj, do_unlink=True)
        elif isinstance(blender_obj, bpy.types.Material):
          bpy.data.materials.remove(blender_obj, do_unlink=True)
        else:
          raise NotImplementedError(f"Cannot remove {asset!r}")
      except ReferenceError:
        pass  # In this case the object is already gone

  @add_asset.register(core.Cube)
  @blender_utils.prepare_blender_object
  def _add_asset(self, asset: core.Cube):
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object

    register_object3d_setters(asset, cube)
    asset.observe(AttributeSetter(cube, "active_material",
                                  converter=self._convert_to_blender_object), "material")
    asset.observe(AttributeSetter(cube, "scale"), "scale")
    asset.observe(KeyframeSetter(cube, "scale"), "scale", type="keyframe")
    return cube

  @add_asset.register(core.Sphere)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.Sphere):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5)
    bpy.ops.object.shade_smooth()
    sphere = bpy.context.active_object

    register_object3d_setters(obj, sphere)
    obj.observe(AttributeSetter(sphere, "active_material",
                                converter=self._convert_to_blender_object), "material")
    obj.observe(AttributeSetter(sphere, "scale"), "scale")
    obj.observe(KeyframeSetter(sphere, "scale"), "scale", type="keyframe")
    return sphere

  @add_asset.register(core.FileBasedObject)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.FileBasedObject):
    if obj.render_filename is None:
      return None  # if there is no render file, then ignore this object
    _, _, extension = obj.render_filename.rpartition(".")
    with RedirectStream(stream=sys.stdout, disabled=self.verbose):  # reduce the logging noise
      with io.StringIO() as fstdout:  # < scratch stdout buffer
        with redirect_stdout(fstdout):  # < also suppresses python stdout
          if extension == "obj":
            bpy.ops.import_scene.obj(filepath=obj.render_filename,
                                     use_split_objects=False,
                                     **obj.render_import_kwargs)
          elif extension in ["glb", "gltf"]:
            bpy.ops.import_scene.gltf(filepath=obj.render_filename,
                                      **obj.render_import_kwargs)

            # Apply all transforms on objects before subselecting the mesh.
            # This is optional to not break backwards compatibility.
            if obj.glb_do_transform_apply_after_import:
              bpy.ops.object.transform_apply(
                  location=True, rotation=True, scale=True
              )

            if obj.use_parenting_instead_of_join:
              parent_obj = add_top_level_empty_parent(obj.uid)
              bpy.ops.object.select_all(action="DESELECT")
              parent_obj.select_set(state=True)
            else:
              # Legacy loader which relies on JOIN. NOTE: This will destroy
              # things like animations.
              # gltf files often contain "Empty" objects as placeholders for
              # camera / lights etc.
              # here we are interested only in the meshes, we filter these out
              # and join all meshes into one.
              mesh = [
                  m for m in bpy.context.selected_objects if m.type == "MESH"
              ]
              assert mesh
              for ob in mesh:
                ob.select_set(state=True)
                bpy.context.view_layer.objects.active = ob

              # make sure one of the objects is active, otherwise join() fails.
              # see https://blender.stackexchange.com/questions/132266/joining-all-meshes-in-any-context-gets-error
              bpy.context.view_layer.objects.active = mesh[0]
              bpy.ops.object.join()

              # Make sure to delete all remaining non-mesh objects. Note that
              # for some reason deleting the non-mesh objets before joining
              # removes parts of the meshes in some cases.
              non_mesh_objects = [
                  obj
                  for obj in bpy.context.selected_objects
                  if obj.type != "MESH"
              ]
              with bpy.context.temp_override(selected_objects=non_mesh_objects):
                bpy.ops.object.delete()

            assert len(bpy.context.selected_objects) == 1
            blender_obj = bpy.context.selected_objects[0]
            blender_obj.rotation_quaternion = (0.707107, -0.707107, 0, 0)
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

          elif extension == "fbx":
            bpy.ops.import_scene.fbx(filepath=obj.render_filename,
                                     **obj.render_import_kwargs)
          elif extension in ["x3d", "wrl"]:
            bpy.ops.import_scene.x3d(filepath=obj.render_filename,
                                     **obj.render_import_kwargs)

          elif extension == "blend":
            # for now we require the paths to be encoded in the render_import_kwargs. That is:
            # - filepath = dir / "Object" / object_name
            # - directory = dir / "Object"
            # - filename = object_name

            bpy.ops.wm.append(**obj.render_import_kwargs)
          else:
            raise ValueError(f"Unknown file-type: '{extension}' for {obj}")

    assert len(bpy.context.selected_objects) == 1
    blender_obj = bpy.context.selected_objects[0]

    # deactivate auto_smooth because for some reason it lead to no smoothing at all
    # TODO: make smoothing configurable
    if hasattr(blender_obj.data, "use_auto_smooth"):
      blender_obj.data.use_auto_smooth = False

    # Calculate and set proper bounds for FileBasedObjects
    if hasattr(blender_obj.data, "vertices") and len(blender_obj.data.vertices) > 0:
      # Get the bounding box of the mesh in local coordinates
      vertices = [v.co for v in blender_obj.data.vertices]
      min_coords = np.min(vertices, axis=0)
      max_coords = np.max(vertices, axis=0)
      
      # Set the bounds property to the local bounding box
      obj.bounds = (tuple(min_coords), tuple(max_coords))
      logger.debug(f"Set bounds for {obj.uid}: {obj.bounds}")

    register_object3d_setters(obj, blender_obj)
    obj.observe(AttributeSetter(blender_obj, "active_material",
                                converter=self._convert_to_blender_object), "material")
    obj.observe(AttributeSetter(blender_obj, "scale"), "scale")
    obj.observe(KeyframeSetter(blender_obj, "scale"), "scale", type="keyframe")
    return blender_obj

  @add_asset.register(core.DirectionalLight)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.DirectionalLight):  # pylint: disable=function-redefined
    sun = bpy.data.lights.new(obj.uid, "SUN")
    sun_obj = bpy.data.objects.new(obj.uid, sun)

    register_object3d_setters(obj, sun_obj)
    obj.observe(AttributeSetter(sun, "color"), "color")
    obj.observe(KeyframeSetter(sun, "color"), "color", type="keyframe")
    obj.observe(AttributeSetter(sun, "energy"), "intensity")
    obj.observe(KeyframeSetter(sun, "energy"), "intensity", type="keyframe")
    return sun_obj

  @add_asset.register(core.SpotLight)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.SpotLight):  # pylint: disable=function-redefined
    spotlight = bpy.data.lights.new(obj.uid, "SPOT")
    spotlight_obj = bpy.data.objects.new(obj.uid, spotlight)

    register_object3d_setters(obj, spotlight_obj)
    obj.observe(AttributeSetter(spotlight, "color"), "color")
    obj.observe(KeyframeSetter(spotlight, "color"), "color", type="keyframe")
    obj.observe(AttributeSetter(spotlight, "energy"), "intensity")
    obj.observe(KeyframeSetter(spotlight, "energy"), "intensity",
                type="keyframe")
    obj.observe(AttributeSetter(spotlight, "spot_blend"), "spot_blend")
    obj.observe(KeyframeSetter(spotlight, "spot_blend"), "spot_blend",
                type="keyframe")
    obj.observe(AttributeSetter(spotlight, "spot_size"), "spot_size")
    obj.observe(KeyframeSetter(spotlight, "spot_size"), "spot_size",
                type="keyframe")
    return spotlight_obj

  @add_asset.register(core.RectAreaLight)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.RectAreaLight):
    area = bpy.data.lights.new(obj.uid, "AREA")
    area_obj = bpy.data.objects.new(obj.uid, area)

    register_object3d_setters(obj, area_obj)
    obj.observe(AttributeSetter(area, "color"), "color")
    obj.observe(KeyframeSetter(area, "color"), "color", type="keyframe")
    obj.observe(AttributeSetter(area, "energy"), "intensity")
    obj.observe(KeyframeSetter(area, "energy"), "intensity", type="keyframe")
    obj.observe(AttributeSetter(area, "size"), "width")
    obj.observe(KeyframeSetter(area, "size"), "width", type="keyframe")
    obj.observe(AttributeSetter(area, "size_y"), "height")
    obj.observe(KeyframeSetter(area, "size_y"), "height", type="keyframe")
    return area_obj

  @add_asset.register(core.PointLight)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.PointLight):
    point_light = bpy.data.lights.new(obj.uid, "POINT")
    point_light_obj = bpy.data.objects.new(obj.uid, point_light)

    register_object3d_setters(obj, point_light_obj)
    obj.observe(AttributeSetter(point_light, "color"), "color")
    obj.observe(KeyframeSetter(point_light, "color"), "color", type="keyframe")
    obj.observe(AttributeSetter(point_light, "energy"), "intensity")
    obj.observe(KeyframeSetter(point_light, "energy"), "intensity", type="keyframe")
    return point_light_obj

  @add_asset.register(core.PerspectiveCamera)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.PerspectiveCamera):
    camera = bpy.data.cameras.new(obj.uid)
    camera.type = "PERSP"
    # fix sensor width and determine sensor height by the aspect ratio of the image:
    camera.sensor_fit = "HORIZONTAL"
    camera_obj = bpy.data.objects.new(obj.uid, camera)

    register_object3d_setters(obj, camera_obj)
    obj.observe(AttributeSetter(camera, "lens"), "focal_length")
    obj.observe(KeyframeSetter(camera, "lens"), "focal_length", type="keyframe")
    obj.observe(AttributeSetter(camera, "sensor_width"), "sensor_width")
    obj.observe(KeyframeSetter(camera, "sensor_width"), "sensor_width", type="keyframe")
    obj.observe(AttributeSetter(camera, "clip_start"), "min_render_distance")
    obj.observe(KeyframeSetter(camera, "clip_start"), "min_render_distance", type="keyframe")
    obj.observe(AttributeSetter(camera, "clip_end"), "max_render_distance")
    obj.observe(KeyframeSetter(camera, "clip_end"), "max_render_distance", type="keyframe")
    obj.observe(AttributeSetter(camera, "shift_x"), "shift_x")
    obj.observe(KeyframeSetter(camera, "shift_x"), "shift_x", type="keyframe")
    obj.observe(AttributeSetter(camera, "shift_y"), "shift_y")
    obj.observe(KeyframeSetter(camera, "shift_y"), "shift_y", type="keyframe")
    return camera_obj

  @add_asset.register(core.OrthographicCamera)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.OrthographicCamera):
    camera = bpy.data.cameras.new(obj.uid)
    camera.type = "ORTHO"
    camera_obj = bpy.data.objects.new(obj.uid, camera)

    register_object3d_setters(obj, camera_obj)
    obj.observe(AttributeSetter(camera, "ortho_scale"), "orthographic_scale")
    obj.observe(KeyframeSetter(camera, "ortho_scale"), "orthographic_scale", type="keyframe")
    obj.observe(AttributeSetter(camera, "clip_start"), "min_render_distance")
    obj.observe(KeyframeSetter(camera, "clip_start"), "min_render_distance", type="keyframe")
    obj.observe(AttributeSetter(camera, "clip_end"), "max_render_distance")
    obj.observe(KeyframeSetter(camera, "clip_end"), "max_render_distance", type="keyframe")
    return camera_obj

  @add_asset.register(core.PrincipledBSDFMaterial)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.PrincipledBSDFMaterial):
    mat = bpy.data.materials.new(obj.uid)
    mat.use_nodes = True
    bsdf_node = mat.node_tree.nodes["Principled BSDF"]

    obj.observe(AttributeSetter(bsdf_node.inputs["Base Color"], "default_value"), "color")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Base Color"], "default_value"), "color",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Roughness"], "default_value"), "roughness")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Roughness"], "default_value"), "roughness",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Metallic"], "default_value"), "metallic")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Metallic"], "default_value"), "metallic",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Specular"], "default_value"), "specular")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Specular"], "default_value"), "specular",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Specular Tint"],
                                "default_value"), "specular_tint")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Specular Tint"], "default_value"), "specular_tint",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["IOR"], "default_value"), "ior")
    obj.observe(KeyframeSetter(bsdf_node.inputs["IOR"], "default_value"), "ior",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Transmission"], "default_value"), "transmission")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Transmission"], "default_value"), "transmission",
                type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Transmission Roughness"], "default_value"),
                "transmission_roughness")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Transmission Roughness"], "default_value"),
                "transmission_roughness", type="keyframe")
    obj.observe(AttributeSetter(bsdf_node.inputs["Emission"], "default_value"), "emission")
    obj.observe(KeyframeSetter(bsdf_node.inputs["Emission"], "default_value"), "emission",
                type="keyframe")
    return mat

  @add_asset.register(core.FlatMaterial)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.FlatMaterial):
    # --- Create node-based material
    mat = bpy.data.materials.new("Holdout")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.remove(tree.nodes["Principled BSDF"])  # remove the default shader

    output_node = tree.nodes["Material Output"]

    # This material is constructed from three different shaders:
    #  1. if holdout=False then emission_node is responsible for giving the object a uniform color
    #  2. if holdout=True, then the holdout_node is responsible for making the object transparent
    #  3. if indirect_visibility=False then transparent_node makes the node invisible for indirect
    #     effects such as shadows or reflections

    light_path_node = tree.nodes.new(type="ShaderNodeLightPath")
    holdout_node = tree.nodes.new(type="ShaderNodeHoldout")
    transparent_node = tree.nodes.new(type="ShaderNodeBsdfTransparent")
    holdout_mix_node = tree.nodes.new(type="ShaderNodeMixShader")
    indirect_mix_node = tree.nodes.new(type="ShaderNodeMixShader")
    overall_mix_node = tree.nodes.new(type="ShaderNodeMixShader")

    emission_node = tree.nodes.new(type="ShaderNodeEmission")

    tree.links.new(transparent_node.outputs["BSDF"], indirect_mix_node.inputs[1])
    tree.links.new(emission_node.outputs["Emission"], indirect_mix_node.inputs[2])
    tree.links.new(emission_node.outputs["Emission"], holdout_mix_node.inputs[1])
    tree.links.new(holdout_node.outputs["Holdout"], holdout_mix_node.inputs[2])
    tree.links.new(light_path_node.outputs["Is Camera Ray"], overall_mix_node.inputs["Fac"])
    tree.links.new(indirect_mix_node.outputs["Shader"], overall_mix_node.inputs[1])
    tree.links.new(holdout_mix_node.outputs["Shader"], overall_mix_node.inputs[2])
    tree.links.new(overall_mix_node.outputs["Shader"], output_node.inputs["Surface"])

    obj.observe(AttributeSetter(emission_node.inputs["Color"], "default_value"), "color")
    obj.observe(KeyframeSetter(emission_node.inputs["Color"], "default_value"), "color",
                type="keyframe")
    obj.observe(AttributeSetter(holdout_mix_node.inputs["Fac"], "default_value"), "holdout")
    obj.observe(KeyframeSetter(holdout_mix_node.inputs["Fac"], "default_value"), "holdout",
                type="keyframe")
    obj.observe(AttributeSetter(indirect_mix_node.inputs["Fac"], "default_value"),
                "indirect_visibility")
    obj.observe(KeyframeSetter(indirect_mix_node.inputs["Fac"], "default_value"),
                "indirect_visibility", type="keyframe")
    return mat

  @add_asset.register(core.SoftBody)
  @blender_utils.prepare_blender_object
  def _add_asset(self, obj: core.SoftBody):
    """Add a soft body to the Blender scene with deformation support."""
    if obj.render_filename is None:
      return None  # if there is no render file, then ignore this object
    
    _, _, extension = obj.render_filename.rpartition(".")
    with RedirectStream(stream=sys.stdout, disabled=self.verbose):
      with io.StringIO() as fstdout:
        with redirect_stdout(fstdout):
          if extension == "obj":
            bpy.ops.import_scene.obj(filepath=obj.render_filename,
                                    use_split_objects=False,
                                    **obj.render_import_kwargs)
            # Can you print Debug for number of vertices and faces?
            logger.debug(f"Soft body {obj.uid} has {len(bpy.context.selected_objects[0].data.vertices)} vertices and {len(bpy.context.selected_objects[0].data.polygons)} faces")
          elif extension in ["glb", "gltf"]:
            bpy.ops.import_scene.gltf(filepath=obj.render_filename,
                                      **obj.render_import_kwargs)
          else:
            raise ValueError(f"Unsupported file format for soft body: {extension}")

    assert len(bpy.context.selected_objects) == 1
    blender_obj = bpy.context.selected_objects[0]

    # Store original vertex positions for reference
    if hasattr(blender_obj.data, 'vertices'):
      original_positions = np.array([v.co for v in blender_obj.data.vertices])
      obj._original_vertex_positions = original_positions
      obj._vertex_count = len(original_positions)
      logger.info(f"Soft body {obj.uid} has {obj._vertex_count} vertices")
      
      # Calculate and set proper bounds for SoftBody objects
      if len(original_positions) > 0:
        min_coords = np.min(original_positions, axis=0)
        max_coords = np.max(original_positions, axis=0)
        
        # Set the bounds property to the local bounding box
        obj.bounds = (tuple(min_coords), tuple(max_coords))
        logger.debug(f"Set bounds for soft body {obj.uid}: {obj.bounds}")

    # Register standard object setters
    register_object3d_setters(obj, blender_obj)
    obj.observe(AttributeSetter(blender_obj, "active_material",
                                converter=self._convert_to_blender_object), "material")
    obj.observe(AttributeSetter(blender_obj, "scale"), "scale")
    obj.observe(KeyframeSetter(blender_obj, "scale"), "scale", type="keyframe")
    
    # Register soft body specific vertex position updates
    obj.observe(self._create_vertex_position_setter(blender_obj), "vertex_positions")
    obj.observe(self._create_vertex_position_keyframe_setter(blender_obj), 
                "vertex_positions", type="keyframe")
    
    return blender_obj

  def _setup_scene_shading(self):
    self.blender_scene.world.use_nodes = True
    tree = self.blender_scene.world.node_tree
    links = tree.links

    # clear the tree
    for node in tree.nodes.values():
      tree.nodes.remove(node)

    # create nodes
    out_node = tree.nodes.new(type="ShaderNodeOutputWorld")
    out_node.location = 1100, 0

    mix_node = tree.nodes.new(type="ShaderNodeMixShader")
    mix_node.location = 900, 0
    lightpath_node = tree.nodes.new(type="ShaderNodeLightPath")
    lightpath_node.location = 700, 350
    self.ambient_node = tree.nodes.new(type="ShaderNodeBackground")
    self.ambient_node.inputs["Color"].default_value = (0., 0., 0., 1.)
    self.ambient_node.location = 700, 0
    self.bg_node = tree.nodes.new(type="ShaderNodeBackground")
    self.bg_node.inputs["Color"].default_value = (0., 0., 0., 1.)
    self.bg_node.location = 700, -120

    links.new(lightpath_node.outputs.get("Is Camera Ray"), mix_node.inputs.get("Fac"))
    links.new(self.ambient_node.outputs.get("Background"), mix_node.inputs[1])
    links.new(self.bg_node.outputs.get("Background"), mix_node.inputs[2])
    links.new(mix_node.outputs.get("Shader"), out_node.inputs.get("Surface"))

    # create nodes for HDRI images, but leave them disconnected until
    # set_ambient_illumination or set_background
    coord_node = tree.nodes.new(type="ShaderNodeTexCoord")

    self.bg_mapping_node = tree.nodes.new(type="ShaderNodeMapping")
    self.bg_mapping_node.location = 200, 200
    self.bg_hdri_node = tree.nodes.new(type="ShaderNodeTexEnvironment")
    self.bg_hdri_node.location = 400, 200
    links.new(coord_node.outputs.get("Generated"), self.bg_mapping_node.inputs.get("Vector"))
    links.new(self.bg_mapping_node.outputs.get("Vector"), self.bg_hdri_node.inputs.get("Vector"))

    self.illum_mapping_node = tree.nodes.new(type="ShaderNodeMapping")
    self.illum_mapping_node.location = 200, -200
    self.ambient_hdri_node = tree.nodes.new(type="ShaderNodeTexEnvironment")
    self.ambient_hdri_node.location = 400, -200
    links.new(coord_node.outputs.get("Generated"), self.illum_mapping_node.inputs.get("Vector"))
    links.new(self.illum_mapping_node.outputs.get("Vector"),
              self.ambient_hdri_node.inputs.get("Vector"))

  def _set_ambient_light_color(self, color=(0., 0., 0., 1.0)):
    # disconnect incoming links from hdri node (if any)
    for link in self.ambient_node.inputs["Color"].links:
      self.blender_scene.world.node_tree.links.remove(link)
    self.ambient_node.inputs["Color"].default_value = color

  def _set_ambient_light_hdri(self, hdri_filepath=None, hdri_rotation=(0., 0., 0.), strength=1.0):
    # ensure hdri_node is connected
    self.blender_scene.world.node_tree.links.new(self.ambient_hdri_node.outputs.get("Color"),
                                                 self.ambient_node.inputs.get("Color"))
    self.ambient_hdri_node.image = bpy.data.images.load(hdri_filepath, check_existing=True)
    self.ambient_node.inputs["Strength"].default_value = strength

    self.illum_mapping_node.inputs.get("Rotation").default_value = hdri_rotation

  def _set_background_color(self, color=core.get_color("black")):
    # disconnect incoming links from hdri node (if any)
    for link in self.bg_node.inputs["Color"].links:
      self.blender_scene.world.node_tree.links.remove(link)
    # set color
    self.bg_node.inputs["Color"].default_value = color

  def _set_background_hdri(self, hdri_filepath=None, hdri_rotation=(0., 0., 0.)):
    # ensure hdri_node is connected
    self.blender_scene.world.node_tree.links.new(self.bg_hdri_node.outputs.get("Color"),
                                                 self.bg_node.inputs.get("Color"))
    self.bg_hdri_node.image = bpy.data.images.load(hdri_filepath, check_existing=True)
    self.bg_mapping_node.inputs.get("Rotation").default_value = hdri_rotation

  def _convert_to_blender_object(self, asset: core.Asset):
    return asset.linked_objects[self]
  
  def _create_vertex_position_setter(self, blender_obj):
    """Create a setter function for updating vertex positions."""
    def vertex_position_setter(change):
      new_positions = change.new
      if new_positions is not None and len(new_positions) > 0:
        self._update_mesh_vertices(blender_obj, new_positions)
    return vertex_position_setter

  def _create_vertex_position_keyframe_setter(self, blender_obj):
    """Create a keyframe setter for vertex positions."""
    def vertex_keyframe_setter(change):
      # Update vertex positions and create shape key for this frame
      new_positions = getattr(change.owner, 'current_vertex_positions', None)
      if new_positions is not None and len(new_positions) > 0:
        self._create_shape_key_for_frame(blender_obj, new_positions, change.frame)
    return vertex_keyframe_setter

  def _update_mesh_vertices(self, blender_obj, new_positions):
    """Update the mesh vertices with new positions (local coordinates)."""
    try:
      mesh = blender_obj.data
      if len(new_positions) != len(mesh.vertices):
        logger.warning(f"Vertex count mismatch: expected {len(mesh.vertices)}, got {len(new_positions)}")
        logger.warning(f"This may indicate a problem with the tri-to-tet mapping or surface mesh")
        # Try to update as many vertices as possible
        min_count = min(len(new_positions), len(mesh.vertices))
        if min_count == 0:
          return
        logger.info(f"Updating {min_count} vertices out of {len(mesh.vertices)} total")

      # CRITICAL: Do NOT reset the object's transform
      # The base position/rotation should be handled by the normal transform system
      # We only update the mesh vertex positions here (local deformation)
      
      # Enter edit mode to modify vertices
      bpy.context.view_layer.objects.active = blender_obj
      bpy.ops.object.mode_set(mode='EDIT')
      
      # Update vertex positions with local coordinates
      update_count = min(len(new_positions), len(mesh.vertices))
      for i in range(update_count):
        mesh.vertices[i].co = new_positions[i]
      
      # Update mesh and return to object mode
      bpy.ops.object.mode_set(mode='OBJECT')
      mesh.update()
      
      logger.debug(f"Successfully updated {update_count} vertices")
      
    except Exception as e:
      logger.error(f"Failed to update mesh vertices: {e}")
      try:
        bpy.ops.object.mode_set(mode='OBJECT')
      except:
        pass
      
  def _create_shape_key_for_frame(self, blender_obj, vertex_positions, frame):
    """Create a shape key for the given frame with deformed vertex positions."""
    try:
      mesh = blender_obj.data
      
      # Ensure the object has a basis shape key
      if not mesh.shape_keys:
        blender_obj.shape_key_add(name="Basis", from_mix=False)
      
      # Create shape key for this frame
      shape_key_name = f"Frame_{frame:04d}"
      shape_key = blender_obj.shape_key_add(name=shape_key_name, from_mix=False)
      
      # Update shape key vertex positions
      update_count = min(len(vertex_positions), len(shape_key.data))
      if update_count != len(shape_key.data):
        logger.warning(f"Shape key vertex count mismatch for frame {frame}: "
                      f"expected {len(shape_key.data)}, got {len(vertex_positions)}")
        logger.warning(f"Updating only {update_count} vertices")
      
      for i in range(update_count):
        shape_key.data[i].co = vertex_positions[i]
      
      # Set shape key value and keyframe it
      shape_key.value = 1.0
      shape_key.keyframe_insert(data_path="value", frame=frame)
      
      # Set value to 0 for previous and next frames to create discrete deformation
      if frame > 0:
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=frame - 1)
      
      shape_key.value = 0.0
      shape_key.keyframe_insert(data_path="value", frame=frame + 1)
      
      logger.debug(f"Created shape key {shape_key_name} for frame {frame} with {update_count} vertices")
      
    except Exception as e:
      logger.error(f"Failed to create shape key for frame {frame}: {e}")

  def _setup_soft_body_animation(self, blender_obj, obj):
    """Setup animation for soft body using vertex positions stored during simulation."""
    if not hasattr(obj, 'vertex_positions_animation'):
      logger.warning(f"No vertex animation data found for soft body {obj.uid}")
      return
      
    vertex_animation = obj.vertex_positions_animation
    frame_start = self.scene.frame_start
    
    logger.info(f"Setting up soft body animation for {obj.uid} with {len(vertex_animation)} frames")
    
    # Clear existing shape keys except basis
    mesh = blender_obj.data
    if mesh.shape_keys:
      for key in reversed(mesh.shape_keys.key_blocks[1:]):  # Skip basis
        blender_obj.shape_key_remove(key)
    
    # Create shape keys for each frame
    for frame_idx, vertex_positions in enumerate(vertex_animation):
      frame_number = frame_start + frame_idx
      self._create_shape_key_for_frame(blender_obj, vertex_positions, frame_number)

  # Add this method to be called after physics simulation
  def update_soft_body_animations(self):
    """Update all soft body animations after physics simulation is complete."""
    for asset in self.scene.assets:
      if isinstance(asset, core.SoftBody) and asset.linked_objects.get(self) is not None:
        blender_obj = asset.linked_objects[self]
        self._setup_soft_body_animation(blender_obj, asset)
        logger.info(f"Updated soft body animation for {asset.uid}")

  # Modified render method to handle soft body animations
  def render(self,
            frames: Optional[Sequence[int]] = None,
            ignore_missing_textures: bool = False,
            return_layers: Sequence[str] = ("rgba", "backward_flow",
                                            "forward_flow", "depth",
                                            "normal", "object_coordinates",
                                            "segmentation"),
            ) -> Dict[str, np.ndarray]:
    """Renders all frames with soft body support."""
    
    # Update soft body animations before rendering
    self.update_soft_body_animations()
    
    # Continue with normal rendering process
    logger.info("Using scratch rendering folder: '%s'", self.scratch_dir)
    if not ignore_missing_textures:
      self._check_missing_textures()
    self.set_exr_output_path(self.scratch_dir / "exr" / "frame_")
    
    # --- starts rendering
    if frames is None:
      frames = range(self.scene.frame_start - 1, self.scene.frame_end + 1)
    with RedirectStream(stream=sys.stdout, disabled=self.verbose):
      for frame_nr in frames:
        start_time = time.time()
        bpy.context.scene.frame_set(frame_nr)
        
        # Update soft body shapes for current frame
        self._update_soft_bodies_for_frame(frame_nr)
        
        # When writing still images Blender doesn't append the frame number to the png path.
        # (but for exr it does, so we only adjust the png path)
        bpy.context.scene.render.filepath = str(
            self.scratch_dir / "images" / f"frame_{frame_nr:04d}.png")
        bpy.ops.render.render(animation=False, write_still=True)
        render_time = time.time() - start_time
        logger.info("Rendered frame '%s' in %.2f seconds", bpy.context.scene.render.filepath, render_time)

    # --- post process the rendered frames
    return self.postprocess(self.scratch_dir, return_layers=return_layers)

  def _update_soft_bodies_for_frame(self, frame_nr):
    """Update soft body deformations for the current frame."""
    for asset in self.scene.assets:
      if isinstance(asset, core.SoftBody) and asset.linked_objects.get(self) is not None:
        blender_obj = asset.linked_objects[self]
        
        # Activate appropriate shape keys for this frame
        mesh = blender_obj.data
        if mesh.shape_keys:
          for key_block in mesh.shape_keys.key_blocks:
            if key_block.name.startswith("Frame_"):
              # Set the shape key value based on current frame
              try:
                key_frame = int(key_block.name.split("_")[1])
                key_block.value = 1.0 if key_frame == frame_nr else 0.0
              except (ValueError, IndexError):
                continue



class AttributeSetter:
  """TODO(klausg): provide high-level description of observer implementation."""

  def __init__(self, blender_obj, attribute: str, converter=None):
    self.blender_obj = blender_obj
    self.attribute = attribute
    self.converter = converter

  def __call__(self, change):
    # change = {"type": "change", "new": (1., 1., 1.), "owner": obj}
    new_value = change.new

    if isinstance(new_value, UndefinedAsset):
      return  # ignore any Undefined values

    if self.converter:
      # use converter if given
      new_value = self.converter(new_value)

    setattr(self.blender_obj, self.attribute, new_value)


class KeyframeSetter:
  def __init__(self, blender_obj, attribute_path: str):
    self.attribute_path = attribute_path
    self.blender_obj = blender_obj

  def __call__(self, change):
    self.blender_obj.keyframe_insert(self.attribute_path, frame=change.frame)


def register_object3d_setters(obj, blender_obj):
  assert isinstance(obj, core.Object3D), f"{obj!r} is not an Object3D"

  obj.observe(AttributeSetter(blender_obj, "location"), "position")
  obj.observe(KeyframeSetter(blender_obj, "location"), "position", type="keyframe")

  obj.observe(AttributeSetter(blender_obj, "rotation_quaternion"), "quaternion")
  obj.observe(KeyframeSetter(blender_obj, "rotation_quaternion"), "quaternion", type="keyframe")
