import kubric as kb
import numpy as np
import bpy
from typing import List, Tuple
import os
import subprocess
import pickle as pkl
import cv2

def obj_drop_visible_from_camera(obj, scene):
    min_corner, max_corner = get_world_object_bounds(obj) 

    drop_bounds = np.array([ # top left and bottom right of plane near camera
        [min_corner[0], max_corner[1], max_corner[2]],
        [min_corner[0], min_corner[1], 0],
    ])
    projected_points = project_points_to_image(drop_bounds, scene)
    return all(
        0 <= p[0] < scene.resolution[0] and 0 <= p[1] < scene.resolution[1]
        for p in projected_points
    )

def project_points_to_image(points, scene):
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    w, h = scene.resolution

    cam_info = kb.get_camera_info(scene.camera)
    intrinsics = cam_info["K"]
    extrinsic_matrix = np.linalg.inv(cam_info["R"])

    points_camera_h = (extrinsic_matrix @ points_h.T).T
    points_image_h = (intrinsics @ points_camera_h[:, :3].T).T

    points_image = (points_image_h[:, :2] / points_image_h[:, 2][:, np.newaxis]) * np.array([w, h])
    return points_image.astype(int)

def get_world_object_bounds(obj: kb.Object3D) -> Tuple[np.ndarray, np.ndarray]:
    """Get the world-space bounding box of an object, accounting for position, rotation, and scale."""
    # Use the object's aabbox property which properly handles all transformations
    if hasattr(obj, 'aabbox') and obj.aabbox is not None:
        aabbox = obj.aabbox
        if aabbox is not None and len(aabbox) == 2:
            return aabbox
    
    # Fallback method for objects without aabbox property or when aabbox returns None
    position = obj.position
    local_min_corner, local_max_corner = get_local_object_bounds(obj)
    min_corner = position + local_min_corner
    max_corner = position + local_max_corner
    return min_corner, max_corner

def get_local_object_bounds(obj: kb.Object3D) -> Tuple[np.ndarray, np.ndarray]:
    """Get the local bounding box of an object, accounting for scale but not rotation or position."""
    min_bound, max_bound = obj.bounds
    scale = obj.scale
    min_corner = min_bound * scale
    max_corner = max_bound * scale
    return min_corner, max_corner   

def get_object_metadata(obj: kb.Object3D) -> dict:
    metadata = obj.metadata
    metadata["asset_id"] = getattr(obj, "asset_id", None)
    metadata["position"] = obj.position.tolist()
    metadata["quaternion"] = obj.quaternion.tolist()
    metadata["scale"] = obj.scale.tolist()
    metadata["mass"] = getattr(obj, "mass", None)
    metadata["friction"] = getattr(obj, "friction", None)
    metadata["restitution"] = getattr(obj, "restitution", None)
    metadata["segmentation_id"] = getattr(obj, "segmentation_id", None)
    # print("Segmentation ID: ", obj.segmentation_id)
    if isinstance(obj, kb.SoftBody):
        metadata["use_mass_spring"] = obj.use_mass_spring
        metadata["use_neo_hookean"] = obj.use_neo_hookean
        metadata["use_bending_constraints"] = obj.use_bending_constraints
        metadata["spring_elastic_stiffness"] = obj.spring_elastic_stiffness
        metadata["spring_damping_stiffness"] = obj.spring_damping_stiffness
        metadata["spring_bending_stiffness"] = obj.spring_bending_stiffness
        metadata["neo_hookean_mu"] = obj.neo_hookean_mu
        metadata["neo_hookean_lambda"] = obj.neo_hookean_lambda
        metadata["neo_hookean_damping"] = obj.neo_hookean_damping
    else:
        metadata["use_mass_spring"] = None
        metadata["use_neo_hookean"] = None
        metadata["use_bending_constraints"] = None
        metadata["spring_elastic_stiffness"] = None
        metadata["spring_damping_stiffness"] = None
        metadata["spring_bending_stiffness"] = None
        metadata["neo_hookean_mu"] = None
        metadata["neo_hookean_lambda"] = None
        metadata["neo_hookean_damping"] = None
    return metadata

def get_camera_metadata(camera: kb.Camera) -> dict:
    return {
        "position": camera.position.tolist(),
        "quaternion": camera.quaternion.tolist(),
        "fov": camera.field_of_view,
        "focal_length": camera.focal_length,
        "intrinsics": camera.intrinsics.tolist(),
        "sensor_width": camera.sensor_width,
        "sensor_height": camera.sensor_height,
        "matrix_world": camera.matrix_world.tolist(),
    }

def sample_obj_dropping_position(
    obj: kb.Object3D, 
    placement_region: Tuple[np.ndarray, np.ndarray], 
    min_drop_height: float
) -> Tuple[float, float, float]:

    object_min_corner, object_max_corner = get_local_object_bounds(obj)
    placement_min_corner, placement_max_corner = placement_region
    placement_min_corner[2] += min_drop_height

    assert np.all(placement_max_corner - placement_min_corner >= object_max_corner - object_min_corner), \
        "Object cannot fit inside the placement region"
    
    object_h, object_w, object_d = object_max_corner - object_min_corner

    placement_min_corner += np.array([object_w/2, object_h/2, object_d/2])
    placement_max_corner -= np.array([object_w/2, object_h/2, object_d/2])

    return np.random.uniform(placement_min_corner, placement_max_corner)

def sample_position_in_region(
        obj: kb.Object3D, 
        placement_region: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[float, float, float]:
    
    object_min_corner, object_max_corner = get_local_object_bounds(obj)
    placement_min_corner, placement_max_corner = placement_region
    
    assert np.all(placement_max_corner - placement_min_corner >= object_max_corner - object_min_corner), \
        "Object cannot fit inside the placement region"
    
    object_h, object_w, object_d = object_max_corner - object_min_corner
    placement_min_corner += np.array([object_w/2, object_h/2, object_d/2])
    placement_max_corner -= np.array([object_w/2, object_h/2, object_d/2])

    return np.random.uniform(placement_min_corner, placement_max_corner)
    


def save_state(
    engine: kb.renderer.Blender,
    save_path: str,
):
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)

    engine.save_state(save_path)


def get_static_placement_bounds(
    obj: kb.Object3D,
    offset: float = 0.1,
):
    min_corner, max_corner = get_world_object_bounds(obj)
    new_min_corner = np.array([min_corner[0], min_corner[1], 0])
    new_max_corner = np.array([max_corner[0], max_corner[1], min_corner[2]])
    new_min_corner -= np.array([offset, offset, 0])
    new_max_corner += np.array([offset, offset, 0])
    return (new_min_corner, new_max_corner)

def get_object_height(obj: kb.Object3D):
    min_corner, max_corner = get_world_object_bounds(obj)
    return max_corner[2] - min_corner[2]

def get_object_width(obj: kb.Object3D):
    min_corner, max_corner = get_world_object_bounds(obj)
    return max_corner[1] - min_corner[1]

def get_object_length(obj: kb.Object3D):
    min_corner, max_corner = get_world_object_bounds(obj)
    return max_corner[0] - min_corner[0]

def get_object_height_from_center(obj: kb.Object3D):
    min_corner, _ = get_world_object_bounds(obj)
    return obj.position[2] - min_corner[2]

def get_object_img_area(obj: kb.Object3D, scene: kb.Scene):
    min_corner, max_corner = get_world_object_bounds(obj)
    projected_points = project_points_to_image(np.array([min_corner, max_corner]), scene)
    return np.prod(np.abs(projected_points[0] - projected_points[1])) / (scene.resolution[0] * scene.resolution[1])

def save_video(output_dir, vid_name, key, fps=16):
    _, ext = os.path.splitext(key)
    
    if ext == ".jpg":
      ffmpeg_cmd = (
          f"ffmpeg -y -framerate {fps} -i {os.path.join(output_dir, key)} "
          f"-pix_fmt yuv420p {os.path.join(output_dir, vid_name)}"
      )
    elif ext == ".png":
      ffmpeg_cmd = (
        f"ffmpeg -y -framerate {fps} -i {os.path.join(output_dir, key)} "
        f"-pix_fmt yuv420p {os.path.join(output_dir, vid_name)}"
      )
    subprocess.run(ffmpeg_cmd, shell=True)

def is_picklable(obj):
    """Check if an object can be pickled."""
    try:
        pkl.dumps(obj)
        return True
    except (TypeError, AttributeError, pkl.PicklingError):
        return False

def make_picklable(obj):
    """
    Recursively filter out unpicklable elements from a data structure.
    Preserves numpy arrays, lists, tuples, ints, floats, strings, and other picklable types.
    """
    # Handle basic picklable types
    if isinstance(obj, (int, float, str, bytes, bool, type(None))):
        return obj
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj
    
    # Handle lists
    if isinstance(obj, list):
        result = []
        for item in obj:
            try:
                picklable_item = make_picklable(item)
                if picklable_item is not None or item is None:
                    result.append(picklable_item)
            except Exception as e:
                print("Error making picklable: ", e)
                # Skip unpicklable items
                continue
        return result
    
    # Handle tuples
    if isinstance(obj, tuple):
        result = []
        for item in obj:
            try:
                picklable_item = make_picklable(item)
                if picklable_item is not None or item is None:
                    result.append(picklable_item)
            except Exception as e:
                print("Error making picklable: ", e)
                # Skip unpicklable items
                continue
        return tuple(result)
    
    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            try:
                # Handle key conversion - for complex objects, convert to string immediately
                if isinstance(key, (int, float, str, bytes, bool, type(None))):
                    picklable_key = key
                elif hasattr(key, 'asset_id'):
                    picklable_key = f"{key.asset_id}"
                elif hasattr(key, '__class__'):
                    picklable_key = f"object_{key.__class__.__name__}_{id(key)}"
                else:
                    picklable_key = str(key)
                
                # Try to make the value picklable
                picklable_value = make_picklable(value)
                if picklable_value is not None or value is None:
                    result[picklable_key] = picklable_value
            except Exception as e:
                print("Error making picklable: ", e)
                # Skip unpicklable key-value pairs
                continue
        return result
    
    # For other objects, try to extract picklable attributes
    if hasattr(obj, '__dict__'):
        try:
            result = {}
            for attr_name, attr_value in obj.__dict__.items():
                if not attr_name.startswith('_'):  # Skip private attributes
                    try:
                        picklable_attr = make_picklable(attr_value)
                        if picklable_attr is not None or attr_value is None:
                            result[attr_name] = picklable_attr
                    except:
                        continue
            # Add some basic info about the object
            if hasattr(obj, 'asset_id'):
                result['asset_id'] = obj.asset_id
            if hasattr(obj, '__class__'):
                result['__class_name__'] = obj.__class__.__name__
            return result
        except:
            pass
    
    # If we can't process the object, try to pickle it directly
    if is_picklable(obj):
        return obj
    
    # If nothing works, return None (will be filtered out)
    return None

def create_segmentation_color_map(segmentation_data):
    """Create a color mapping for all unique segmentation IDs in the data."""
    # Get all unique segmentation IDs from all frames
    unique_ids = set()
    
    # Handle different shapes of segmentation data
    if len(segmentation_data.shape) == 3:  # TxHxW
        for frame_idx in range(segmentation_data.shape[0]):
            frame_ids = np.unique(segmentation_data[frame_idx])
            unique_ids.update(frame_ids)
    elif len(segmentation_data.shape) == 4:  # TxHxWxC
        for frame_idx in range(segmentation_data.shape[0]):
            # Take first channel if multiple channels
            frame_data = segmentation_data[frame_idx, :, :, 0] if segmentation_data.shape[3] > 1 else segmentation_data[frame_idx, :, :, 0]
            frame_ids = np.unique(frame_data)
            unique_ids.update(frame_ids)
    else:
        raise ValueError(f"Unexpected segmentation data shape: {segmentation_data.shape}")
    
    unique_ids = sorted(list(unique_ids))
    color_map = {}
    
    # Generate distinct colors for each segmentation ID
    for i, seg_id in enumerate(unique_ids):
        if seg_id == 0:  # Background is typically black
            color_map[int(seg_id)] = [0, 0, 0]
        else:
            # Generate distinct colors using HSV color space for better separation
            hue = (i * 137.508) % 360  # Golden angle for good distribution
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.2  # Vary brightness slightly
            
            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
            color_map[int(seg_id)] = [int(r*255), int(g*255), int(b*255)]
    
    print(f"Created color map for {len(unique_ids)} segmentation IDs")
    # logging.info(f"Created color map for {len(unique_ids)} segmentation IDs")
    return color_map

def apply_segmentation_colors(segmentation_frame, color_map):
    """Apply colors to a segmentation frame based on the color map."""
    # Ensure we have a 2D array (HxW)
    if len(segmentation_frame.shape) == 3:
        # If 3D, squeeze or take first channel
        if segmentation_frame.shape[-1] == 1:
            segmentation_frame = segmentation_frame.squeeze(-1)
        else:
            segmentation_frame = segmentation_frame[:, :, 0]
    elif len(segmentation_frame.shape) == 4:
        # If 4D, take first frame and first channel
        segmentation_frame = segmentation_frame[0, :, :, 0]
    
    # Now we should have a 2D array
    if len(segmentation_frame.shape) != 2:
        raise ValueError(f"Expected 2D segmentation frame, got shape: {segmentation_frame.shape}")
    
    height, width = segmentation_frame.shape
    colored_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    for seg_id, color in color_map.items():
        mask = segmentation_frame == seg_id
        colored_frame[mask] = color
        
    return colored_frame

def create_depth_video(raw_depth, output_path, fps=30, max_depth_meters=15.0, min_depth_meters=0.1):
    """
    Create an MP4 video from normalized depth frames.
    
    Args:
        raw_depth: numpy array of shape (num_frames, height, width)
        output_path: path to save the output MP4 file
        fps: frames per second for the output video
    """

    depth_frames = normalize_depth(raw_depth, max_depth_meters, min_depth_meters)
    if len(depth_frames) == 0:
        print("Error: No frames to process")
        return False
    
    # Convert to 8-bit grayscale
    depth_8bit = (depth_frames * 255).astype(np.uint8)
    
    # Get dimensions
    num_frames, height, width = depth_8bit.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Write frames
    for frame in depth_8bit:
        out.write(frame)
    
    out.release()
    return True

def normalize_depth(depth_frames, max_depth_meters=15.0, min_depth_meters=0.01):
    """
    Normalize depth frames where close objects have value 1 and far objects have value 0.
    Uses a fixed maximum depth to ensure consistent normalization across scenes.
    
    Args:
        depth_frames: numpy array of shape (num_frames, height, width) containing depth values
        max_depth_meters: maximum depth to consider (values beyond this are clamped)
        min_depth_meters: minimum valid depth (closer values are clamped)
    
    Returns:
        normalized_frames: numpy array with values normalized to [0, 1] range using log encoding
    """
    # Find actual depth range in the data
    valid_mask = depth_frames > 0  # Assume 0 or negative values are invalid
    
    if not np.any(valid_mask):
        print("Warning: No valid depth values found")
        return np.zeros_like(depth_frames)
    
    actual_min_depth = np.min(depth_frames[valid_mask])
    actual_max_depth = np.max(depth_frames[valid_mask])
    
    print(f"Actual depth range: {actual_min_depth:.3f} to {actual_max_depth:.3f} meters")
    
    # Clamp depth values to the specified range
    depth_clamped = np.copy(depth_frames)
    depth_clamped = np.maximum(depth_clamped, min_depth_meters)
    depth_clamped = np.minimum(depth_clamped, max_depth_meters)
    max_depth_value = np.max(depth_clamped)
    min_depth_value = np.min(depth_clamped)
    print(f"Using normalized range: {min_depth_value:.3f} to {max_depth_value:.3f} meters")
    
    # Normalize depth values to [0, 1]
    normalized = (max_depth_value - depth_clamped) / (max_depth_value - min_depth_value)
    
    # Set invalid pixels to 0 (black)
    normalized[~valid_mask] = 0
    
    # Ensure values are in [0, 1] range
    normalized = np.clip(normalized, 0, 1)
    
    return normalized
