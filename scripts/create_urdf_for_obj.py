#!/usr/bin/env python3
"""
Script to automatically generate URDF files for OBJ meshes with correct origin offsets.

This script can process either:
- A single OBJ file
- All OBJ files within a directory (recursively)

For each OBJ file, the script:
1. Parses the OBJ file to find vertex positions
2. Calculates the geometric center (bounding box center)
3. Creates a URDF file with the origin set to the geometric center
4. Saves the URDF in the same directory as the OBJ file

Usage:
    # Process a single OBJ file:
    python create_urdf_for_obj.py <path_to_obj_file>
    python create_urdf_for_obj.py objs/bricks/brick_x-1-0.obj
    
    # Process all OBJ files in a directory:
    python create_urdf_for_obj.py <path_to_directory>
    python create_urdf_for_obj.py objs/bricks/
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path


def parse_obj_vertices(obj_file_path):
    """
    Parse an OBJ file and extract all vertex positions.
    
    Args:
        obj_file_path (str): Path to the OBJ file
        
    Returns:
        np.ndarray: Array of vertex positions, shape (N, 3)
    """
    vertices = []
    
    try:
        with open(obj_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):  # Vertex line
                    parts = line.split()
                    if len(parts) >= 4:  # 'v x y z' (and possibly more)
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
    except Exception as e:
        print(f"Error reading OBJ file {obj_file_path}: {e}")
        return None
    
    if not vertices:
        print(f"No vertices found in OBJ file {obj_file_path}")
        return None
        
    return np.array(vertices)


def calculate_geometric_center(vertices):
    """
    Calculate the geometric center (bounding box center) of vertices.
    
    Args:
        vertices (np.ndarray): Array of vertex positions
        
    Returns:
        tuple: (min_coords, max_coords, center)
    """
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2.0
    
    return min_coords, max_coords, center


def estimate_inertia_properties(vertices, mass=1.0):
    """
    Estimate basic inertia properties for the object.
    This is a simple approximation using bounding box dimensions.
    
    Args:
        vertices (np.ndarray): Array of vertex positions
        mass (float): Mass of the object
        
    Returns:
        dict: Inertia properties (ixx, iyy, izz, ixy, ixz, iyz)
    """
    min_coords, max_coords, _ = calculate_geometric_center(vertices)
    
    # Bounding box dimensions
    dx = max_coords[0] - min_coords[0]
    dy = max_coords[1] - min_coords[1] 
    dz = max_coords[2] - min_coords[2]
    
    # Simple box inertia approximation
    # For a box: I = (1/12) * m * (h^2 + w^2) for each axis
    ixx = (mass / 12.0) * (dy*dy + dz*dz)
    iyy = (mass / 12.0) * (dx*dx + dz*dz)
    izz = (mass / 12.0) * (dx*dx + dy*dy)
    
    # Off-diagonal terms are zero for a box aligned with axes
    ixy = 0.0
    ixz = 0.0
    iyz = 0.0
    
    return {
        'ixx': ixx, 'iyy': iyy, 'izz': izz,
        'ixy': ixy, 'ixz': ixz, 'iyz': iyz
    }


def generate_urdf_content(obj_filename, origin_offset, mass=1.0, inertia_props=None):
    """
    Generate URDF file content with the specified origin offset.
    
    Args:
        obj_filename (str): Name of the OBJ file (just filename, not full path)
        origin_offset (tuple): (x, y, z) offset for the origin
        mass (float): Mass of the object
        inertia_props (dict): Inertia properties
        
    Returns:
        str: URDF file content
    """
    if inertia_props is None:
        inertia_props = {
            'ixx': 0.01, 'iyy': 0.01, 'izz': 0.01,
            'ixy': 0.0, 'ixz': 0.0, 'iyz': 0.0
        }
    
    # Get object name from filename (without extension)
    obj_name = Path(obj_filename).stem
    
    urdf_content = f"""<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass:.3f}"/>
      <inertia
          ixx="{inertia_props['ixx']:.6f}" ixy="{inertia_props['ixy']:.6f}" ixz="{inertia_props['ixz']:.6f}"
          iyy="{inertia_props['iyy']:.6f}" iyz="{inertia_props['iyz']:.6f}"
          izz="{inertia_props['izz']:.6f}"/>
      <origin xyz="{origin_offset[0]:.6f} {origin_offset[1]:.6f} {origin_offset[2]:.6f}" rpy="0 0 0"/>
    </inertial>
  </link>
</robot>"""
    
    return urdf_content


def find_obj_files(directory_path):
    """
    Find all OBJ files in the given directory (recursively).
    
    Args:
        directory_path (str): Path to the directory to search
        
    Returns:
        list: List of paths to OBJ files
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory_path}")
        return []
    
    if not directory.is_dir():
        print(f"Error: Path is not a directory: {directory_path}")
        return []
    
    # Find all .obj files recursively
    obj_files = []
    for obj_file in directory.rglob("*.obj"):
        obj_files.append(obj_file)
    
    # Also check for .OBJ files (case insensitive)
    for obj_file in directory.rglob("*.OBJ"):
        obj_files.append(obj_file)
    
    return sorted(obj_files)


def create_urdf_for_obj(obj_file_path, mass=1.0, output_path=None):
    """
    Create a URDF file for the given OBJ file.
    
    Args:
        obj_file_path (str): Path to the OBJ file
        mass (float): Mass for the object
        output_path (str): Optional output path for URDF. If None, saves next to OBJ file.
        
    Returns:
        bool: True if successful, False otherwise
    """
    obj_path = Path(obj_file_path)
    
    if not obj_path.exists():
        print(f"Error: OBJ file does not exist: {obj_file_path}")
        return False
    
    print(f"Processing OBJ file: {obj_file_path}")
    
    # Parse vertices from OBJ file
    vertices = parse_obj_vertices(obj_file_path)
    if vertices is None:
        return False
    
    print(f"Found {len(vertices)} vertices")
    
    # Calculate geometric center
    min_coords, max_coords, center = calculate_geometric_center(vertices)
    
    print(f"Bounding box:")
    print(f"  Min: [{min_coords[0]:.6f}, {min_coords[1]:.6f}, {min_coords[2]:.6f}]")
    print(f"  Max: [{max_coords[0]:.6f}, {max_coords[1]:.6f}, {max_coords[2]:.6f}]")
    print(f"  Center: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    
    # Estimate inertia properties
    inertia_props = estimate_inertia_properties(vertices, mass)
    print(f"Estimated inertia properties:")
    print(f"  Ixx: {inertia_props['ixx']:.6f}, Iyy: {inertia_props['iyy']:.6f}, Izz: {inertia_props['izz']:.6f}")
    
    # Generate URDF content
    obj_filename = obj_path.name
    urdf_content = generate_urdf_content(obj_filename, center, mass, inertia_props)
    
    # Determine output path
    if output_path is None:
        urdf_path = obj_path.with_suffix('.urdf')
    else:
        urdf_path = Path(output_path)
    
    # Write URDF file
    try:
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        print(f"Successfully created URDF file: {urdf_path}")
        return True
    except Exception as e:
        print(f"Error writing URDF file {urdf_path}: {e}")
        return False


def process_directory(directory_path, mass=1.0):
    """
    Process all OBJ files in a directory and create URDF files for each.
    
    Args:
        directory_path (str): Path to the directory containing OBJ files
        mass (float): Mass for all objects
        
    Returns:
        tuple: (successful_count, total_count)
    """
    print(f"Processing directory: {directory_path}")
    
    # Find all OBJ files in the directory
    obj_files = find_obj_files(directory_path)
    
    if not obj_files:
        print(f"No OBJ files found in directory: {directory_path}")
        return 0, 0
    
    print(f"Found {len(obj_files)} OBJ file(s) to process:")
    for obj_file in obj_files:
        print(f"  - {obj_file}")
    print()
    
    successful_count = 0
    total_count = len(obj_files)
    
    for i, obj_file in enumerate(obj_files, 1):
        print(f"[{i}/{total_count}] Processing: {obj_file}")
        
        success = create_urdf_for_obj(str(obj_file), mass)
        if success:
            successful_count += 1
            print(f"  ✓ Successfully created URDF for {obj_file.name}")
        else:
            print(f"  ✗ Failed to create URDF for {obj_file.name}")
        print()
    
    return successful_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate URDF files for OBJ meshes with correct origin offsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single OBJ file:
  python create_urdf_for_obj.py objs/bricks/brick_x-1-0.obj
  python create_urdf_for_obj.py objs/bricks/brick_x-1-0.obj --mass 2.5
  python create_urdf_for_obj.py objs/bricks/brick_x-1-0.obj --output custom_output.urdf
  
  # Process all OBJ files in a directory:
  python create_urdf_for_obj.py objs/bricks/
  python create_urdf_for_obj.py objs/ --mass 1.5
        """
    )
    
    parser.add_argument('input_path', help='Path to the OBJ file or directory containing OBJ files')
    parser.add_argument('--mass', type=float, default=1.0, 
                       help='Mass of the object(s) (default: 1.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for URDF file (only valid for single OBJ file input)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    if input_path.is_file():
        # Process single OBJ file
        if not str(input_path).lower().endswith('.obj'):
            print(f"Error: Input file is not an OBJ file: {args.input_path}")
            sys.exit(1)
        
        print("Processing single OBJ file...")
        success = create_urdf_for_obj(args.input_path, args.mass, args.output)
        
        if success:
            print("\n✓ URDF creation completed successfully!")
            print("\nThe URDF file is now ready for use with PyBullet physics simulation.")
            print("The origin offset ensures proper physics simulation and rendering alignment.")
        else:
            print("\n✗ URDF creation failed!")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Process directory containing OBJ files
        if args.output is not None:
            print("Warning: --output argument is ignored when processing a directory")
        
        print("Processing directory...")
        successful_count, total_count = process_directory(args.input_path, args.mass)
        
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Successfully processed: {successful_count}/{total_count} OBJ files")
        
        if successful_count == total_count:
            print("✓ All URDF files created successfully!")
            print("\nAll URDF files are now ready for use with PyBullet physics simulation.")
            print("The origin offsets ensure proper physics simulation and rendering alignment.")
        elif successful_count > 0:
            print(f"⚠ Partially successful: {total_count - successful_count} files failed to process")
            sys.exit(1)
        else:
            print("✗ No URDF files were created successfully!")
            sys.exit(1)
    
    else:
        print(f"Error: Input path is neither a file nor a directory: {args.input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
