#!/usr/bin/env python3
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

"""
Fixed example demonstrating soft body simulation using PyBullet in Kubric.
"""

import os
import tempfile
import pathlib
import numpy as np

import sys; sys.path = ["kubric"] + sys.path
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import logging
from skimage import io

# Import pybullet for debugging mesh data
from kubric.redirect_io import RedirectStream
with RedirectStream(stream=sys.stderr):
  import pybullet as pb

logging.basicConfig(level="DEBUG")

# Debug function to inspect mesh properties
def debug_mesh_properties(mesh_file, mesh_type="Unknown"):
    """Debug function to inspect mesh properties"""
    if not mesh_file:
        print(f"\n=== {mesh_type} MESH DEBUG: No file specified ===")
        return
        
    path = pathlib.Path(mesh_file)
    print(f"\n=== {mesh_type} MESH DEBUG: {path.name} ===")
    print(f"File exists: {path.exists()}")
    print(f"File size: {path.stat().st_size if path.exists() else 'N/A'} bytes")
    
    if not path.exists():
        return
    
    if path.suffix == ".obj":
        # Quick OBJ file inspection
        vertices = []
        faces = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except ValueError:
                            pass
                elif line.startswith('f '):
                    faces.append(line)
        
        print(f"OBJ vertices: {len(vertices)}")
        print(f"OBJ faces: {len(faces)}")
        
        if vertices:
            vertices = np.array(vertices)
            print(f"OBJ bounds: {np.min(vertices, axis=0)} to {np.max(vertices, axis=0)}")
            print(f"OBJ center: {np.mean(vertices, axis=0)}")
            print(f"OBJ scale: {np.max(vertices, axis=0) - np.min(vertices, axis=0)}")
    
    elif path.suffix == ".vtk":
        print(f"VTK file detected")
        # Try to read basic VTK info
        try:
            with open(path, 'r') as f:
                lines = f.readlines()[:20]  # First 20 lines
                print("First few lines of VTK file:")
                for i, line in enumerate(lines[:5]):
                    print(f"  {i+1}: {line.strip()}")
                
                # Look for POINTS line
                for line in lines:
                    if 'POINTS' in line:
                        print(f"Found: {line.strip()}")
                        break
        except Exception as e:
            print(f"Error reading VTK file: {e}")
    
    elif path.suffix == ".txt":
        print("Text file detected (likely mapping)")
        try:
            with open(path, 'r') as f:
                lines = [line.strip() for line in f.readlines() 
                        if not line.strip().startswith('#') and line.strip()]
                print(f"Non-comment lines: {len(lines)}")
                if lines:
                    print("First 5 lines:")
                    for i, line in enumerate(lines[:5]):
                        print(f"  {i+1}: {line}")
        except Exception as e:
            print(f"Error reading mapping file: {e}")

# Create scene
scene = kb.Scene(resolution=(256, 256))
scene.frame_start = 1
scene.frame_end = 25

# Create renderer
renderer = Blender(scene, scratch_dir=tempfile.mkdtemp(), samples_per_pixel=16)

# Create simulator with proper deformable world setup
simulator = PyBullet(scene, scratch_dir=tempfile.mkdtemp())

# IMPORTANT: Reset to deformable world BEFORE adding any objects
print("Resetting simulation to deformable world...")
simulator.resetSimulationToDeformableWorld()

# Set additional parameters for soft body simulation
simulator._physics_client.setGravity(0, 0, -10)
simulator._physics_client.setRealTimeSimulation(0)

# Create a ground plane (rigid body)
ground = kb.Cube(name="ground", scale=(5, 5, 0.1), position=(0, 0, -0.1))
ground.material = kb.PrincipledBSDFMaterial(color=(0.8, 0.8, 0.8, 1.0))
ground.static = True
scene += ground

# Test with a simple approach - create a regular rigid body first
print("Creating rigid body for comparison...")
rigid_cube = kb.Cube(name="rigid_cube", position=(-1, 0, 2), scale=(0.3, 0.3, 0.3))
rigid_cube.material = kb.PrincipledBSDFMaterial(color=(0.2, 0.2, 1.0, 1.0))
rigid_cube.mass = 1.0
scene += rigid_cube

# Try with minimal parameters first
test_soft_body = kb.SoftBody(
    asset_id="test_soft_body",
    simulation_filename="objs/tetgen/ball_volume.vtk",
    render_filename="objs/tetgen/ball_surface.obj", 
    tri_to_tet_mapping_filename="objs/tetgen/ball_vertex_mapping.txt",
    position=(2, -1, 1),
    scale=(0.6, 0.6, 0.6),
    mass=10.0,  # Lighter mass
    spring_elastic_stiffness=0.1,
    spring_damping_stiffness=0.1,
    collision_margin=0.1  # Smaller margin
)

print(f"\n=== PRE-SIMULATION DEBUG ===")
print(f"Soft body initial position: {test_soft_body.position}")
print(f"Soft body scale: {test_soft_body.scale}")

print("✓ SoftBody object creation successful!")

# =================== DEBUGGING SECTION 1: FILE INSPECTION ===================
print("\n" + "="*60)
print("DEBUGGING SECTION 1: MESH FILE INSPECTION")
print("="*60)

debug_mesh_properties(test_soft_body.simulation_filename, "SIMULATION (VTK)")
debug_mesh_properties(test_soft_body.render_filename, "RENDERING (OBJ)")
debug_mesh_properties(test_soft_body.tri_to_tet_mapping_filename, "MAPPING")

print(f"\nSoft body properties:")
print(f"  Position: {test_soft_body.position}")
print(f"  Scale: {test_soft_body.scale}")
print(f"  Mass: {test_soft_body.mass}")

# Add to scene (this is where PyBullet loading happens)
print("\nAdding SoftBody to scene...")
scene.add(test_soft_body)
print("✓ SoftBody added to scene successfully!")

# Add this after adding to scene
if hasattr(test_soft_body, 'linked_objects') and simulator in test_soft_body.linked_objects:
    sb_idx = test_soft_body.linked_objects[simulator]
    pos, orn = simulator._physics_client.getBasePositionAndOrientation(sb_idx)
    print(f"Actual physics position after loading: {pos}")
    print(f"Position difference: {np.array(pos) - np.array(test_soft_body.position)}")
    
# =================== DEBUGGING SECTION 2: MAPPING VERIFICATION ===================
print("\n" + "="*60)
print("DEBUGGING SECTION 2: MAPPING VERIFICATION")
print("="*60)

# Check tri-to-tet mapping file format
if hasattr(test_soft_body, 'tri_to_tet_mapping_filename'):
    mapping_file = test_soft_body.tri_to_tet_mapping_filename
    print(f"Mapping file: {mapping_file}")
    
    if pathlib.Path(mapping_file).exists():
        # Read and display first few lines
        with open(mapping_file, 'r') as f:
            lines = f.readlines()[:10]  # First 10 lines
            print("First 10 lines of mapping file:")
            for i, line in enumerate(lines):
                print(f"  {i+1}: {line.strip()}")
        
        # Count total entries
        with open(mapping_file, 'r') as f:
            all_lines = [line.strip() for line in f.readlines() 
                        if not line.strip().startswith('#') and line.strip()]
            print(f"Total mapping entries: {len(all_lines)}")
    else:
        print("WARNING: Mapping file does not exist!")

# Check the loaded mapping
if hasattr(test_soft_body, 'tri_to_tet_mapping') and test_soft_body.tri_to_tet_mapping:
    mapping = test_soft_body.tri_to_tet_mapping
    print(f"Loaded mapping entries: {len(mapping)}")
    
    if len(mapping) > 0:
        mapping_array = np.array(mapping)
        print(f"Mapping shape: {mapping_array.shape}")
        print(f"Surface vertex range: {np.min(mapping_array[:, 0])} to {np.max(mapping_array[:, 0])}")
        print(f"Tet vertex range: {np.min(mapping_array[:, 1])} to {np.max(mapping_array[:, 1])}")
        
        # Show first few mappings
        print("First 5 mappings [surface_id, tet_id]:")
        for i in range(min(5, len(mapping))):
            print(f"  {mapping[i]}")
else:
    print("WARNING: No tri-to-tet mapping loaded!")

# Set up lighting
scene += kb.DirectionalLight(name="sun", position=(1, 2, 3), look_at=(0, 0, 0), intensity=1.5)

# Set up camera
scene.camera = kb.PerspectiveCamera(position=(3, -3, 2), look_at=(0, 0, 0), focal_length=18.0)

print("\nRunning simulation...")

# Run the simulation
animation, collisions = simulator.run()

print(f"Simulation complete. Recorded {len(animation)} object animations.")
print(f"Found {len(collisions)} collision events.")

# =================== DEBUGGING SECTION 3: POST-SIMULATION ANALYSIS ===================
print("\n" + "="*60)
print("DEBUGGING SECTION 3: POST-SIMULATION ANALYSIS")
print("="*60)

# Check mesh vertex counts from physics simulation
print("=== PHYSICS MESH INFO ===")
simulator_physics_client = simulator._physics_client

# Get number of bodies in simulation
num_bodies = simulator_physics_client.getNumBodies()
print(f"Total bodies in simulation: {num_bodies}")

# Find the soft body
soft_body_idx = None
if hasattr(test_soft_body, 'linked_objects') and simulator in test_soft_body.linked_objects:
    soft_body_idx = test_soft_body.linked_objects[simulator]
    print(f"Soft body physics index: {soft_body_idx}")
    
    # Try to get mesh data
    try:
        kwargs = {}
        if hasattr(pb, "MESH_DATA_SIMULATION_MESH"):
            kwargs["flags"] = pb.MESH_DATA_SIMULATION_MESH
        
        num_vertices, mesh_data = simulator_physics_client.getMeshData(soft_body_idx, **kwargs)
        print(f"Physics mesh vertices: {num_vertices}")
        
        if num_vertices > 0 and mesh_data:
            positions = np.array(mesh_data).reshape((-1, 3))
            print(f"Physics mesh position range:")
            print(f"  X: {np.min(positions[:, 0]):.3f} to {np.max(positions[:, 0]):.3f}")
            print(f"  Y: {np.min(positions[:, 1]):.3f} to {np.max(positions[:, 1]):.3f}")
            print(f"  Z: {np.min(positions[:, 2]):.3f} to {np.max(positions[:, 2]):.3f}")
            print(f"  Center: [{np.mean(positions[:, 0]):.3f}, {np.mean(positions[:, 1]):.3f}, {np.mean(positions[:, 2]):.3f}]")
        
    except Exception as e:
        print(f"Error getting physics mesh data: {e}")
else:
    print("WARNING: Could not find soft body in physics simulation!")

# Check rendering mesh vertex count
print("\n=== RENDERING MESH INFO ===")
if hasattr(test_soft_body, '_vertex_count'):
    print(f"Rendering mesh vertices: {test_soft_body._vertex_count}")
else:
    print("WARNING: No rendering vertex count stored!")

if hasattr(test_soft_body, '_original_vertex_positions'):
    orig_pos = test_soft_body._original_vertex_positions
    print(f"Original rendering positions:")
    print(f"  X: {np.min(orig_pos[:, 0]):.3f} to {np.max(orig_pos[:, 0]):.3f}")
    print(f"  Y: {np.min(orig_pos[:, 1]):.3f} to {np.max(orig_pos[:, 1]):.3f}")
    print(f"  Z: {np.min(orig_pos[:, 2]):.3f} to {np.max(orig_pos[:, 2]):.3f}")
    print(f"  Center: [{np.mean(orig_pos[:, 0]):.3f}, {np.mean(orig_pos[:, 1]):.3f}, {np.mean(orig_pos[:, 2]):.3f}]")

# Verify the mapping makes sense
if (hasattr(test_soft_body, 'tri_to_tet_mapping') and test_soft_body.tri_to_tet_mapping and 
    hasattr(test_soft_body, '_vertex_count')):
    mapping_count = len(test_soft_body.tri_to_tet_mapping)
    surface_count = test_soft_body._vertex_count
    
    print(f"\n=== MAPPING CONSISTENCY CHECK ===")
    print(f"Surface mesh vertices: {surface_count}")
    print(f"Mapping entries: {mapping_count}")
    
    if mapping_count == surface_count:
        print("✓ Mapping count matches surface vertex count")
    else:
        print(f"✗ MISMATCH: Mapping has {mapping_count} entries but surface has {surface_count} vertices")
        
    # Check if surface IDs in mapping correspond to vertex indices
    if test_soft_body.tri_to_tet_mapping:
        mapping_array = np.array(test_soft_body.tri_to_tet_mapping)
        surface_ids = mapping_array[:, 0]
        expected_ids = np.arange(surface_count)
        
        if np.array_equal(np.sort(surface_ids), expected_ids):
            print("✓ Surface IDs in mapping are consecutive 0-based indices")
        else:
            print("✗ Surface IDs in mapping are not consecutive 0-based indices")
            print(f"  Expected: 0 to {surface_count-1}")
            print(f"  Found: {np.min(surface_ids)} to {np.max(surface_ids)}")
            print(f"  Unique IDs: {len(np.unique(surface_ids))}")

# Check vertex positions animation data
print(f"\n=== VERTEX ANIMATION DATA ===")
if hasattr(test_soft_body, 'vertex_positions_animation'):
    print(f"Vertex animation frames: {len(test_soft_body.vertex_positions_animation)}")
    if len(test_soft_body.vertex_positions_animation) > 0:
        first_frame = test_soft_body.vertex_positions_animation[0]
        last_frame = test_soft_body.vertex_positions_animation[-1]
        print(f"First frame vertex count: {len(first_frame)}")
        print(f"Last frame vertex count: {len(last_frame)}")
        
        if len(first_frame) > 0:
            positions = np.array(first_frame)
            print(f"First frame vertex range: {np.min(positions, axis=0)} to {np.max(positions, axis=0)}")
            print(f"First frame vertex center: {np.mean(positions, axis=0)}")
            
        if len(last_frame) > 0:
            positions = np.array(last_frame)
            print(f"Last frame vertex range: {np.min(positions, axis=0)} to {np.max(positions, axis=0)}")
            print(f"Last frame vertex center: {np.mean(positions, axis=0)}")
            
            # Check if the object moved significantly
            if len(first_frame) > 0 and len(last_frame) > 0:
                first_center = np.mean(np.array(first_frame), axis=0)
                last_center = np.mean(np.array(last_frame), axis=0)
                movement = last_center - first_center
                print(f"Object movement: {movement}")
                print(f"Total movement distance: {np.linalg.norm(movement):.3f}")
else:
    print("WARNING: No vertex animation data found!")

print("\n" + "="*60)

# Render the animation
print("Rendering frames...")
data_stack = renderer.render(return_layers=["rgba","segmentation"])

output_dir = "./sim_output/00000"
os.makedirs(output_dir, exist_ok=True)

for i in range(data_stack['rgba'].shape[0]):
    img_path = os.path.join(output_dir, f"rgba_{i:05d}.jpg")
    io.imsave(img_path, 
            data_stack["rgba"][i][..., :3])

print("Example complete!")
print("\nNOTE: If soft body loading failed, this is expected with .obj files.")
print("To use soft bodies properly:")
print("1. Convert your mesh to .vtk format using TetWild, GMSH, or similar tools")
print("2. Use tetrahedral meshes instead of surface meshes")
print("3. Ensure the mesh is manifold and well-formed")

print("\n" + "="*60)
print("DEBUG OUTPUT COMPLETE - Check the console output above for issues!")
print("="*60)