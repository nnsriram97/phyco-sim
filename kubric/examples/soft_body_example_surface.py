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

import sys; sys.path = ["kubric"] + sys.path
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import logging
from skimage import io

logging.basicConfig(level="INFO")

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
    simulation_filename="objs/tetgen/ball_surface.obj",
    render_filename="objs/tetgen/ball_surface.obj", 
    position=(2, -1, 1),
    scale=(0.6, 0.6, 0.6),
    mass=1.0,  # Lighter mass
    spring_elastic_stiffness=0.7,
    spring_damping_stiffness=0.7,
    collision_margin=0.1  # Smaller margin
)

print("✓ SoftBody object creation successful!")

# Add to scene (this is where PyBullet loading happens)
print("Adding SoftBody to scene...")
scene.add(test_soft_body)
print("✓ SoftBody added to scene successfully!")

# Set up lighting
scene += kb.DirectionalLight(name="sun", position=(1, 2, 3), look_at=(0, 0, 0), intensity=1.5)

# Set up camera
scene.camera = kb.PerspectiveCamera(position=(3, -3, 2), look_at=(0, 0, 0), focal_length=18.0)

print("Running simulation...")

# Run the simulation
animation, collisions = simulator.run()

print(f"Simulation complete. Recorded {len(animation)} object animations.")
print(f"Found {len(collisions)} collision events.")

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