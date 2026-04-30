# Soft Body Simulation in Kubric

This guide explains how to use the new soft body simulation capabilities in Kubric with PyBullet.

## Overview

Soft bodies are deformable objects that can bend, stretch, and compress during physics simulation. Unlike rigid bodies, soft bodies have internal structure (vertices, edges, faces) that can deform under forces.

## Requirements

1. **Mesh Files**: Soft bodies require tetrahedral mesh files in `.vtk` format for PyBullet simulation
2. **Surface Meshes**: For rendering, you'll need surface mesh files (`.obj`, `.gltf`, etc.)

## Creating Soft Bodies

### Basic Soft Body

```python
import kubric as kb

# Create a soft body object
soft_body = kb.SoftBody(
    asset_id="soft_cube",
    simulation_filename="path/to/tetrahedral_mesh.vtk",  # Required for simulation
    render_filename="path/to/surface_mesh.obj",         # Required for rendering
    position=(0, 0, 2),
    mass=1.0
)

# Add to scene
scene += soft_body
```

### Configuring Soft Body Parameters

```python
soft_body = kb.SoftBody(
    asset_id="soft_cube",
    simulation_filename="cube.vtk",
    render_filename="cube.obj",
    position=(0, 0, 2),
    mass=1.0,
    
    # Spring stiffness parameters (0.0 to 1.0)
    spring_elastic_stiffness=0.7,    # How stiff the springs are
    spring_damping_stiffness=0.1,    # How much energy is lost in oscillations
    spring_bending_stiffness=0.5,    # Resistance to bending
    
    # Collision settings
    collision_margin=0.02,           # Safety margin for collision detection
    self_collision=False,            # Whether object can collide with itself
    
    # Material model settings
    use_mass_spring=True,            # Use mass-spring model
    use_bending_constraints=True,    # Add bending resistance
    use_neo_hookean=False,           # Use Neo-Hookean material model
    
    # Neo-Hookean parameters (when use_neo_hookean=True)
    neo_hookean_mu=150.0,           # Material stiffness parameter
    neo_hookean_lambda=200.0,       # Material bulk modulus parameter
    neo_hookean_damping=0.01,       # Neo-Hookean damping
    
    # Performance optimization
    cluster_count=16                 # Number of collision clusters
)
```

## Material Models

### Mass-Spring Model
- **Best for**: General deformable objects, cloth-like materials
- **Parameters**: `spring_elastic_stiffness`, `spring_damping_stiffness`, `spring_bending_stiffness`
- **Pros**: Fast, intuitive parameters
- **Cons**: Less physically accurate

```python
soft_cloth = kb.SoftBody(
    asset_id="cloth",
    simulation_filename="cloth.vtk",
    use_mass_spring=True,
    spring_elastic_stiffness=0.9,    # High stiffness for cloth
    spring_damping_stiffness=0.05,   # Low damping for flowing motion
    spring_bending_stiffness=0.3,    # Medium bending resistance
)
```

### Neo-Hookean Model
- **Best for**: Rubber-like, elastic materials
- **Parameters**: `neo_hookean_mu`, `neo_hookean_lambda`, `neo_hookean_damping`
- **Pros**: Physically accurate, good for elastic materials
- **Cons**: More computationally expensive

```python
soft_rubber = kb.SoftBody(
    asset_id="rubber_ball",
    simulation_filename="ball.vtk",
    use_neo_hookean=True,
    neo_hookean_mu=100.0,           # Lower values = softer
    neo_hookean_lambda=100.0,       # Higher values = less compressible
    neo_hookean_damping=0.01,       # Energy dissipation
)
```

## Advanced Operations

### Applying Forces to Soft Body Nodes

```python
# During simulation, apply force to specific nodes
simulator = kb.simulator.PyBullet(scene)

# Get the soft body index
soft_body_idx = soft_body.linked_objects[simulator]

# Apply force to node 0
simulator.apply_soft_body_force(
    obj_idx=soft_body_idx,
    node_index=0,
    force=(0, 0, 10)  # Upward force
)
```

### Creating Anchors

Anchors fix soft body nodes to positions or rigid bodies:

```python
# Anchor to world position
anchor_id = simulator.create_soft_body_anchor(
    soft_body_idx=soft_body_idx,
    node_index=5,
    anchor_position=(0, 0, 5)  # Fix node 5 to this world position
)

# Anchor to rigid body
rigid_body_idx = rigid_cube.linked_objects[simulator]
anchor_id = simulator.create_soft_body_anchor(
    soft_body_idx=soft_body_idx,
    node_index=10,
    rigid_body_idx=rigid_body_idx  # Attach node 10 to rigid body
)

# Remove anchor later
simulator.remove_soft_body_anchor(anchor_id)
```

### Getting Soft Body State

```python
# Get current node positions and velocities
soft_body_data = simulator.get_soft_body_data(soft_body_idx)
node_positions = soft_body_data["node_positions"]
node_velocities = soft_body_data["node_velocities"]
```

## Mesh Preparation

### Converting OBJ to VTK

To use soft bodies, you need tetrahedral meshes. Here's how to create them:

#### Method 1: Using TetWild
```bash
# Install TetWild
git clone https://github.com/Yixin-Hu/TetWild
cd TetWild && mkdir build && cd build
cmake .. && make

# Convert OBJ to STL in Blender
# Then use TetWild to create tetrahedral mesh
./TetWild input.stl

# Convert MSH to VTK using GMSH
gmsh input_.msh -save_all -format vtk
```

#### Method 2: Using GMSH
```python
import gmsh

gmsh.initialize()
gmsh.open("input.obj")
gmsh.model.mesh.generate(3)  # Generate 3D tetrahedral mesh
gmsh.write("output.vtk")
gmsh.finalize()
```

## Complete Example

```python
import tempfile
import kubric as kb

# Create scene
scene = kb.Scene(resolution=(512, 512))
scene.frame_start = 1
scene.frame_end = 200

# Create simulator and renderer
simulator = kb.simulator.PyBullet(scene)
renderer = kb.renderer.Blender(scene)

# Create ground
ground = kb.Cube(scale=(10, 10, 0.1), position=(0, 0, -0.5), static=True)
ground.material = kb.PrincipledBSDFMaterial(color=(0.7, 0.7, 0.7, 1.0))
scene += ground

# Create soft body
soft_ball = kb.SoftBody(
    asset_id="bouncy_ball",
    simulation_filename="ball.vtk",
    render_filename="ball.obj",
    position=(0, 0, 3),
    mass=1.0,
    spring_elastic_stiffness=0.8,
    spring_damping_stiffness=0.1,
    collision_margin=0.02,
    use_bending_constraints=True
)
soft_ball.material = kb.PrincipledBSDFMaterial(color=(1.0, 0.2, 0.2, 1.0))
scene += soft_ball

# Set up lighting and camera
scene += kb.DirectionalLight(position=(2, 2, 3), look_at=(0, 0, 0))
scene.camera = kb.PerspectiveCamera(position=(5, -5, 3), look_at=(0, 0, 0))

# Add to scene
scene += simulator
scene += renderer

# Run simulation
animation, collisions, soft_body_animation = simulator.run()

# Render
renderer.render()
```

## Tips and Best Practices

1. **Mesh Quality**: Use high-quality tetrahedral meshes for better simulation stability
2. **Parameter Tuning**: Start with default parameters and adjust gradually
3. **Performance**: Use `cluster_count` to optimize collision detection for complex meshes
4. **Stability**: Lower time steps improve stability but slow simulation
5. **Debugging**: Enable PyBullet GUI mode during development to visualize simulation

## Common Issues

1. **Mesh Loading Fails**: Ensure `.vtk` file contains tetrahedral elements
2. **Unstable Simulation**: Reduce time step or adjust stiffness parameters
3. **Poor Performance**: Reduce mesh complexity or increase `cluster_count`
4. **Unexpected Behavior**: Check that mesh is properly centered and scaled

## Parameter Reference

| Parameter | Range | Description |
|-----------|-------|-------------|
| `spring_elastic_stiffness` | 0.0-1.0 | How stiff the material is |
| `spring_damping_stiffness` | 0.0-1.0 | Energy dissipation in oscillations |
| `spring_bending_stiffness` | 0.0-1.0 | Resistance to bending deformation |
| `collision_margin` | >0.0 | Safety margin for collision detection |
| `neo_hookean_mu` | >0.0 | Material shear modulus (Neo-Hookean) |
| `neo_hookean_lambda` | >0.0 | Material bulk modulus (Neo-Hookean) |
| `cluster_count` | >0 | Number of collision clusters for optimization | 