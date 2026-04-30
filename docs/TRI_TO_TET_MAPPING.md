# Tri-to-Tet Mapping for Soft Body Simulation

This document explains the tri-to-tet mapping implementation that allows using tetrahedral meshes for physics simulation while rendering surface meshes in Blender.

## Overview

When simulating soft bodies, we often need two different mesh representations:

1. **Tetrahedral Mesh (.vtk)**: Used by PyBullet for physics simulation. Contains volume elements (tetrahedra) that allow realistic deformation.
2. **Surface Mesh (.obj/.gltf)**: Used by Blender for rendering. Contains only surface triangles for efficient rendering.

The **tri-to-tet mapping** connects these two representations by mapping each surface vertex to its corresponding tetrahedral vertex.

## How It Works

### 1. File Structure

```
your_object/
├── ball_volume.vtk              # Tetrahedral mesh for simulation
├── ball_surface.obj             # Surface mesh for rendering
└── ball_vertex_mapping.txt      # Mapping file
```

### 2. Mapping File Format

The mapping file contains lines mapping surface vertex IDs to tetrahedral vertex IDs:

```
# Surface vertex ID -> Volume vertex ID mapping
# Format: surface_id volume_id
0 0
1 21
2 19
3 288
4 30
...
```

Each line means: "Surface vertex N corresponds to tetrahedral vertex M"

### 3. Simulation Flow

```
PyBullet Simulation:
   ball_volume.vtk (1000 vertices) → [deformed positions]
                     ↓
Tri-to-Tet Mapping:
   Extract surface positions using mapping file
                     ↓
Blender Rendering:
   ball_surface.obj (400 vertices) → [rendered with deformation]
```

## Implementation Details

### In PyBullet Simulator (`pybullet.py`)

1. **Loading the mapping**:
   ```python
   if path.suffix == ".vtk":
     tri_to_tet_mapping_path = pathlib.Path(obj.tri_to_tet_mapping_filename).resolve()
     if tri_to_tet_mapping_path.exists():
       with open(tri_to_tet_mapping_path, "r") as f:
         tri_to_tet_mapping = [list(map(int, line.split())) 
                              for line in f.readlines() 
                              if not line.startswith('#')]
       obj.tri_to_tet_mapping = tri_to_tet_mapping
   ```

2. **Extracting surface positions**:
   ```python
   def get_surface_positions_from_tet(self, body_id: int, asset: core.SoftBody):
     tet_data = self.get_soft_body_data(body_id)
     
     if asset.tri_to_tet_mapping is None:
       return tet_data  # Fallback to all positions
     
     surface_positions = []
     for surface_id, tet_id in asset.tri_to_tet_mapping:
       if tet_id < len(tet_data["positions"]):
         surface_positions.append(tet_data["positions"][tet_id])
     
     return {"positions": np.array(surface_positions), 
             "num_vertices": len(surface_positions)}
   ```

3. **During simulation**:
   ```python
   # Use surface positions if tri-to-tet mapping is available
   soft_body_data = self.get_surface_positions_from_tet(soft_body_idx, asset)
   ```

### In Blender Renderer (`blender.py`)

The renderer receives only the surface vertex positions and updates the surface mesh:

```python
def _update_mesh_vertices(self, blender_obj, new_positions):
  mesh = blender_obj.data
  for i in range(min(len(new_positions), len(mesh.vertices))):
    mesh.vertices[i].co = new_positions[i]
  mesh.update()
```

## Usage Example

```python
import kubric as kb

# Create soft body with tri-to-tet mapping
soft_ball = kb.SoftBody(
    asset_id="soft_ball",
    simulation_filename="ball_volume.vtk",           # Tetrahedral mesh
    render_filename="ball_surface.obj",             # Surface mesh
    tri_to_tet_mapping_filename="ball_vertex_mapping.txt",  # Mapping file
    position=(0, 0, 3),
    mass=1.0,
    spring_elastic_stiffness=0.8
)

scene += soft_ball

# Run simulation - mapping happens automatically
animation, collisions = simulator.run()

# Render - only surface vertices are deformed
frames = renderer.render()
```

## Generating Mapping Files

You can generate the required files using the `scripts/tetgen_processor.py` script:

```bash
python scripts/tetgen_processor.py input.obj --output-dir objs/tetgen/
```

This will create:
- `input_volume.vtk`: Tetrahedral mesh
- `input_surface.obj`: Surface mesh  
- `input_vertex_mapping.txt`: Mapping file

## Benefits

1. **Accurate Physics**: Tetrahedral meshes provide realistic deformation
2. **Efficient Rendering**: Surface meshes render faster than volume meshes
3. **Automatic Mapping**: No manual intervention needed during simulation
4. **Backward Compatibility**: Works with existing soft body code

## Validation

The system includes several validation checks:

- **File existence**: Warns if mapping file doesn't exist
- **Vertex count validation**: Handles mismatches gracefully
- **Range checking**: Ensures tet vertex IDs are valid
- **Fallback behavior**: Uses all vertices if no mapping available

## Error Handling

Common issues and solutions:

1. **Vertex count mismatch**: 
   - Check that surface mesh matches the mapping file
   - Regenerate mapping file if needed

2. **Out of range tet IDs**:
   - Verify tetrahedral mesh has enough vertices
   - Check mapping file for invalid indices

3. **Missing mapping file**:
   - System falls back to using all tetrahedral vertices
   - Warning logged but simulation continues

## Performance Notes

- Mapping lookup is O(n) where n = number of surface vertices
- Memory usage reduced by storing only surface positions
- Rendering performance improved with smaller vertex count
- Physics accuracy maintained with full tetrahedral mesh

This implementation provides a seamless bridge between accurate physics simulation and efficient rendering for soft body objects. 