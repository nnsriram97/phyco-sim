#!/usr/bin/env python3
"""
OBJ Tetrahedralization with Texture Preservation

This script takes an OBJ file with texture information, tetrahedralizes it using tetgen,
and outputs:
1. A VTK file containing the tetrahedral mesh
2. An OBJ file containing the surface mesh of the tetrahedralized volume
3. Preserves texture mapping and material information
4. Saves vertex correspondence data between surface OBJ and volumetric VTK
"""

import os
import shutil
import numpy as np
import pyvista as pv
import tetgen
from pathlib import Path


def load_obj_with_texture(obj_path):
    """
    Load OBJ file and preserve texture information including normals.
    
    Parameters
    ----------
    obj_path : str or Path
        Path to the OBJ file
        
    Returns
    -------
    mesh : pyvista.PolyData
        The loaded mesh
    texture_data : dict
        Dictionary containing texture information (UV coordinates, normals, material info)
    """
    obj_path = Path(obj_path)
    
    # Load the mesh using PyVista
    mesh = pv.read(str(obj_path))
    
    # Parse OBJ file manually to extract texture coordinates, normals and material info
    texture_data = {
        'uv_coords': None,
        'normals': None,
        'material_file': None,
        'material_name': None,
        'texture_file': None,
        'face_uv_indices': None,
        'face_normal_indices': None
    }
    
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    
    vertices = []
    uv_coords = []
    normals = []
    face_uv_indices = []
    face_normal_indices = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('v '):
            # Vertex
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith('vn '):
            # Vertex normal
            parts = line.split()
            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith('mtllib '):
            # Material library file
            texture_data['material_file'] = line.split()[1]
        elif line.startswith('usemtl '):
            # Material name
            texture_data['material_name'] = line.split()[1]
        elif line.startswith('vt '):
            # Texture coordinate
            parts = line.split()
            uv_coords.append([float(parts[1]), float(parts[2])])
        elif line.startswith('f '):
            # Face with potential texture and normal indices
            parts = line.split()[1:]
            face_uv = []
            face_normals = []
            for vertex_data in parts:
                if '/' in vertex_data:
                    # Format: v/vt/vn or v/vt or v//vn
                    vertex_parts = vertex_data.split('/')
                    if len(vertex_parts) > 1 and vertex_parts[1]:
                        face_uv.append(int(vertex_parts[1]) - 1)  # OBJ uses 1-based indexing
                    if len(vertex_parts) > 2 and vertex_parts[2]:
                        face_normals.append(int(vertex_parts[2]) - 1)  # OBJ uses 1-based indexing
            if face_uv:
                face_uv_indices.append(face_uv)
            if face_normals:
                face_normal_indices.append(face_normals)
    
    if uv_coords:
        texture_data['uv_coords'] = np.array(uv_coords)
        texture_data['face_uv_indices'] = face_uv_indices
    
    if normals:
        texture_data['normals'] = np.array(normals)
        texture_data['face_normal_indices'] = face_normal_indices
    
    # Try to find texture file from MTL file
    if texture_data['material_file']:
        mtl_path = obj_path.parent / texture_data['material_file']
        if mtl_path.exists():
            with open(mtl_path, 'r') as f:
                mtl_lines = f.readlines()
            
            for line in mtl_lines:
                line = line.strip()
                if line.startswith('map_Kd '):
                    texture_data['texture_file'] = line.split()[1]
                    break
    
    print(f"Loaded OBJ with:")
    print(f"  - {len(vertices)} vertices")
    print(f"  - {len(uv_coords)} UV coordinates" if uv_coords else "  - No UV coordinates")
    print(f"  - {len(normals)} normals" if normals else "  - No normals")
    print(f"  - Material: {texture_data['material_name']}" if texture_data['material_name'] else "  - No material")
    print(f"  - Texture: {texture_data['texture_file']}" if texture_data['texture_file'] else "  - No texture")
    
    return mesh, texture_data


def tetrahedralize_mesh(mesh, quality_params=None):
    """
    Tetrahedralize the input mesh using tetgen.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Input surface mesh
    quality_params : dict, optional
        Parameters for tetgen quality control
        
    Returns
    -------
    tet_grid : pyvista.UnstructuredGrid
        Tetrahedral mesh
    """
    if quality_params is None:
        quality_params = {
            'order': 1,
            'mindihedral': 20,
            'minratio': 1.5,
            'quality': True
        }
    
    # Ensure mesh is triangulated and manifold
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    
    # Create TetGen object and tetrahedralize
    tet = tetgen.TetGen(mesh)
    
    # Optionally make manifold (requires pymeshfix)
    try:
        tet.make_manifold()
        print("Made mesh manifold")
    except ImportError:
        print("Warning: pymeshfix not available, skipping manifold repair")
    except Exception as e:
        print(f"Warning: Could not make mesh manifold: {e}")
    
    # Tetrahedralize
    tet.tetrahedralize(**quality_params)
    
    return tet.grid


def extract_surface_with_mapping(tet_grid):
    """
    Extract surface mesh from tetrahedral grid and create vertex mapping.
    
    Parameters
    ----------
    tet_grid : pyvista.UnstructuredGrid
        Tetrahedral mesh
        
    Returns
    -------
    surface_mesh : pyvista.PolyData
        Surface mesh
    surface_to_volume_map : numpy.ndarray
        Array mapping surface vertex indices to volume vertex indices
    """
    # Extract surface
    surface_mesh = tet_grid.extract_surface()
    
    # Find mapping from surface vertices to volume vertices
    surface_points = surface_mesh.points
    volume_points = tet_grid.points
    
    # For each surface point, find corresponding volume point
    surface_to_volume_map = np.zeros(len(surface_points), dtype=int)
    
    for i, surf_pt in enumerate(surface_points):
        # Find closest point in volume mesh (should be exact match)
        distances = np.linalg.norm(volume_points - surf_pt, axis=1)
        surface_to_volume_map[i] = np.argmin(distances)
    
    return surface_mesh, surface_to_volume_map


def map_texture_to_surface(original_mesh, surface_mesh, texture_data):
    """
    Map texture coordinates and normals from original mesh to extracted surface mesh.
    
    Parameters
    ----------
    original_mesh : pyvista.PolyData
        Original input mesh
    surface_mesh : pyvista.PolyData
        Extracted surface mesh
    texture_data : dict
        Texture information from original mesh
        
    Returns
    -------
    tuple
        (surface_uv_coords, surface_normals) - UV coordinates and normals for surface mesh vertices
    """
    surface_points = surface_mesh.points
    original_points = original_mesh.points
    
    surface_uv_coords = None
    surface_normals = None
    
    # Map UV coordinates if available
    if texture_data['uv_coords'] is not None:
        surface_uv_coords = np.zeros((len(surface_points), 2))
        
        # Create a mapping from original vertex indices to UV coordinates
        # This is tricky because face-based UV mapping can have multiple UVs per vertex
        vertex_to_uv = {}
        
        if texture_data['face_uv_indices']:
            # Use face-based UV mapping for better accuracy
            original_faces = original_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove face size prefix
            for face_idx, face_vertices in enumerate(original_faces):
                if face_idx < len(texture_data['face_uv_indices']):
                    face_uv_indices = texture_data['face_uv_indices'][face_idx]
                    for vertex_idx, uv_idx in zip(face_vertices, face_uv_indices):
                        if uv_idx < len(texture_data['uv_coords']):
                            # Store the UV coordinate for this vertex (may overwrite, but that's OK)
                            vertex_to_uv[vertex_idx] = texture_data['uv_coords'][uv_idx]
        else:
            # Fallback: direct vertex-to-UV mapping
            for i, uv in enumerate(texture_data['uv_coords']):
                if i < len(original_points):
                    vertex_to_uv[i] = uv
        
        # Map surface vertices to original vertices and get their UV coordinates
        for i, surf_pt in enumerate(surface_points):
            # Find closest point in original mesh
            distances = np.linalg.norm(original_points - surf_pt, axis=1)
            closest_idx = np.argmin(distances)
            
            # Check if the closest point is very close (same vertex)
            if distances[closest_idx] < 1e-10:
                if closest_idx in vertex_to_uv:
                    surface_uv_coords[i] = vertex_to_uv[closest_idx]
                elif closest_idx < len(texture_data['uv_coords']):
                    surface_uv_coords[i] = texture_data['uv_coords'][closest_idx]
    
    # Map normals if available
    if texture_data['normals'] is not None:
        surface_normals = np.zeros((len(surface_points), 3))
        
        # Create a mapping from original vertex indices to normals
        vertex_to_normal = {}
        
        if texture_data['face_normal_indices']:
            # Use face-based normal mapping for better accuracy
            original_faces = original_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove face size prefix
            for face_idx, face_vertices in enumerate(original_faces):
                if face_idx < len(texture_data['face_normal_indices']):
                    face_normal_indices = texture_data['face_normal_indices'][face_idx]
                    for vertex_idx, normal_idx in zip(face_vertices, face_normal_indices):
                        if normal_idx < len(texture_data['normals']):
                            vertex_to_normal[vertex_idx] = texture_data['normals'][normal_idx]
        else:
            # Fallback: direct vertex-to-normal mapping
            for i, normal in enumerate(texture_data['normals']):
                if i < len(original_points):
                    vertex_to_normal[i] = normal
        
        # Map surface vertices to original vertices and get their normals
        for i, surf_pt in enumerate(surface_points):
            # Find closest point in original mesh
            distances = np.linalg.norm(original_points - surf_pt, axis=1)
            closest_idx = np.argmin(distances)
            
            # Check if the closest point is very close (same vertex)
            if distances[closest_idx] < 1e-10:
                if closest_idx in vertex_to_normal:
                    surface_normals[i] = vertex_to_normal[closest_idx]
                elif closest_idx < len(texture_data['normals']):
                    surface_normals[i] = texture_data['normals'][closest_idx]
    
    print(f"Mapped to surface:")
    print(f"  - UV coordinates: {surface_uv_coords is not None}")
    print(f"  - Normals: {surface_normals is not None}")
    
    return surface_uv_coords, surface_normals


def write_obj_with_texture(surface_mesh, surface_uv_coords, surface_normals, texture_data, output_path):
    """
    Write surface mesh as OBJ file with texture information and normals.
    
    Parameters
    ----------
    surface_mesh : pyvista.PolyData
        Surface mesh to write
    surface_uv_coords : numpy.ndarray or None
        UV coordinates for vertices
    surface_normals : numpy.ndarray or None
        Normal vectors for vertices
    texture_data : dict
        Original texture information
    output_path : str or Path
        Output OBJ file path
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("# OBJ file generated from tetrahedralized mesh\n")
        f.write("# Preserving texture mapping and normals from original\n\n")
        
        # Write material file reference if available
        if texture_data['material_file']:
            f.write(f"mtllib {texture_data['material_file']}\n\n")
        
        # Write vertices
        for vertex in surface_mesh.points:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        f.write("\n")
        
        # Write normals if available
        if surface_normals is not None:
            for normal in surface_normals:
                f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            f.write("\n")
        
        # Write texture coordinates if available
        if surface_uv_coords is not None:
            for uv in surface_uv_coords:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            f.write("\n")
        
        # Write material usage if available
        if texture_data['material_name']:
            f.write(f"usemtl {texture_data['material_name']}\n")
        
        # Write faces with appropriate format based on available data
        faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove face size prefix
        for face in faces:
            f.write("f")
            for vertex_idx in face:
                # Build face vertex specification based on available data
                vertex_spec = str(vertex_idx + 1)  # OBJ uses 1-based indexing
                
                # Add texture coordinate index if available
                if surface_uv_coords is not None:
                    vertex_spec += f"/{vertex_idx + 1}"
                elif surface_normals is not None:
                    vertex_spec += "/"  # Empty texture coordinate slot
                
                # Add normal index if available
                if surface_normals is not None:
                    if surface_uv_coords is None:
                        vertex_spec += "/"  # Need to add the empty texture slot first
                    vertex_spec += f"/{vertex_idx + 1}"
                
                f.write(f" {vertex_spec}")
            f.write("\n")
    
    print(f"Written OBJ file: {output_path}")
    print(f"  - {len(surface_mesh.points)} vertices")
    print(f"  - {len(faces)} faces")
    print(f"  - UV coordinates: {'Yes' if surface_uv_coords is not None else 'No'}")
    print(f"  - Normals: {'Yes' if surface_normals is not None else 'No'}")
    print(f"  - Material: {texture_data['material_name'] or 'None'}")
    print(f"  - Texture file: {texture_data['texture_file'] or 'None'}")


def write_vtk_with_mapping(tet_grid, surface_to_volume_map, output_path):
    """
    Write tetrahedral mesh as VTK file in ASCII (text-readable) format.
    Ensures only tetrahedra (cell type 10) are included.
    
    Parameters
    ----------
    tet_grid : pyvista.UnstructuredGrid
        Tetrahedral mesh
    surface_to_volume_map : numpy.ndarray
        Mapping from surface to volume vertices (not used in VTK, kept for API consistency)
    output_path : str or Path
        Output VTK file path
    """
    # Filter to keep only tetrahedral cells (cell type 10)
    cell_types = tet_grid.celltypes
    tetra_mask = (cell_types == 10)  # VTK cell type 10 = tetrahedron
    
    if not tetra_mask.all():
        print(f"Warning: Found {(~tetra_mask).sum()} non-tetrahedral cells, removing them...")
        # Extract only tetrahedral cells
        tetra_indices = np.where(tetra_mask)[0]
        clean_grid = tet_grid.extract_cells(tetra_indices)
    else:
        clean_grid = tet_grid
    
    # Verify all cells are tetrahedra with 4 vertices each
    cells = clean_grid.cells
    cell_array = cells.reshape(-1, 5)  # Should be [4, v1, v2, v3, v4] for each tetrahedron
    
    # Check that all cells start with 4 (indicating 4 vertices per cell)
    if not (cell_array[:, 0] == 4).all():
        raise ValueError("Error: Not all cells are tetrahedra (should have 4 vertices each)")
    
    # Verify cell types are all 10 (tetrahedra)
    if not (clean_grid.celltypes == 10).all():
        raise ValueError("Error: Found non-tetrahedral cell types after filtering")
    
    print(f"Writing VTK with {clean_grid.n_points} vertices and {clean_grid.n_cells} tetrahedra")
    
    # Write VTK file in ASCII format (text-readable)
    clean_grid.save(str(output_path), binary=False)
    
    # Verify the written file by reading it back and checking format
    verify_vtk_format(output_path)


def write_legacy_vtk_format(tet_grid, output_path):
    """
    Write VTK file in the exact legacy format you specified.
    
    Parameters
    ----------
    tet_grid : pyvista.UnstructuredGrid
        Tetrahedral mesh
    output_path : str or Path
        Output VTK file path
    """
    points = tet_grid.points
    cells = tet_grid.cells
    cell_types = tet_grid.celltypes
    
    # Ensure we only have tetrahedra
    if not (cell_types == 10).all():
        # Filter to keep only tetrahedral cells
        tetra_mask = (cell_types == 10)
        tetra_indices = np.where(tetra_mask)[0]
        tet_grid = tet_grid.extract_cells(tetra_indices)
        points = tet_grid.points
        cells = tet_grid.cells
        cell_types = tet_grid.celltypes
    
    # Parse cells into the format we need
    cell_array = cells.reshape(-1, 5)  # [4, v1, v2, v3, v4] for each tetrahedron
    
    # Verify format
    if not (cell_array[:, 0] == 4).all():
        raise ValueError("Not all cells are tetrahedra")
    
    num_points = len(points)
    num_cells = len(cell_array)
    total_cell_size = num_cells * 5  # Each tetrahedron: 4 + 4 vertices
    
    with open(output_path, 'w') as f:
        # Write VTK header
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Tetrahedral mesh, Created by tetgen processor\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Write points
        f.write(f"POINTS {num_points} double\n")
        for point in points:
            f.write(f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g}\n")
        
        # Write cells in legacy format
        f.write(f"\nCELLS {num_cells} {total_cell_size}\n")
        for cell in cell_array:
            f.write(f"4 {cell[1]} {cell[2]} {cell[3]} {cell[4]}\n")
        
        # Write cell types
        f.write(f"\nCELL_TYPES {num_cells}\n")
        for _ in range(num_cells):
            f.write("10\n")


def verify_vtk_format(vtk_path):
    """
    Verify that the VTK file matches the exact format you specified.
    
    Parameters
    ----------
    vtk_path : str or Path
        Path to VTK file to verify
    """
    with open(vtk_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Find key sections
    cells_start = None
    cell_types_start = None
    
    for i, line in enumerate(lines):
        if line.startswith('CELLS'):
            cells_start = i
        elif line.startswith('CELL_TYPES'):
            cell_types_start = i
    
    if cells_start is None or cell_types_start is None:
        raise ValueError("Could not find CELLS or CELL_TYPES sections in VTK file")
    
    # Parse headers
    cells_header = lines[cells_start].split()
    num_cells = int(cells_header[1])
    total_size = int(cells_header[2])
    
    cell_types_header = lines[cell_types_start].split()
    num_cell_types = int(cell_types_header[1])
    
    print(f"Verifying VTK format:")
    print(f"  - CELLS: {num_cells} cells, {total_size} total size")
    print(f"  - CELL_TYPES: {num_cell_types} entries")
    
    # Verify consistency
    if num_cells != num_cell_types:
        raise ValueError(f"Number of cells ({num_cells}) != number of cell types ({num_cell_types})")
    
    expected_total_size = num_cells * 5  # Each tetrahedron: 4 + 4 vertex indices
    if total_size != expected_total_size:
        raise ValueError(f"Total size {total_size} != expected {expected_total_size} for {num_cells} tetrahedra")
    
    # Verify CELLS section format
    cells_data = []
    for i in range(cells_start + 1, cell_types_start):
        line = lines[i]
        if not line:
            continue
        try:
            cells_data.extend(map(int, line.split()))
        except ValueError:
            break
    
    print(f"  - Found {len(cells_data)} cell data entries")
    
    if len(cells_data) != total_size:
        raise ValueError(f"Cell data length {len(cells_data)} != header total size {total_size}")
    
    # Check each cell format: 4 v1 v2 v3 v4
    pos = 0
    cell_count = 0
    while pos < len(cells_data):
        if pos + 4 >= len(cells_data):
            break
        
        num_vertices = cells_data[pos]
        if num_vertices != 4:
            raise ValueError(f"Cell {cell_count} starts with {num_vertices}, expected 4")
        
        pos += 5  # Move to next cell
        cell_count += 1
    
    if cell_count != num_cells:
        raise ValueError(f"Parsed {cell_count} cells, expected {num_cells}")
    
    # Verify CELL_TYPES section
    cell_types_data = []
    for i in range(cell_types_start + 1, len(lines)):
        line = lines[i]
        if not line or line.startswith('POINT_DATA') or line.startswith('CELL_DATA'):
            break
        try:
            cell_types_data.extend(map(int, line.split()))
        except ValueError:
            break
    
    print(f"  - Found {len(cell_types_data)} cell type entries")
    
    if len(cell_types_data) != num_cells:
        raise ValueError(f"Cell types length {len(cell_types_data)} != num_cells {num_cells}")
    
    # Check all cell types are 10 (tetrahedra)
    non_tetra_types = [ct for ct in cell_types_data if ct != 10]
    if non_tetra_types:
        raise ValueError(f"Found non-tetrahedral cell types: {set(non_tetra_types)}")
    
    print("✓ VTK format verification passed!")
    print(f"  - {num_cells} tetrahedra")
    print(f"  - All cells have format: 4 v1 v2 v3 v4")
    print(f"  - All cell types are 10 (tetrahedra)")
    print(f"  - Legacy VTK format (ASCII)")


def write_vtk_with_mapping(tet_grid, surface_to_volume_map, output_path):
    """
    Write tetrahedral mesh as VTK file in the exact legacy format specified.
    
    Parameters
    ----------
    tet_grid : pyvista.UnstructuredGrid
        Tetrahedral mesh
    surface_to_volume_map : numpy.ndarray
        Mapping from surface to volume vertices (not used in VTK, kept for API consistency)
    output_path : str or Path
        Output VTK file path
    """
    # Filter to keep only tetrahedral cells (cell type 10)
    cell_types = tet_grid.celltypes
    tetra_mask = (cell_types == 10)  # VTK cell type 10 = tetrahedron
    
    if not tetra_mask.all():
        print(f"Warning: Found {(~tetra_mask).sum()} non-tetrahedral cells, removing them...")
        # Extract only tetrahedral cells
        tetra_indices = np.where(tetra_mask)[0]
        clean_grid = tet_grid.extract_cells(tetra_indices)
    else:
        clean_grid = tet_grid
    
    # Verify all cells are tetrahedra with 4 vertices each
    cells = clean_grid.cells
    cell_array = cells.reshape(-1, 5)  # Should be [4, v1, v2, v3, v4] for each tetrahedron
    
    # Check that all cells start with 4 (indicating 4 vertices per cell)
    if not (cell_array[:, 0] == 4).all():
        raise ValueError("Error: Not all cells are tetrahedra (should have 4 vertices each)")
    
    # Verify cell types are all 10 (tetrahedra)
    if not (clean_grid.celltypes == 10).all():
        raise ValueError("Error: Found non-tetrahedral cell types after filtering")
    
    print(f"Writing VTK with {clean_grid.n_points} vertices and {clean_grid.n_cells} tetrahedra")
    
    # Write VTK file in the exact legacy format
    write_legacy_vtk_format(clean_grid, output_path)
    
    # Verify the written file
    verify_vtk_format(output_path)


def copy_texture_files(original_obj_path, output_dir, texture_data):
    """
    Copy material and texture files to output directory.
    
    Parameters
    ----------
    original_obj_path : Path
        Path to original OBJ file
    output_dir : Path
        Output directory
    texture_data : dict
        Texture information
    """
    original_dir = Path(original_obj_path).parent
    
    # Copy MTL file
    if texture_data['material_file']:
        mtl_src = original_dir / texture_data['material_file']
        if mtl_src.exists():
            mtl_dst = output_dir / texture_data['material_file']
            shutil.copy2(mtl_src, mtl_dst)
            print(f"Copied material file: {mtl_dst}")
    
    # Copy texture file
    if texture_data['texture_file']:
        texture_src = original_dir / texture_data['texture_file']
        if texture_src.exists():
            texture_dst = output_dir / texture_data['texture_file']
            shutil.copy2(texture_src, texture_dst)
            print(f"Copied texture file: {texture_dst}")


def process_obj_tetrahedralization(input_obj_path, output_dir=None, quality_params=None):
    """
    Main function to process OBJ tetrahedralization with texture preservation.
    
    Parameters
    ----------
    input_obj_path : str or Path
        Path to input OBJ file
    output_dir : str or Path, optional
        Output directory (default: same as input file)
    quality_params : dict, optional
        Parameters for tetgen quality control
        
    Returns
    -------
    dict
        Dictionary with paths to output files
    """
    input_obj_path = Path(input_obj_path)
    
    if output_dir is None:
        output_dir = input_obj_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = input_obj_path.stem
    
    print(f"Processing: {input_obj_path}")
    
    # Step 1: Load OBJ with texture information
    print("Loading OBJ file and texture data...")
    mesh, texture_data = load_obj_with_texture(input_obj_path)
    print(f"Loaded mesh with {mesh.n_points} vertices and {mesh.n_cells} faces")
    
    # Step 2: Tetrahedralize
    print("Tetrahedralizing mesh...")
    tet_grid = tetrahedralize_mesh(mesh, quality_params)
    print(f"Generated tetrahedral mesh with {tet_grid.n_points} vertices and {tet_grid.n_cells} tetrahedra")
    
    # Step 3: Extract surface with mapping
    print("Extracting surface mesh...")
    surface_mesh, surface_to_volume_map = extract_surface_with_mapping(tet_grid)
    print(f"Extracted surface with {surface_mesh.n_points} vertices and {surface_mesh.n_cells} faces")
    
    # Step 4: Map texture coordinates and normals
    print("Mapping texture coordinates and normals...")
    surface_uv_coords, surface_normals = map_texture_to_surface(mesh, surface_mesh, texture_data)
    
    # Step 5: Write outputs
    output_obj_path = output_dir / f"{base_name}.obj"
    output_vtk_path = output_dir / f"{base_name}.vtk"
    
    print(f"Writing surface OBJ: {output_obj_path}")
    write_obj_with_texture(surface_mesh, surface_uv_coords, surface_normals, texture_data, output_obj_path)
    
    print(f"Writing volume VTK: {output_vtk_path}")
    write_vtk_with_mapping(tet_grid, surface_to_volume_map, output_vtk_path)
    
    # Step 6: Copy texture files
    print("Copying texture files...")
    copy_texture_files(input_obj_path, output_dir, texture_data)
    
    # Save mapping data as additional file
    mapping_path = output_dir / f"{base_name}.txt"
    with open(mapping_path, 'w') as f:
        f.write("# Surface vertex ID -> Volume vertex ID mapping\n")
        f.write("# Format: surface_id volume_id\n")
        for surf_idx, vol_idx in enumerate(surface_to_volume_map):
            f.write(f"{surf_idx} {vol_idx}\n")
    
    print("Processing complete!")
    
    return {
        'surface_obj': output_obj_path,
        'volume_vtk': output_vtk_path,
        'vertex_mapping': mapping_path,
        'output_dir': output_dir
    }


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tetrahedralize OBJ file with texture preservation")
    parser.add_argument("input_obj", help="Input OBJ file path")
    parser.add_argument("-o", "--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("--mindihedral", type=float, default=20, help="Minimum dihedral angle")
    parser.add_argument("--minratio", type=float, default=1.5, help="Minimum radius-edge ratio")
    parser.add_argument("--order", type=int, default=1, help="Element order (1 or 2)")
    
    args = parser.parse_args()
    
    quality_params = {
        'order': args.order,
        'mindihedral': args.mindihedral,
        'minratio': args.minratio,
        'quality': True
    }
    
    try:
        results = process_obj_tetrahedralization(
            args.input_obj, 
            args.output_dir, 
            quality_params
        )
        
        print("\nOutput files:")
        for key, path in results.items():
            if key != 'output_dir':
                print(f"  {key}: {path}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()