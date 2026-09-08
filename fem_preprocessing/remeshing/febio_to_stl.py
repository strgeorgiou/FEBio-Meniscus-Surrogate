import os
import subprocess
import xml.etree.ElementTree as ET

import meshio
import numpy as np


# Define paths
input_febio_file = r"path\to\input_model.fs2"
working_directory = r"path\to\working_directory"
out_stl_file = os.path.join(working_directory, "output.stl")


# Step 1: Extract the surface mesh from the FEBio file and write a TetGen .poly file
def extract_surface_mesh(febio_file, output_poly_file):
    tree = ET.parse(febio_file)
    root = tree.getroot()

    nodes = []
    faces = []

    # Extract nodes
    for node in root.findall(".//node"):
        node_id = int(node.attrib["id"])
        x, y, z = map(float, node.text.strip().split(","))
        nodes.append((node_id, x, y, z))

    # Extract surface faces from the FEBio surface definitions
    for surface in root.findall(".//surface"):
        for face in surface.findall("face"):
            face_nodes = list(map(int, face.text.strip().split(",")))
            faces.append(face_nodes)

    # Write TetGen .poly file
    with open(output_poly_file, "w") as f:
        # Points
        f.write(f"{len(nodes)} 3 0 0\n")
        for node in nodes:
            f.write(f"{node[0]} {node[1]} {node[2]} {node[3]}\n")

        # Surface faces
        f.write(f"{len(faces)} 0\n")
        for face in faces:
            f.write(f"1 0 {face[0]} {face[1]} {face[2]}\n")

        # No holes
        f.write("0\n")

        # No region attributes
        f.write("0\n")

    return nodes


# Step 2: Estimate a characteristic mesh size from the average nodal distance
def estimate_target_volume(nodes):
    coords = np.array([node[1:] for node in nodes])

    total_length = 0
    count = 0

    # Compute the average distance between node pairs
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            total_length += dist
            count += 1

    avg_nodal_distance = total_length / count
    target_length = avg_nodal_distance / 2

    # Approximate target tetrahedral volume
    target_volume = (target_length ** 3) / 6

    return target_volume


# Step 3: Call TetGen to remesh
def run_tetgen(input_poly_file, max_volume):
    cmd = [
        "tetgen",
        f"-pqag -a{max_volume}",
        input_poly_file
    ]

    subprocess.run(" ".join(cmd), check=True, shell=True)


# Step 4: Convert TetGen output to STL
def convert_to_stl(tetgen_output_prefix, output_stl):
    mesh = meshio.read(f"{tetgen_output_prefix}.1.vtk")
    meshio.write(output_stl, mesh, file_format="stl")


# Step 5: Clean up intermediate files
def cleanup(files):
    for file in files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    poly_file = os.path.join(working_directory, "mesh.poly")
    tetgen_prefix = os.path.join(working_directory, "mesh")

    print("Extracting surface mesh from FEBio FS2 file...")
    nodes = extract_surface_mesh(input_febio_file, poly_file)

    print("Estimating target element volume...")
    target_volume = estimate_target_volume(nodes)
    print(f"Target volume: {target_volume}")

    print("Running TetGen remeshing...")
    run_tetgen(poly_file, max_volume=target_volume)

    print("Converting mesh to STL...")
    convert_to_stl(tetgen_prefix, out_stl_file)

    print("Cleaning up intermediate files...")
    cleanup([
        poly_file,
        f"{tetgen_prefix}.1.node",
        f"{tetgen_prefix}.1.ele",
        f"{tetgen_prefix}.1.face",
        f"{tetgen_prefix}.1.neigh",
        f"{tetgen_prefix}.1.vtk"
    ])

    print(f"STL file saved to: {out_stl_file}")