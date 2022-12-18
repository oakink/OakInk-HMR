import os
import time

import numpy as np
import skimage.measure
import torch
import trimesh
from termcolor import cprint


def dump_obj_mesh(filename, vertices, faces=None):
    assert vertices.shape[1] == 3 and (faces is None or faces.shape[1] == 3)
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if faces is not None and isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    with open(filename, "w") as obj_file:
        for v in vertices:
            obj_file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        if faces is not None:
            for f in faces + 1:
                obj_file.write("f {} {} {}\n".format(f[0], f[1], f[2]))


def create_query_points(N=256, voxel_origin=[-1, -1, -1], voxel_size=None):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    if voxel_size is None:
        voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index

    samples[:, 2] = overall_index % N
    samples[:, 1] = (torch.div(overall_index.long(), N, rounding_mode="trunc")) % N
    samples[:, 0] = (torch.div(torch.div(overall_index.long(), N, rounding_mode="trunc"), N, rounding_mode="trunc")) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]

    num_samples = N**3
    return samples, num_samples


def get_higher_res_cube(sdf_value, N, raw_voxel_N):
    raw_voxel_size = 2.0 / (raw_voxel_N - 1)
    sdf = sdf_value[:, 3].reshape(raw_voxel_N, raw_voxel_N, raw_voxel_N)
    indices = torch.nonzero(sdf < 0).float()
    if indices.shape[0] == 0:
        min_idx = torch.zeros(3, dtype=torch.float32)
        max_idx = torch.zeros(3, dtype=torch.float32)
    else:
        min_idx = torch.min(indices, dim=0)[0]
        max_idx = torch.max(indices, dim=0)[0]

    # Buffer 2 voxels each side
    new_cube_size = (torch.max(max_idx - min_idx) + 4) * raw_voxel_size

    new_voxel_size = new_cube_size / (N - 1)
    # [z,y,x]
    new_origin = (min_idx - 2) * raw_voxel_size - 1.0  # (-1,-1,-1) origin

    return new_voxel_size, new_origin


def convert_sdf_samples_to_ply(
    sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    eval_mode=False,
    task="obman",
    obj_rot=None,
    obj_tsl=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(sdf_tensor, level=0.0, spacing=[voxel_size] * 3)
    except Exception as e:
        # cprint(e, "red")
        cprint("Marching cubes failed, returning empty mesh. DO NOT USE this when reporting the score", "red")
        # ! warning: DO NOT USE this when reporting the score
        m = trimesh.primitives.Sphere(radius=1, center=[0, 0, 0], subdivisions=1)
        verts = np.asfarray(m.vertices, dtype=np.float32)
        faces = np.asarray(m.faces, dtype=np.int32)

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points * scale
    if offset is not None:
        mesh_points = mesh_points + offset

    if obj_rot is not None:
        mesh_points = mesh_points @ obj_rot.T
    if obj_tsl is not None:
        mesh_points = mesh_points + obj_tsl

    source_mesh = trimesh.Trimesh(vertices=mesh_points, faces=faces, process=False)

    # * >>>> Only preserve the largest connected component
    # TODO only need to remove the noise, not the whole mesh
    split_mesh = trimesh.graph.split(source_mesh)
    if len(split_mesh) > 1:
        max_area = -1
        final_mesh = split_mesh[0]
        for per_mesh in split_mesh:
            if per_mesh.area > max_area:
                max_area = per_mesh.area
                final_mesh = per_mesh
        source_mesh = final_mesh
    # * <<<<

    trans = np.array([0, 0, 0])
    scale = np.array([1])
    if False:  # if eval_mode:
        mesh_dir = "mesh_" + ply_filename_out.split("_")[-1].split(".")[0]
        gt_mesh_name = ply_filename_out.split("/")[-1].split("_")[0] + ".obj"
        gt_mesh_path = os.path.join(f"data/{task}/test", mesh_dir, gt_mesh_name)

        target_mesh = trimesh.load(gt_mesh_path, process=False)
        icp_solver = ICP_T_S(source_mesh, target_mesh)
        icp_solver.sample_mesh(30000, "both")
        icp_solver.run_icp_f(max_iter=100)
        icp_solver.export_source_mesh(ply_filename_out)
        trans, scale = icp_solver.get_trans_scale()
    else:
        if ply_filename_out is not None:
            source_mesh.export(ply_filename_out)
    verts = np.asfarray(source_mesh.vertices, dtype="float32")
    faces = np.asfarray(source_mesh.faces, dtype="int32")
    return verts, faces, trans, scale


def triangle_direction_intersection(tri, trg):
    """
    Finds where an origin-centered ray going in direction trg intersects a triangle.
    Args:
        tri: 3 X 3 vertex locations. tri[0, :] is 0th vertex.
    Returns:
        alpha, beta, gamma
    """
    p0 = np.copy(tri[0, :])
    # Don't normalize
    d1 = np.copy(tri[1, :]) - p0
    d2 = np.copy(tri[2, :]) - p0
    d = trg / np.linalg.norm(trg)

    mat = np.stack([d1, d2, d], axis=1)

    try:
        inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return False, 0

    # inv_mat = np.linalg.inv(mat)

    a_b_mg = -1 * np.matmul(inv_mat, p0)
    is_valid = (a_b_mg[0] >= 0) and (a_b_mg[1] >= 0) and ((a_b_mg[0] + a_b_mg[1]) <= 1) and (a_b_mg[2] < 0)
    if is_valid:
        return True, -a_b_mg[2] * d
    else:
        return False, 0


def project_verts_on_mesh(verts, mesh_verts, mesh_faces):
    verts_out = np.copy(verts)
    from tqdm import tqdm

    for nv in tqdm(range(verts.shape[0])):
        max_norm = 0
        vert = np.copy(verts_out[nv, :])
        for f in range(mesh_faces.shape[0]):
            face = mesh_faces[f]
            tri = mesh_verts[face, :]
            # is_v=True if it does intersect and returns the point
            is_v, pt = triangle_direction_intersection(tri, vert)
            # Take the furthest away intersection point
            if is_v and np.linalg.norm(pt) > max_norm:
                max_norm = np.linalg.norm(pt)
                verts_out[nv, :] = pt

    return verts_out


def compute_edges2verts(verts, faces):
    """
    Returns a list: [A, B, C, D] the 4 vertices for each edge.
    """
    edge_dict = {}
    for face_id, (face) in enumerate(faces):
        for e1, e2, o_id in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
            edge = tuple(sorted((face[e1], face[e2])))
            other_v = face[o_id]
            if edge not in edge_dict.keys():
                edge_dict[edge] = [other_v]
            else:
                if other_v not in edge_dict[edge]:
                    edge_dict[edge].append(other_v)
    result = np.stack([np.hstack((edge, other_vs)) for edge, other_vs in edge_dict.items()])
    return result
