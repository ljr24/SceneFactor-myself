import os, sys
import numpy as np
import json
import trimesh
import gc
from tqdm import tqdm

import mesh2sdf.core
from scipy.spatial import cKDTree


FRONT3DMESHESMANIFOLD = '/home/lijiarui/Desktop/scene_factor/SceneFactor/assets/SELECTED_FRONT_SCENES'
FUTURE3DMODEL = '/home/lijiarui/Desktop/scene_factor/SceneFactor/assets/SELECTED_FUTURE_MODELS'
SAVEDIR = '/home/lijiarui/Desktop/scene_factor/SceneFactor/assets/chunked_data_lowres'
os.makedirs(SAVEDIR, exist_ok=True)


cat_to_id = {'Bed': 2,
             'Pier/Stool': 3,
             'Cabinet/Shelf/Desk': 4,
             'Lighting': 5,
             'Sofa': 6,
             'Chair': 7,
             'Table': 8,
             'Others': 9}


def put_scene_to_center(vertices, cell_center, cell_size):
    vertices = vertices - cell_center
    vertices = vertices / float(cell_size / 2)
    return vertices


def put_scene_back_from_center(vertices, cell_center, cell_size):
    vertices = vertices * cell_size / 2
    vertices = vertices + cell_center
    return vertices


def load_furniture(jid):
    folder_path = os.path.join(FUTURE3DMODEL, jid)
    if os.path.exists(os.path.join(folder_path, "raw_model_fixed.obj")):
        obj_file = os.path.join(folder_path, "raw_model_fixed.obj")
    else:
        obj_file = os.path.join(folder_path, "raw_model.obj")
    if os.path.exists(obj_file) and "7e101ef3-7722-4af8-90d5-7c562834fabd" not in obj_file:
        furniture_mesh = trimesh.load(obj_file, force='mesh')
        return furniture_mesh
    else:
        return None


def transform_furniture(obj, trans):
    current_obj = obj.copy()
    vertices = np.array(current_obj.vertices)
    vertices = np.hstack([vertices, np.ones([len(vertices), 1])])
    vertices = vertices @ trans.T
    current_obj.vertices = vertices[:, :3]
    return current_obj


def compute_points_canonic(surface_points):
    from copy import deepcopy
    surface_points_canonic = deepcopy(np.array(surface_points))
    surface_points_canonic = surface_points_canonic - surface_points_canonic.mean(axis=0)
    max_coord = surface_points_canonic.max()
    surface_points_canonic = surface_points_canonic / (2 * max_coord) / 1.1
    surface_points_canonic = surface_points_canonic + [0.5, 0.5, 0.5]
    return surface_points_canonic


def aabb_intersect(aabb1, aabb2):
    """检查两个 AABB 是否相交。每个 AABB 是 [[xmin,ymin,zmin],[xmax,ymax,zmax]]"""
    return (aabb1[0][0] <= aabb2[1][0] and aabb1[1][0] >= aabb2[0][0] and
            aabb1[0][1] <= aabb2[1][1] and aabb1[1][1] >= aabb2[0][1] and
            aabb1[0][2] <= aabb2[1][2] and aabb1[1][2] >= aabb2[0][2])


def compute_furniture_world_aabbs(furniture_pts):
    """从家具点云预计算每件家具在世界坐标中的 AABB。
    内存开销极低：只存 6 个 float 每件家具。
    """
    furniture_world_aabbs = {}
    for key in furniture_pts:
        pts = np.array(furniture_pts[key][0])
        if len(pts) > 0:
            furniture_world_aabbs[key] = np.array([pts.min(axis=0), pts.max(axis=0)])
    return furniture_world_aabbs


def load_furniture_for_chunk(obj_id, intersecting_indices, valid_jids):
    """只加载与当前 chunk 相交的家具 mesh。

    加载两种家具源：
    1. FUTURE3D 模型 (*_trans.npy 变换)
    2. 场景目录中的 *_furniture.obj（已在场景坐标系中，无需变换）

    返回: (vertices, faces) 的元组，若没有家具则为 (None, None)
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for idx in intersecting_indices:
        # 优先加载 *_furniture.obj（已在场景坐标系中）
        furniture_obj_path = os.path.join(FRONT3DMESHESMANIFOLD, obj_id, f"{idx}_furniture.obj")
        if os.path.exists(furniture_obj_path):
            try:
                furniture_mesh = trimesh.load(furniture_obj_path, force='mesh')
                all_vertices.append(furniture_mesh.vertices)
                all_faces.append(furniture_mesh.faces + vertex_offset)
                vertex_offset += len(furniture_mesh.vertices)
                del furniture_mesh
                continue
            except Exception:
                pass

        # 回退到 FUTURE3D 模型 + 变换矩阵
        jid = valid_jids.get(str(idx)) or valid_jids.get(idx)
        if jid is None or jid == "Cabinet":
            continue

        furniture_mesh = load_furniture(jid)
        if furniture_mesh is None:
            continue

        trans = np.load(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, f"{idx}_trans.npy"))
        transformed = transform_furniture(furniture_mesh, trans)

        all_vertices.append(transformed.vertices)
        all_faces.append(transformed.faces + vertex_offset)
        vertex_offset += len(transformed.vertices)

        del furniture_mesh, transformed

    if not all_vertices:
        return None, None

    return np.vstack(all_vertices), np.vstack(all_faces)


def compute_chunks(num_proc=1, proc=0):

    all_obj_ids = sorted(os.listdir(FRONT3DMESHESMANIFOLD))[:]
    all_obj_ids = [x for i, x in enumerate(all_obj_ids) if i % num_proc == proc]

    with open('valid_scenes.json', 'r') as fin:
        valid_scenes = json.load(fin)['valid_scenes']

    chunk_size = 3.8
    trunc_dist = 0.08
    scale_margin = 1.0
    max_coord_abs = 1. / scale_margin
    num_voxels = 90
    num_trunc_points = 200000

    # 预创建全局的体素网格坐标（所有 chunk 共用，节省重复分配）
    dx = 2.0 / num_voxels
    grid_1d = np.linspace(-1.0, 1.0 - dx, num_voxels)
    i_grid, j_grid, k_grid = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    GLOBAL_XYZ_GRID = np.stack([i_grid, j_grid, k_grid], axis=-1)  # (90, 90, 90, 3)
    GLOBAL_XYZ_GRID_FLAT = GLOBAL_XYZ_GRID.reshape((-1, 3))

    # 预计算每个 chunk 共用的网格索引（避免重复查找）
    GLOBAL_INSIDE_CHUNK_IDS = np.where(
        (GLOBAL_XYZ_GRID_FLAT[:, 0] >= -max_coord_abs) &
        (GLOBAL_XYZ_GRID_FLAT[:, 1] >= -max_coord_abs) &
        (GLOBAL_XYZ_GRID_FLAT[:, 2] >= -max_coord_abs) &
        (GLOBAL_XYZ_GRID_FLAT[:, 0] < max_coord_abs) &
        (GLOBAL_XYZ_GRID_FLAT[:, 1] < max_coord_abs) &
        (GLOBAL_XYZ_GRID_FLAT[:, 2] < max_coord_abs)
    )[0]

    for obj_id in tqdm(all_obj_ids):
        print(f"\n开始处理场景: {obj_id}")

        if obj_id not in valid_scenes:
            continue

        LOCAL_SAVEDIR = os.path.join(SAVEDIR, obj_id)
        if os.path.exists(LOCAL_SAVEDIR) and (len(os.listdir(LOCAL_SAVEDIR)) > 0):
            continue
        os.makedirs(LOCAL_SAVEDIR, exist_ok=True)

        # --- 跳过家具过多的场景 ---
        with open(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'furniture_jids.json')) as f:
            jids = json.load(f)
            num_furniture = len(jids)
        if num_furniture >= 90:
            print(f"{obj_id} : {num_furniture} furniture items. Skipping memory overflow protection.")
            continue

        # --- 预计算有效家具映射（排除 Cabinet）---
        valid_jids = {k: v for k, v in jids.items() if v != "Cabinet"}

        # --- 加载场景 mesh（轻量：只用 vertices + faces）---
        try:
            mesh_scene = trimesh.load(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'scene.obj'))
            SCENE_VERTS_ORIG = mesh_scene.vertices.copy()
            SCENE_FACES = mesh_scene.faces.copy()
            del mesh_scene  # 立即释放 trimesh 对象
        except Exception:
            print(f"{obj_id} : Loading scene obj failed.")
            continue

        # --- 加载 furniture_points.json & 计算家具世界坐标 AABB ---
        try:
            with open(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'furniture_points.json'), 'r') as fin:
                furniture_pts = json.load(fin)
            furniture_world_aabbs = compute_furniture_world_aabbs(furniture_pts)
        except FileNotFoundError:
            print(f"{obj_id} : furniture_points.json not found.")
            continue

        # --- 加载 scene_points.json ---
        all_scene_pts = []
        try:
            with open(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'scene_points.json'), 'r') as fin:
                scene_pts = json.load(fin)
                for key in scene_pts:
                    all_scene_pts += scene_pts[key]
            all_scene_pts = np.array(all_scene_pts)
        except FileNotFoundError:
            print(f"{obj_id} : scene_points.json not found.")
            continue

        # --- 获取场景边界 ---
        # 需要重新获取 bounds（之前已删除 mesh_scene）
        try:
            mesh_scene_temp = trimesh.load(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'scene.obj'))
            starting_point = mesh_scene_temp.bounds[0] - 0.14
            bounds = mesh_scene_temp.bounds
            del mesh_scene_temp
        except Exception:
            print(f"{obj_id} : Getting scene bounds failed.")
            continue

        n_x = int(np.abs(bounds[0][0] - bounds[1][0]) // chunk_size + 1)
        n_y = int(np.abs(bounds[0][1] - bounds[1][1]) // chunk_size + 1)
        n_z = int(np.abs(bounds[0][2] - bounds[1][2]) // chunk_size + 1)

        first_center_point = starting_point + chunk_size / 2.
        x_centers = np.array([first_center_point + [i * chunk_size, 0, 0] for i in range(n_x)])
        xy_centers = np.array([x_centers + [0, i * chunk_size, 0] for i in range(n_y)])
        xyz_centers = np.array([xy_centers + [0, 0, i * chunk_size] for i in range(n_z)])
        xyz_centers = np.reshape(xyz_centers, (-1, 1, 3))

        xyz_indices = np.array([[[(i, j, k) for i in range(n_x)] for j in range(n_y)] for k in range(n_z)])
        xyz_indices = np.reshape(xyz_indices, (-1, 3))

        plane_offsets = np.array([[1, 0, 0],
                                  [-1, 0, 0],
                                  [0, 1, 0],
                                  [0, -1, 0],
                                  [0, 0, 1],
                                  [0, 0, -1]]).reshape(1, 6, 3)
        plane_offsets = plane_offsets * (scale_margin * chunk_size / 2.)
        plane_origins = xyz_centers + plane_offsets
        cell_bounds = np.array([[np.array([x[1][0], x[3][1], x[5][2]]),
                                 np.array([x[0][0], x[2][1], x[4][2]])] for x in plane_origins])

        resulting_full_chunk_size_m = chunk_size * scale_margin
        furniture_pts_keys = list(furniture_pts.keys())

        for i, cell_bound in enumerate(cell_bounds):
            xyz_index = xyz_indices[i]
            index_filename = '_'.join([str(k) for k in xyz_index])

            if xyz_index[1] != 0:
                continue

            cell_center = (cell_bound[0] + cell_bound[1]) / 2.

            # --- 找出与当前 chunk 相交的家具（仅 AABB 碰撞检测，不加载 mesh）---
            chunk_world_aabb = cell_bound
            intersecting_indices = [
                int(key) for key in furniture_pts_keys
                if key in furniture_world_aabbs and aabb_intersect(furniture_world_aabbs[key], chunk_world_aabb)
            ]

            # --- 只加载相交的家具 mesh ---
            furniture_verts, furniture_faces = load_furniture_for_chunk(obj_id, intersecting_indices, valid_jids)

            # --- 合并场景 mesh + 当前 chunk 的家具 mesh ---
            if furniture_verts is not None:
                combined_vertices = np.vstack([SCENE_VERTS_ORIG, furniture_verts])
                combined_faces = np.vstack([SCENE_FACES, furniture_faces + len(SCENE_VERTS_ORIG)])
                del furniture_verts, furniture_faces
            else:
                combined_vertices = SCENE_VERTS_ORIG
                combined_faces = SCENE_FACES

            # --- 变换到归一化坐标 ---
            combined_vertices = put_scene_to_center(combined_vertices, cell_center, resulting_full_chunk_size_m)

            # --- 计算 SDF（复用全局 grid 避免重复分配）---
            sdf_tensor = mesh2sdf.core.compute(combined_vertices, combined_faces, num_voxels)

            # 合并 sdf 到全局 grid（避免重建 xyz_grid）
            xyzd = np.hstack([GLOBAL_XYZ_GRID_FLAT, sdf_tensor.reshape((-1, 1))])

            # 恢复 mesh 变换（保持一致性）
            combined_vertices = put_scene_back_from_center(combined_vertices, cell_center, resulting_full_chunk_size_m)
            del combined_vertices, combined_faces

            # --- 处理家具点（只加载当前 chunk 相交的家具的点）---
            all_furniture_pts = []
            all_furniture_pts_canonic = []
            all_furniture_sem_ids = []
            all_furniture_inst_ids = []
            id_inst = 1

            # 用 jids 迭代确保实例 ID 顺序与旧代码一致
            for key in jids:
                if str(key) not in furniture_pts:
                    id_inst += 1
                    continue
                if int(key) not in intersecting_indices:
                    id_inst += 1
                    continue
                pts = furniture_pts[str(key)][0]
                all_furniture_pts += pts
                if len(pts) <= 0:
                    all_furniture_pts_canonic += pts
                else:
                    all_furniture_pts_canonic += compute_points_canonic(pts).tolist()
                all_furniture_sem_ids += [cat_to_id[furniture_pts[str(key)][1]] for _ in range(len(pts))]
                all_furniture_inst_ids += [id_inst for _ in range(len(pts))]
                id_inst += 1

            all_furniture_pts = np.array(all_furniture_pts) if all_furniture_pts else np.empty((0, 3))
            all_furniture_pts_canonic = np.array(all_furniture_pts_canonic) if all_furniture_pts_canonic else np.empty((0, 3))
            all_furniture_sem_ids = np.array(all_furniture_sem_ids) if all_furniture_sem_ids else np.empty((0,))
            all_furniture_inst_ids = np.array(all_furniture_inst_ids) if all_furniture_inst_ids else np.empty((0,))

            # --- 转换点到 chunk 中心坐标并过滤 ---
            if len(all_scene_pts) > 0:
                scene_pts_transformed = put_scene_to_center(all_scene_pts[:, :3].copy(), cell_center, resulting_full_chunk_size_m)
                scene_chunk_mask = (
                    (-1.5 <= scene_pts_transformed[:, 0]) &
                    (-1.5 <= scene_pts_transformed[:, 1]) &
                    (-1.5 <= scene_pts_transformed[:, 2]) &
                    (1.5 >= scene_pts_transformed[:, 0]) &
                    (1.5 >= scene_pts_transformed[:, 1]) &
                    (1.5 >= scene_pts_transformed[:, 2])
                )
                scene_pts_chunk = all_scene_pts[scene_chunk_mask]
            else:
                scene_pts_chunk = np.empty((0, 3))
                scene_pts_transformed = np.empty((0, 3))

            if len(all_furniture_pts) > 0:
                furniture_pts_transformed = put_scene_to_center(all_furniture_pts.copy(), cell_center, resulting_full_chunk_size_m)
                furniture_chunk_mask = (
                    (-1.5 <= furniture_pts_transformed[:, 0]) &
                    (-1.5 <= furniture_pts_transformed[:, 1]) &
                    (-1.5 <= furniture_pts_transformed[:, 2]) &
                    (1.5 >= furniture_pts_transformed[:, 0]) &
                    (1.5 >= furniture_pts_transformed[:, 1]) &
                    (1.5 >= furniture_pts_transformed[:, 2])
                )
                furniture_pts_chunk = all_furniture_pts[furniture_chunk_mask]
                furniture_canonic_pts_chunk = all_furniture_pts_canonic[furniture_chunk_mask]
                furniture_sem_ids_chunk = all_furniture_sem_ids[furniture_chunk_mask]
                furniture_inst_ids_chunk = all_furniture_inst_ids[furniture_chunk_mask]
            else:
                furniture_pts_chunk = np.empty((0, 3))
                furniture_canonic_pts_chunk = np.empty((0, 3))
                furniture_sem_ids_chunk = np.empty((0,))
                furniture_inst_ids_chunk = np.empty((0,))
                furniture_pts_transformed = np.empty((0, 3))

            del scene_pts_transformed, furniture_pts_transformed

            # --- 截断 SDF ---
            inside_chunk_ids = GLOBAL_INSIDE_CHUNK_IDS
            xyz_grid_inside = GLOBAL_XYZ_GRID_FLAT[inside_chunk_ids]
            sdf_grid_inside = sdf_tensor.reshape((-1))[inside_chunk_ids]

            inside_chunk_furniture_ids = np.where(
                (furniture_pts_chunk.shape[0] > 0) &  # 避免空数组广播问题
                (furniture_pts_chunk[:, 0] >= -max_coord_abs) &
                (furniture_pts_chunk[:, 1] >= -max_coord_abs) &
                (furniture_pts_chunk[:, 2] >= -max_coord_abs) &
                (furniture_pts_chunk[:, 0] < max_coord_abs) &
                (furniture_pts_chunk[:, 1] < max_coord_abs) &
                (furniture_pts_chunk[:, 2] < max_coord_abs)
            )[0] if len(furniture_pts_chunk) > 0 else np.array([], dtype=int)
            flag_furniture = len(inside_chunk_furniture_ids) > 0

            non_truncation_ids = np.where(np.abs(sdf_grid_inside) <= trunc_dist)[0]
            xyz_grid_non_trunc = xyz_grid_inside[non_truncation_ids]
            sdf_grid_non_trunc = sdf_grid_inside[non_truncation_ids]

            truncation_ids = np.where(np.abs(sdf_grid_inside) > trunc_dist)[0]
            xyz_grid_trunc = xyz_grid_inside[truncation_ids]
            sdf_grid_trunc = sdf_grid_inside[truncation_ids]

            xyz_grid_non_trunc = put_scene_back_from_center(xyz_grid_non_trunc, cell_center, resulting_full_chunk_size_m)
            xyz_grid_trunc = put_scene_back_from_center(xyz_grid_trunc, cell_center, resulting_full_chunk_size_m)

            xyz_sdf = np.hstack([xyz_grid_non_trunc, sdf_grid_non_trunc[..., None]])
            xyz_sdf = xyz_sdf.astype('float32')

            xyz_sdf_trunc = np.hstack([xyz_grid_trunc, sdf_grid_trunc[..., None]])
            xyz_sdf_trunc[:, 3] = trunc_dist
            xyz_sdf_trunc = xyz_sdf_trunc.astype('float32')
            if len(xyz_sdf_trunc) > num_trunc_points:
                random_indices = np.random.choice(len(xyz_sdf_trunc), num_trunc_points, replace=False)
                xyz_sdf_trunc = xyz_sdf_trunc[random_indices]

            # --- 组装语义/实例标签 ---
            scene_pts_chunk = np.hstack([scene_pts_chunk, np.ones((len(scene_pts_chunk), 1)) * 1, np.zeros((len(scene_pts_chunk), 4))])

            if len(furniture_pts_chunk) > 0:
                furniture_pts_chunk = np.hstack([
                    furniture_pts_chunk,
                    furniture_sem_ids_chunk[:, None],
                    furniture_inst_ids_chunk[:, None],
                    furniture_canonic_pts_chunk
                ])
                all_pts_chunk = np.vstack([scene_pts_chunk, furniture_pts_chunk])
            else:
                all_pts_chunk = scene_pts_chunk

            # --- 体素语义/实例/Canonic 计算 ---
            try:
                non_trunc_points_sem = xyzd[:, :3][np.abs(xyzd[:, 3]) < 0.05]
                non_trunc_points_ids_sem = np.where(np.abs(xyzd[:, 3]) < 0.05)[0]

                if len(all_pts_chunk) > 0 and len(non_trunc_points_sem) > 0:
                    tree = cKDTree(all_pts_chunk[:, :3])
                    dists, idx = tree.query(non_trunc_points_sem)

                    voxel_distribution = all_pts_chunk[idx][:, 3:5]
                    all_voxel_distribution_sem = np.zeros(len(GLOBAL_XYZ_GRID_FLAT))
                    all_voxel_distribution_sem[non_trunc_points_ids_sem] = voxel_distribution[:, 0]
                    all_voxel_distribution_sem = all_voxel_distribution_sem.reshape(
                        (num_voxels, num_voxels, num_voxels)).astype('int16')

                    all_voxel_distribution_inst = np.zeros(len(GLOBAL_XYZ_GRID_FLAT))
                    all_voxel_distribution_inst[non_trunc_points_ids_sem] = voxel_distribution[:, 1]
                    all_voxel_distribution_inst = all_voxel_distribution_inst.reshape(
                        (num_voxels, num_voxels, num_voxels)).astype('int16')

                    canonic_coords = all_pts_chunk[idx][:, 5:]
                    all_canonic_distribution = np.zeros((len(GLOBAL_XYZ_GRID_FLAT), 3))
                    all_canonic_distribution[non_trunc_points_ids_sem] = canonic_coords[:]
                    all_canonic_distribution = (all_canonic_distribution.reshape(
                        (num_voxels, num_voxels, num_voxels, 3)) * 256).astype('int16')
                else:
                    all_voxel_distribution_sem = np.zeros((num_voxels, num_voxels, num_voxels), dtype='int16')
                    all_voxel_distribution_inst = np.zeros((num_voxels, num_voxels, num_voxels), dtype='int16')
                    all_canonic_distribution = np.zeros((num_voxels, num_voxels, num_voxels, 3), dtype='int16')

            except Exception:
                import traceback
                print(f"Voxel 语义计算或 cKDTree 崩溃, 场景: {obj_id}, Chunk: {index_filename}")
                traceback.print_exc()
                continue

            # --- 保存 ---
            if len(non_truncation_ids) != 0:
                meta_data = {
                    'chink_size': chunk_size,
                    'scale_margin': scale_margin,
                    'max_coord_abs': max_coord_abs,
                    'starting_point': starting_point.tolist(),
                    'num_voxels': num_voxels,
                    'trunc_dist': trunc_dist,
                    'furniture': flag_furniture,
                    'cell_center': cell_center.tolist(),
                    'cell_scale': resulting_full_chunk_size_m,
                    'subchunk_coords': cell_bound.tolist()
                }
                with open(os.path.join(LOCAL_SAVEDIR, f'{index_filename}.json'), 'w') as fout:
                    json.dump(meta_data, fout)

                # 保存时转 float16 节省磁盘
                sdf_save = sdf_tensor.astype('float16')
                np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}.npy'), sdf_save)
                np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_semantic.npy'), all_voxel_distribution_sem)
                np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_instance.npy'), all_voxel_distribution_inst)
                np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_canonic.npy'), all_canonic_distribution)

            # --- 关键：释放当前 chunk 的所有中间数组 ---
            del sdf_tensor, xyzd, xyz_grid_inside, sdf_grid_inside
            del xyz_grid_non_trunc, sdf_grid_non_trunc, xyz_grid_trunc, sdf_grid_trunc
            del xyz_sdf, xyz_sdf_trunc, all_pts_chunk, scene_pts_chunk
            del all_furniture_pts, all_furniture_pts_canonic, all_furniture_sem_ids, all_furniture_inst_ids
            del furniture_pts_chunk, furniture_canonic_pts_chunk, furniture_sem_ids_chunk, furniture_inst_ids_chunk
            if 'tree' in dir():
                del tree
            if 'all_voxel_distribution_sem' in dir():
                del all_voxel_distribution_sem, all_voxel_distribution_inst, all_canonic_distribution
            gc.collect()

        # --- 场景级清理 ---
        del SCENE_VERTS_ORIG, SCENE_FACES, valid_jids, jids
        del all_scene_pts, furniture_pts, furniture_world_aabbs, furniture_pts_keys
        gc.collect()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    compute_chunks(args.num_proc, args.proc)
