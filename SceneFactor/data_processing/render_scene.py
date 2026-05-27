import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 你的路径（和之前一样）
SAVE_DIR = "/mnt/d/Datasets/3D_FRONT/SAVEDIR"

def render_scene_pointcloud(scene_name):
    scene_path = os.path.join(SAVE_DIR, scene_name)
    print(f"渲染: {scene_name}")

    # 1. 加载场景点云
    scene_points_path = os.path.join(scene_path, "scene_points.json")
    if not os.path.exists(scene_points_path):
        print(f"无 scene_points.json: {scene_name}")
        return
    
    with open(scene_points_path, 'r') as f:
        scene_data = json.load(f)
    scene_pts = np.array(scene_data['0'])

    # 2. 加载家具点云
    furniture_points_path = os.path.join(scene_path, "furniture_points.json")
    if not os.path.exists(furniture_points_path):
        print(f"无 furniture_points.json: {scene_name}")
        return
    
    with open(furniture_points_path, 'r') as f:
        furniture_data = json.load(f)

    # 3. 绘制 3D 点云
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制场景（灰色）
    ax.scatter(
        scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
        c='gray', s=0.2, alpha=0.4, label='Scene'
    )

    # 绘制家具（蓝色）
    for obj_id, (pts, cat) in furniture_data.items():
        pts = np.array(pts)
        if len(pts) == 0:
            continue
        
        color = 'orange' if cat == 'Cabinet/Shelf/Desk' else 'deepskyblue'
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, s=0.5, alpha=0.8
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Scene: {scene_name}')
    ax.legend()
    plt.tight_layout()

    # 保存图片（不打开窗口！）
    save_img = os.path.join(scene_path, "render_pointcloud.png")
    plt.savefig(save_img, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 保存渲染图: {save_img}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str)
    args = parser.parse_args()

    scenes = sorted(os.listdir(SAVE_DIR))
    
    if args.scene:
        render_scene_pointcloud(args.scene)
    else:
        for s in scenes:
            render_scene_pointcloud(s)