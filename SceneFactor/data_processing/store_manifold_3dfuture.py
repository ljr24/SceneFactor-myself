import os, sys
from tqdm import tqdm


FUTURE3DMESHES = '/home/ps/Dataset/3D-FUTURE-model/3D-FUTURE-model'
SAVEDIR = '/mnt/d/Datasets/3D_FUTURE/Models' 

# Choose Manifold or ManifoldPlus version (Manifold is preferred)
#manifold = '/home/ps/Git/Manifold/build/manifold'
manifold = '/home/ps/Git/ManifoldPlus/build/manifold'
# manifold = '/home/bohovkin/cluster/abokhovkin_home/projects/ManifoldPlus/build/manifold'

for obj_id in tqdm(os.listdir(FUTURE3DMESHES)):
    LOCAL_SAVEDIR = os.path.join(SAVEDIR, obj_id)

        # 检验文件夹是否存在的代码
    if os.path.exists(LOCAL_SAVEDIR):
        # 如果你只想知道它存在，可以 print 一下
        # print(f"Skip: {obj_id} already exists.")
        continue # 跳过当前循环，不执行后续的 os.system

    os.makedirs(LOCAL_SAVEDIR, exist_ok=True)

    os.system(f"{manifold} --input {os.path.join(FUTURE3DMESHES, obj_id, 'raw_model.obj')} --output {os.path.join(LOCAL_SAVEDIR, 'raw_model.obj')} --depth 9")