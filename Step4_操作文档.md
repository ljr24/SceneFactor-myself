# Step 4 操作文档：SDF 体素计算

> **目标**：将 3D-FRONT 场景数据转换为按块分割的 SDF 体素格式
> **使用代码**：`data_processing/compute_sdf_3dfront_hjm.py`（团队优化版）
> **执行位置**：本地机器（非 Slurm 集群）

---

## 一、前置条件

- [x] 项目已解压到 `/home/lijiarui/Desktop/scene_factor/SceneFactor/`
- [x] Conda 环境 `scenefactor` 已创建并通过验证
- [ ] 数据集已确认：
  - `assets/SELECTED_FRONT_SCENES/` — 50 个场景
  - `assets/SELECTED_FUTURE_MODELS/` — 937 个家具模型

---

## 二、修改代码

编辑 `data_processing/compute_sdf_3dfront_hjm.py`，共 4 处修改：

### 1. 修改路径（第 11、13、14 行）

| 变量 | 原值 | 改为 |
|:-----|:-----|:-----|
| `FRONT3DMESHESMANIFOLD` | `/mnt/d/Datasets/3D_FRONT/SAVEDIR` | `assets/SELECTED_FRONT_SCENES` |
| `FUTURE3DMODEL` | `/mnt/d/Datasets/3D_FUTURE/Models` | `assets/SELECTED_FUTURE_MODELS` |
| `SAVEDIR` | `/mnt/d/Datasets/chunked_data_lowres` | `assets/chunked_data_lowres` |

### 2. 修复内存保护逻辑（第 143-146 行）

**原代码（有 bug）**：
```python
if len(os.listdir(os.path.join(FRONT3DMESHESMANIFOLD, obj_id))) >= 50:
    print(f"{obj_id} : Scene with {len(all_mesh_furniture)} objs. Memory overflow protection.")
    continue
```

**改为**：
```python
with open(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'furniture_jids.json')) as f:
    num_furniture = len(json.load(f))
if num_furniture >= 100:
    print(f"{obj_id} : {num_furniture} furniture items. Skipping memory overflow protection.")
    continue
```

> 说明：原代码用 `os.listdir` 数文件总数来判断，且引用了未定义的变量。改为用 `furniture_jids.json` 中的家具数量判断，上限 100。你的数据集家具数最多 93，一个都不会跳过。

---

## 三、准备 valid_scenes.json

该文件用于过滤需要处理的场景。如果 `data_processing/` 下没有，或需要更新：

```bash
cd /home/lijiarui/Desktop/scene_factor/SceneFactor

conda run -n scenefactor python3 -c "
import json, os
scenes = [d for d in os.listdir('assets/SELECTED_FRONT_SCENES') 
          if os.path.isdir(f'assets/SELECTED_FRONT_SCENES/{d}')]
with open('data_processing/valid_scenes.json', 'w') as f:
    json.dump({'valid_scenes': scenes}, f)
print(f'valid_scenes.json 已创建，包含 {len(scenes)} 个场景')
"
```

---

## 四、运行

```bash
cd /home/lijiarui/Desktop/scene_factor/SceneFactor/data_processing

conda run -n scenefactor python -u compute_sdf_3dfront_hjm.py -n 1 -p 0
```

**参数说明**：
| 参数 | 含义 | 示例 |
|:-----|:-----|:-----|
| `-n` | 总进程数 | `-n 1` 单进程, `-n 4` 四进程 |
| `-p` | 当前进程 ID（从 0 开始） | `-p 0`, `-p 1` 等 |

**单进程 vs 多进程**：
```bash
# 单进程（跑全部 50 个场景）
conda run -n scenefactor python -u compute_sdf_3dfront_hjm.py -n 1 -p 0

# 多进程（开 4 个终端分别跑，各处理不同场景）
conda run -n scenefactor python -u compute_sdf_3dfront_hjm.py -n 4 -p 0
conda run -n scenefactor python -u compute_sdf_3dfront_hjm.py -n 4 -p 1
conda run -n scenefactor python -u compute_sdf_3dfront_hjm.py -n 4 -p 2
conda run -n scenefactor python -u compute_sdf_3dfront_hjm.py -n 4 -p 3
```

---

## 五、输出与验证

### 输出目录结构

```
assets/chunked_data_lowres/<scene_uuid>/
├── {x}_0_{z}.npy              ← SDF 张量 (90×90×90, float16)
├── {x}_0_{z}_semantic.npy     ← 语义类别 ID (90×90×90, int16)
├── {x}_0_{z}_instance.npy     ← 实例 ID (90×90×90, int16)
├── {x}_0_{z}_canonic.npy      ← 正则化坐标 (90×90×90×3, int16)
└── {x}_0_{z}.json              ← chunk 元数据
```

### 验证命令

```bash
# 查看生成了哪些场景
ls /home/lijiarui/Desktop/scene_factor/SceneFactor/assets/chunked_data_lowres/

# 查看某个场景的 chunk 数量
ls /home/lijiarui/Desktop/scene_factor/SceneFactor/assets/chunked_data_lowres/<scene_uuid>/ | head

# 运行完整性检查
conda run -n scenefactor python3 data_processing/check_chunks.py
```

---

## 六、运行中日志解读

运行时会输出类似以下信息：

```
100%|████████████████████| 50/50 [01:23<00:00]
  场景1 : Loading scene obj failed.      ← scene.obj 加载失败，一般跳过即可
  场景2 : furniture_points.json not found.  ← 该场景缺少点云数据
  场景3 : Voxel 语义计算或 cKDTree 崩溃    ← 单个 chunk 出错，不影响其他 chunk
  场景4 : Scene with >=100 items. Skipping  ← 家具超上限被跳过（上限100时不会出现）
```

大部分错误是单个场景/单个 chunk 的问题，脚本会 `continue` 继续处理下一个。

---

## 七、注意事项

1. **运行时间**：每个场景几分钟到十几分钟不等，50 个场景可能需要数小时
2. **内存**：你的 60GB 内存 + 48GB 显存足够，如遇 OOM 可降低内存保护上限到 80
3. **无法恢复中断**：脚本对有输出的场景会跳过（第 138-139 行），但正在处理的场景如果中断需要重跑
4. **只处理 y=0 层**：第 239 行 `if xyz_index[1] != 0: continue`，只处理 Y 方向第一层 chunk
