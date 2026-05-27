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
  场景4 : Scene with >=100 items. Skipping  ← 家具超上限被跳过
```

大部分错误是单个场景/单个 chunk 的问题，脚本会 `continue` 继续处理下一个。

---

## 七、注意事项

1. **运行时间**：每个场景几分钟到十几分钟不等，50 个场景可能需要数小时
2. **内存**：你的 60GB 内存 + 48GB 显存足够，如遇 OOM 可降低内存保护上限
3. **无法恢复中断**：脚本对有输出的场景会跳过（第 138-139 行），但正在处理的场景如果中断需要重跑
4. **只处理 y=0 层**：第 239 行 `if xyz_index[1] != 0: continue`，只处理 Y 方向第一层 chunk

---

## 八、Git 提交（完整速查）

> 以下内容整合自 `Git指令速查.md`，覆盖项目协作全流程。

### 8.1 首次配置

```bash
# 设置身份
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"

# SSH 免密连接 GitHub
ssh-keygen -t ed25519 -C "你的邮箱"
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
```

### 8.2 忽略大数据目录

```bash
cd /home/lijiarui/Desktop/scene_factor/SceneFactor
echo "assets/" >> .gitignore
git status
```

### 8.3 本地仓库操作

```bash
git status                  # 查看当前文件状态
git add <文件名>             # 暂存某个文件
git add .                   # 暂存所有变更
git commit -m "提交说明"     # 提交已暂存的文件
git commit -am "说明"        # 跳过 add 直接提交
git log --oneline           # 查看提交历史
```

### 8.4 远程仓库操作

```bash
git remote -v                              # 查看远程地址
git remote add origin 地址                  # 关联远程仓库
git push -u origin main                    # 首次推送
git push                                    # 后续推送
git pull --rebase                           # 拉取最新代码
```

### 8.5 分支操作

```bash
git checkout -b <分支名>       # 创建并切换到新分支
git checkout <分支名>          # 切换分支
git merge <分支名>             # 合并分支到当前
git branch -d <分支名>         # 删除已合并的分支
```

### 8.6 撤销与回退

```bash
git restore <文件名>           # 撤销工作区修改
git restore --staged <文件名>  # 取消暂存
git reset --soft HEAD~1       # 撤销上次 commit，保留修改
git reset --hard HEAD~1       # 撤销上次 commit，不保留修改（慎用）
```

### 8.7 完整工作流

```bash
cd SceneFactor
git pull --rebase
# 修改代码...
git add .
git commit -m "feat: 说明"
git push
```

### 推到导师项目

```bash
git remote add team git@github.com:AkiraJM/SceneFactor.git
git push team main
```

### 8.8 常见报错

| 报错 | 原因 | 解决 |
|:-----|:------|:------|
| `Please tell me who you are` | 没配用户名邮箱 | `git config --global user.name/email` |
| `Permission denied (publickey)` | SSH Key 没配好 | `ssh-keygen` + 加到 GitHub |
| `Repository not found` | 仓库不存在/没权限 | 检查地址 + 问导师 |
| `failed to push` | 远程有本地没有的文件 | `git pull --rebase` 再重试 |

### 在提交前：忽略大数据目录

`assets/` 下是数据集（~73GB），不要提交到 Git。先配置 `.gitignore`：

```bash
cd /home/lijiarui/Desktop/scene_factor/SceneFactor
echo "assets/" >> .gitignore
```

确认一下：

```bash
git status          # assets/ 不应该出现在列表里
```

### 日常提交流程

```bash
# 1. 查看当前改了什么
git status

# 2. 暂存修改
git add <文件名>       # 暂存具体文件
git add .             # 暂存所有变更（不包括 .gitignore 忽略的）

# 3. 提交到本地
git commit -m "feat: 修改内容说明"

# 4. 推送到远程
git push
```

> 第一次推送需关联上游分支：`git push -u origin main`

### 推送前先拉取

```bash
git pull --rebase     # 拉取远程最新代码，避免冲突
```

### 完整流程图

```
cd SceneFactor
echo "assets/" >> .gitignore                    ← 忽略数据集
git add .                                       ← 暂存代码修改
git commit -m "feat: 配置 Step 4，修复内存逻辑"    ← 本地提交
git pull --rebase                               ← 拉取远程最新
git push                                        ← 推送到 GitHub
```

### Git 配置速查

| 场景 | 命令 |
|:-----|:------|
| 首次用 Git | `git config --global user.name "名字"` |
| | `git config --global user.email "邮箱"` |
| 配置 SSH 免密 | `ssh-keygen -t ed25519` → 公钥贴到 GitHub |
| 查看远程地址 | `git remote -v` |
| 切分支 | `git checkout -b 分支名` |
| 撤销工作区修改 | `git restore 文件名` |
| 取消暂存 | `git restore --staged 文件名` |
| 废弃最近提交 | `git reset --soft HEAD~1`（保留修改） |
