# SceneFactor: 分解式潜空间3D扩散，实现可控3D场景生成
### [项目主页](https://alexeybokhovkin.github.io/scenefactor/) | [论文](http://arxiv.org/abs/2412.01801) | [视频](https://youtu.be/wZqX09IFveA)

官方 PyTorch 实现

![pipeline](assets/teaser.jpg)

我们提出 SceneFactor，一种基于扩散的大规模 3D 场景生成方法，支持可控生成和便捷编辑。

SceneFactor 通过我们的分解式扩散公式实现文本引导的 3D 场景合成，利用潜空间中的语义和几何流形生成任意大小的 3D 场景。虽然文本输入可以实现轻松可控的生成，但对于直观的局部编辑和操作而言，文本引导仍然不够精确。

我们的分解式语义扩散生成一个由语义 3D 盒子组成的代理语义空间，通过添加、移除、改变语义 3D 代理盒子的大小来实现对生成场景的可控编辑，并以此引导高保真、一致的 3D 几何编辑。大量实验表明，我们的方法通过分解式扩散方法实现了高保真的 3D 场景合成和有效的可控编辑。

<br>

https://github.com/user-attachments/assets/ca466d81-c6a4-4975-a768-bbe23be71e27


## 方法概述

![training](assets/method.jpg)


## 依赖环境

本项目使用 CUDA 11.7、PyTorch 1.13.0 和 PyTorch Lightning 1.8.0。我们附带了 Conda 环境文件，可通过以下方式安装：
```commandline
conda env create -f environment.yml
```

此外，数据处理需要 [Manifold](https://github.com/hjwdzh/Manifold)、[ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus)。

最后，安装以下包：

```commandline
pip install mesh2sdf
pip install transformers
pip install einops
pip install rotary-embedding-torch
pip install prettytable
```


## 数据准备

我们的方法使用了 3D-FRONT 和 3D-FUTURE 数据集，将它们组合并处理成带有文字标注的块数据。请先下载这些数据集，再运行以下数据处理脚本。

### 3D-FUTURE 和 3D-FRONT 初始处理（此阶段已完成）

1. 在 ```data_processing/store_manifold_3dfuture.py``` 中设置 3D-FUTURE 数据集路径。使用 Slurm 运行 bash 脚本 ```data_processing/store_manifold_3dfuture.sh```，或修改后用 bash 直接运行。该脚本使用 Manifold 处理 3D-FUTURE 网格，修复非流形网格。

2. [可选但推荐] 在 ```data_processing/fix_3dfuture_meshes.py``` 中设置 3D-FUTURE 数据集路径。使用 Slurm 运行 bash 脚本 ```data_processing/fix_3dfuture_meshes.sh```，或修改后用 bash 直接运行。该脚本简化特定 3D-FUTURE 网格并填充孔洞。

3. 在 ```data_processing/store_3dfuturefront.py``` 中设置 3D-FUTURE（步骤1和2之后）和 3D-FRONT 数据集的路径。使用 Slurm 运行 bash 脚本 ```data_processing/store_3dfuturefront.sh```，或修改后用 bash 直接运行。该脚本组装 3D-FRONT 和处理后的 3D-FUTURE 数据集，并存储每个场景的最终网格几何体。（优化了预处理代码，将重复储存的家具文件改成仅储存转换矩阵的形式，缓解了原代码所需要消耗大量储存空间和内存的情况，具体见 store_3dfuturefront.py 中的注释部分）

### 训练/推理几何数据处理（此阶段正在进行中）

4. 在 ```data_processing/compute_sdf_3dfront.py``` 中设置处理后的 3D-FRONT 数据集路径（步骤3之后）。使用 Slurm 运行 bash 脚本 ```data_processing/compute_sdf_3dfront.sh```，或修改后用 bash 直接运行。该脚本将处理后的 3D-FRONT 数据集存储为按块分割的 SDF 体素格式。（由于上述第3步的优化将储存文件的形式改变了，这部分的代码使用的是 compute_sdf_3dfront_hjm.py）（将 compute_sdf_3dfront_hjm.py 代码中的 FRONT3DMESHESMANIFOLD 变量改成下载的 SELECTED_FRONT_SCENES 文件夹的路径，FUTURE3DMODEL 变量改成 SELECTED_FUTURE_MODELS 文件夹的路径）**（此部分的代码存在大幅优化内存的可能性，目前版本可能内存无法支持运行完整代码）**

5. 在 ```data_processing/compute_train_chunks_vox_lowres_2x2_sem.py``` 中设置 SDF 体素 3D-FRONT 数据集路径（步骤4之后）。使用 Slurm 运行 bash 脚本 ```data_processing/compute_train_chunks_vox_lowres_2x2_sem.sh```，或修改后用 bash 直接运行。该脚本从上一步处理的数据中随机采样块，用于训练几何和语义 VQ-VAE。

6. 在 ```data_processing/compute_inference_chunks.py``` 中设置处理后的 3D-FRONT 数据集路径（步骤3之后）。使用 Slurm 运行 bash 脚本 ```data_processing/compute_inference_chunks.sh```，或修改后用 bash 直接运行。该脚本为每个场景生成用于测试/推理的块。

### 训练/推理文本数据处理

7. 在 ```data_processing/compute_captions.py``` 中设置处理后的 3D-FRONT 数据集路径（步骤3之后）、3D-FRONT 元数据文件夹路径以及步骤5（训练）或步骤6（推理）生成的块文件夹路径。使用 Slurm 运行 bash 脚本 ```data_processing/compute_captions.sh```，或修改后用 bash 直接运行。该脚本为训练和推理块计算简单的合成标注文本。

8. 在 ```data_processing/compute_captions_spatial.py``` 中设置处理后的 3D-FRONT 数据集路径（步骤3之后）、3D-FRONT 元数据文件夹路径以及步骤5（训练）或步骤6（推理）生成的块文件夹路径。使用 Slurm 运行 bash 脚本 ```data_processing/compute_captions_spatial.sh```，或修改后用 bash 直接运行。该脚本为训练和推理块计算带空间关系的合成标注文本。

9. 在 ```data_processing/compute_all_captions_qwen_train.py``` 中设置步骤5（训练）或步骤6（推理）生成的块文件夹路径。使用 Slurm 运行 bash 脚本 ```data_processing/compute_all_captions_qwen_train.sh```，或修改后用 bash 直接运行。该脚本使用 Qwen1.4 为步骤7和8生成的标注文本生成更自然的版本。


## 训练与实验

训练流程包含 4 个步骤。首先，我们训练两个自编码器，将几何和语义数据压缩到 3D 潜空间中。然后在此潜空间流形之上训练语义和几何扩散模型。在推理阶段，我们的方法支持逐块和大规模场景合成。每个阶段（VQ-VAE 模型训练/推理、扩散模型训练/推理）都需要创建一个实验文件夹，将对应的 ```specs.json``` 文件（类似于 ```configs``` 中的文件）放入该文件夹，并在启动脚本中指定该文件夹的路径。

### 训练语义和几何 VQ-VAE

1. 使用 ```configs/specs_sem_vqvae.json``` 训练语义 VQ-VAE 模型，将粗略的语义地图压缩为潜空间网格。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch train_modulation.sh
```
在 ```train_modulation.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -b <批大小> -w <工作进程数>```。

2. 使用 ```configs/specs_geo_vqvae_onestage.json``` 训练几何 VQ-VAE 模型，将 SDF 块网格压缩为潜空间网格。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch train_modulation.sh
```
在 ```train_modulation.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -b <批大小> -w <工作进程数>```。

#### 替代方案：几何 VQ-VAE 两阶段训练

2a. 使用 ```configs/specs_geo_vqvae_stage1.json``` 训练第一个几何 VQ-VAE 模型，将高分辨率 SDF 块网格压缩为低分辨率潜空间网格。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch train_modulation.sh
```
在 ```train_modulation.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -b <批大小> -w <工作进程数>```。

2b. 然后使用 ```configs/specs_geo_vqvae_stage2.json``` 训练第二个几何 VQ-VAE 模型，将高分辨率 SDF 块网格压缩为高分辨率潜空间网格。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch train_modulation.sh
```
在 ```train_modulation.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -b <批大小> -w <工作进程数>```。

### 训练语义和几何扩散模型

3. 使用 ```configs/specs_sem_diff.json``` 训练语义扩散模型。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch train_diffusion_ddp.sh
```
在 ```train_diffusion_ddp.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -b <批大小> -w <工作进程数> -r <可选：检查点轮次数>```。

4. 使用 ```configs/specs_geo_diff_onestage.json``` 训练几何扩散模型。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch train_diffusion_ddp.sh
```
在 ```train_diffusion_ddp.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -b <批大小> -w <工作进程数> -r <可选：检查点轮次数>```。

#### 替代方案：几何扩散两阶段训练（如果使用了步骤 1a/1b）

4a. 使用 ```configs/specs_geo_diff_stage1.json``` 训练第一个几何扩散模型。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch train_diffusion_ddp.sh
```
在 ```train_diffusion_ddp.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -b <批大小> -w <工作进程数> -r <可选：检查点轮次数>```。

4b. 然后使用 ```configs/specs_geo_diff_stage2.json``` 训练第二个几何扩散模型。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch train_diffusion_ddp.sh
```
在 ```train_diffusion_ddp.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -b <批大小> -w <工作进程数> -r <可选：检查点轮次数>```。

### 场景块生成

5. 使用 ```configs/specs_sem_infer.json```（指定训练好的语义 VQ-VAE 模型文件夹和对应的语义扩散模型文件夹）从文本输入启动粗略语义块的推理。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_separate_sem.sh
```
在 ```inference_separate_sem.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -s <可选：并行推理时的分割索引> -d <生成结果保存文件夹>```。

6. 使用 ```configs/specs_geo_infer_stage1_onestage.json```（指定训练好的几何 VQ-VAE 模型文件夹、对应的几何扩散模型文件夹以及生成的语义块文件夹）启动几何块的推理。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_separate_geo.sh
```
在 ```inference_separate_geo.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -d <语义块文件夹> -s <生成结果保存文件夹> -n <块生成进程数(GPU数)> -p <进程ID>```。

#### 替代方案：几何扩散生成（如果使用了步骤 4a/4b）

6a. 使用 ```configs/specs_geo_infer_stage1_onestage.json```（指定训练好的几何 VQ-VAE 模型文件夹、对应的几何扩散模型文件夹以及生成的语义块文件夹）启动几何块的推理。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_separate_geo.sh
```
在 ```inference_separate_geo.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -d <语义块文件夹> -s <生成结果保存文件夹> -n <块生成进程数(GPU数)> -p <进程ID>```。

6b. 使用 ```configs/specs_geo_infer_stage2.json```（指定训练好的几何 VQ-VAE 模型文件夹、对应的精炼几何扩散模型文件夹以及生成的几何块文件夹）启动精炼几何块的推理。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_separate_geo_stage2.sh
```
在 ```inference_separate_geo_stage2.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -d <生成的几何块文件夹> -n <块生成进程数(GPU数)> -p <进程ID>```。

### 大规模场景生成

7. 使用 ```configs/specs_sem_infer.json```（指定训练好的语义 VQ-VAE 模型文件夹和对应的语义扩散模型文件夹）从文本输入启动粗略语义场景的推理（逐块）。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_text_to_sem.sh
```
在 ```inference_text_to_sem.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -c <场景ID到标注文本类型的JSON映射文件路径> -s <生成结果保存文件夹> -n <块生成进程数(GPU数)> -p <进程ID>```。

8. 使用 ```configs/specs_geo_infer_stage1_onestage.json```（指定训练好的几何 VQ-VAE 模型文件夹、对应的几何扩散模型文件夹以及生成的语义场景文件夹）启动几何场景的推理。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_sem_to_geo_overlap_stage1_onestage.sh
```
在 ```inference_sem_to_geo_overlap_stage1_onestage.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -s <生成结果保存文件夹> -t <要处理的场景ID文本文件路径> -n <块生成进程数(GPU数)> -p <进程ID>```。

#### 替代方案：几何扩散生成（如果使用了步骤 6a/6b）

8a. 使用 ```configs/specs_geo_infer_stage1_onestage.json```（指定训练好的几何 VQ-VAE 模型文件夹、对应的几何扩散模型文件夹以及生成的语义场景文件夹）启动几何场景的推理。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_sem_to_geo_overlap_stage1_onestage.sh
```
在 ```inference_sem_to_geo_overlap_stage1_onestage.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -s <生成结果保存文件夹> -t <要处理的场景ID文本文件路径> -n <块生成进程数(GPU数)> -p <进程ID>```。

8b. 使用 ```configs/specs_geo_infer_stage2.json```（指定训练好的几何 VQ-VAE 模型文件夹、对应的精炼几何扩散模型文件夹以及生成的几何场景文件夹）启动精炼几何场景的推理。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_sem_to_geo_nooverlap_stage2.sh
```
在 ```inference_sem_to_geo_nooverlap_stage2.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -s <生成结果保存文件夹> -n <块生成进程数(GPU数)> -p <进程ID>```。

### 编辑实验

9. 使用 ```configs/specs_geo_infer_stage1_onestage.json```（指定训练好的几何 VQ-VAE 模型文件夹、对应的精炼几何扩散模型文件夹以及生成的几何场景文件夹）启动精炼几何场景的推理。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_sem_to_geo_overlap_edit_stage1_onestage.sh
```
编辑前先执行步骤7和8生成语义和几何场景。编辑推理需要一个 ```edit_scenes_folder``` 文件夹，其中包含场景与编辑操作组合对应的子文件夹（添加/移除/替换/移动/缩放）。每个子文件夹应包含 ```full_sem.npy```（场景原始语义地图）、```full_sem_edited.npy```（场景编辑后的语义地图）、```full_sem_edited.json```（指定编辑坐标的编辑操作标注）、```full_lowres_latent.npy```（原始几何3D潜空间网格）。在 ```inference_sem_to_geo_overlap_edit_stage1_onestage.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -d <要编辑的场景目录，每个场景包含原始语义图、编辑后语义图、编辑标注、原始几何潜空间网格> -k <每个场景的保存索引>```。

#### 替代方案：编辑（如果使用了步骤 8a/8b）

9a. 使用 ```configs/specs_geo_infer_stage1_onestage.json```（指定训练好的几何 VQ-VAE 模型文件夹和对应的精炼几何扩散模型文件夹）。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_sem_to_geo_overlap_edit_stage1_onestage.sh
```
编辑前先执行步骤7、8a、8b生成语义和几何场景。编辑推理需要一个 ```edit_scenes_folder``` 文件夹（类似步骤9）。在 ```inference_sem_to_geo_overlap_edit_stage1_onestage.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -d <要编辑的场景目录，每个场景包含原始语义图、编辑后语义图、编辑标注、原始几何潜空间网格> -k <每个场景的保存索引>```。

9b. 使用 ```configs/specs_geo_infer_stage2.json```（指定训练好的几何 VQ-VAE 模型文件夹和对应的精炼几何扩散模型文件夹）。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_sem_to_geo_nooverlap_edit_stage2.sh
```
编辑前先执行步骤7、8a、8b、9a生成语义和几何场景并编辑第一个几何地图。编辑推理需要一个 ```edit_scenes_folder``` 文件夹（类似步骤9a）。每个子文件夹应包含 ```full_lowres_latent.npy```（用于条件的精炼前原始几何3D潜空间网格）、```full_room_latent_lowres_edited.npy```（步骤9a后的编辑后几何3D潜空间网格，用作条件）、```full_sem_edited.json```（指定编辑坐标的编辑操作标注）、```full_latent.npy```（精炼场景的原始几何3D潜空间网格）。在 ```inference_sem_to_geo_nooverlap_edit_stage2.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -d <要编辑的场景目录> -k <每个场景的保存索引>```。

9c. [9b 的替代方案] 使用 ```configs/specs_geo_infer_stage2.json```（指定训练好的几何 VQ-VAE 模型文件夹和对应的精炼几何扩散模型文件夹）。然后使用 Slurm 运行或修改后用 bash 直接运行：
```commandline
sbatch inference_sem_to_geo_nooverlap_edit_stage2.sh
```
编辑前先执行步骤7、8a、8b、9a生成语义和几何场景并编辑第一个几何地图。编辑推理需要一个 ```edit_scenes_folder``` 文件夹（类似步骤9b）。此版本较慢，因为使用带重叠的滑动窗口选项。在 ```inference_sem_to_geo_nooverlap_edit_stage2.sh``` 中指定 ```-e <实验文件夹路径，包含specs.json> -d <要编辑的场景目录> -k <每个场景的保存索引>```。


## 预训练检查点

我们提供两种几何生成选项的预训练检查点。

### 单阶段几何方案（选项1）

#### VQ-VAE 检查点

- [语义 VQ-VAE](https://kaldir.vc.in.tum.de/scenefactor/sem_vqvae.ckpt)
- [几何 VQ-VAE](https://kaldir.vc.in.tum.de/scenefactor/geo_vqvae_onestage.ckpt)

#### 扩散检查点

- [语义扩散条件编码器](https://kaldir.vc.in.tum.de/scenefactor/sem_diff_encoder.ckpt)
- [语义潜扩散模型](https://kaldir.vc.in.tum.de/scenefactor/sem_diff_main.ckpt)
- [几何扩散条件编码器](https://kaldir.vc.in.tum.de/scenefactor/geo_diff_encoder_onestage.ckpt)
- [几何潜扩散模型](https://kaldir.vc.in.tum.de/scenefactor/geo_diff_main_onestage.ckpt)

### 两阶段几何方案（选项2）

#### VQ-VAE 检查点

- [语义 VQ-VAE](https://kaldir.vc.in.tum.de/scenefactor/sem_vqvae.ckpt)
- [几何 VQ-VAE, 阶段1](https://kaldir.vc.in.tum.de/scenefactor/geo_vqvae_stage1.ckpt)
- [几何 VQ-VAE, 阶段2](https://kaldir.vc.in.tum.de/scenefactor/geo_vqvae_stage2.ckpt)

#### 扩散检查点

- [语义扩散条件编码器](https://kaldir.vc.in.tum.de/scenefactor/sem_diff_encoder.ckpt)
- [语义潜扩散模型](https://kaldir.vc.in.tum.de/scenefactor/sem_diff_main.ckpt)
- [几何扩散条件编码器, 阶段1](https://kaldir.vc.in.tum.de/scenefactor/geo_diff_encoder_stage1.ckpt)
- [几何潜扩散模型, 阶段1](https://kaldir.vc.in.tum.de/scenefactor/geo_diff_main_stage1.ckpt)
- [几何扩散条件编码器, 阶段2](https://kaldir.vc.in.tum.de/scenefactor/geo_diff_encoder_stage2.ckpt)
- [几何潜扩散模型, 阶段2](https://kaldir.vc.in.tum.de/scenefactor/geo_diff_main_stage2.ckpt)


## 引用

如果您觉得我们的工作对您的研究有帮助，请考虑引用：

	@misc{bokhovkin2024scenefactor,
		title={SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation}, 
		author={Bokhovkin, Alexey and Meng, Quan and Tulsiani, Shubham and Dai, Angela},
		journal={arXiv preprint arXiv:2412.01801},
		year={2024}
	}
