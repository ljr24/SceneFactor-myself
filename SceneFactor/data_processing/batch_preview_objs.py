import pyvista as pv
import os
import glob
import sys

class OBJBatchViewer:
    def __init__(self, root_dir, file_pattern="scene.obj"):
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        
        # 1. 递归搜索所有匹配的文件
        search_path = os.path.join(root_dir, "**", file_pattern)
        print(f"正在搜索目录下所有的 {file_pattern} ... (这可能需要一些时间)")
        self.file_list = sorted(glob.glob(search_path, recursive=True))
        self.total_count = len(self.file_list)
        
        if self.total_count == 0:
            print(f"Error: 在 {root_dir} 中找不到任何 {file_pattern}")
            sys.exit(1)
            
        print(f"共找到 {self.total_count} 个模型。")
        self.current_index = 0
        
        # 2. 初始化 Plotter
        self.plotter = pv.Plotter(title="3D-FRONT OBJ Batch Viewer")
        
        # 设置键盘事件监听
        self.plotter.add_key_event("Right", self.next_mesh)
        self.plotter.add_key_event("Left", self.prev_mesh)
        self.plotter.add_key_event("q", self.plotter.close) # 按 Q 退出
        
        # 添加辅助组件
        self.plotter.add_axes()
        self.plotter.set_background("royalblue", top="aliceblue")
        
        # 用于保存当前显示的 mesh actor，方便更新
        self.current_actor = None
        self.text_actor = None
        
        # 加载第一个模型
        self.load_current_mesh()

    def load_current_mesh(self):
        """加载并渲染当前索引的模型"""
        file_path = self.file_list[self.current_index]
        
        # 移除旧的模型和文字
        if self.current_actor:
            self.plotter.remove_actor(self.current_actor)
        if self.text_actor:
            self.plotter.remove_actor(self.text_actor)
            
        print(f"[{self.current_index + 1}/{self.total_count}] 正在加载: {file_path}")
        
        try:
            # 1. 读取网格
            # pyvista.read 非常稳健，能处理大部分不规范的 OBJ
            mesh = pv.read(file_path)
            
            # 2. 验证模型是否有实质内容 (防止空文件)
            if mesh.n_points == 0:
                raise ValueError("模型顶点数为 0")

            # 3. 添加到 Plotter
            # 重点验证：开启 show_edges 检查拓扑，开启 smooth_shading 检查法线
            self.current_actor = self.plotter.add_mesh(
                mesh,
                color='tan',
                show_edges=True,        # 关键：显示网格线，破洞和混乱拓扑一目了然
                edge_color='gray',
                smooth_shading=True,    # 关键：检查表面法线是否连续
                specular=0.5,           # 添加一点镜面反射，更容易看出表面凹凸
                name="current_mesh_actor"
            )
            
            # 4. 更新标题文字
            relative_path = os.path.relpath(file_path, self.root_dir)
            info_text = f"ID: {self.current_index + 1}/{self.total_count}\nPath: {relative_path}\nVerts: {mesh.n_points}\nFaces: {mesh.n_faces}"
            self.text_actor = self.plotter.add_text(info_text, position='upper_left', font_size=10, color='black')
            
            # 5. 重置相机，确保模型在视野中央
            self.plotter.reset_camera()
            
        except Exception as e:
            print(f"渲染模型失败: {file_path}\n错误信息: {e}")
            self.text_actor = self.plotter.add_text(f"LOAD ERROR:\n{os.path.basename(file_path)}", position='upper_left', font_size=12, color='red')

    def next_mesh(self):
        """切换到下一个模型"""
        if self.current_index < self.total_count - 1:
            self.current_index += 1
            self.load_current_mesh()
        else:
            print("已是最后一个模型。")

    def prev_mesh(self):
        """切换到上一个模型"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_mesh()
        else:
            print("已是第一个模型。")

    def show(self):
        """启动交互窗口"""
        print("\n" + "="*50)
        print("交互说明:")
        print("  -> (右方向键): 下一个模型")
        print("  <- (左方向键): 上一个模型")
        print("  鼠标左键: 旋转")
        print("  鼠标中键/滚轮: 平移/缩放")
        print("  Q: 退出程序")
        print("="*50 + "\n")
        self.plotter.show()

if __name__ == "__main__":
    # 你指定的数据集根目录
    dataset_root = '/mnt/d/Datasets/3D_FRONT/SAVEDIR'
    
    # 也可以通过命令行传入路径：python batch_preview_objs.py D:/其他数据集
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
        
    viewer = OBJBatchViewer(dataset_root, file_pattern="scene.obj")
    viewer.show()