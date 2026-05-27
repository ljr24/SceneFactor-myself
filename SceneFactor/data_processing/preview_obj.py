import pyvista as pv
import os
import sys

def preview_scene_obj(file_path):
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: 找不到文件 {file_path}")
        return

    print(f"正在加载模型: {file_path}")
    
    try:
        # 2. 加载模型
        # pyvista 会自动处理读取逻辑
        mesh = pv.read(file_path)

        # 3. 创建渲染窗口
        plotter = pv.Plotter(title=f"OBJ Preview: {os.path.basename(file_path)}")
        
        # 添加模型，设置颜色和显示网格线（方便检查拓扑错误）
        plotter.add_mesh(mesh, 
                         color='tan', 
                         show_edges=True,  # 显示网格边缘，容易看清是否有破洞
                         edge_color='gray',
                         smooth_shading=True)

        # 添加坐标轴
        plotter.add_axes()
        
        # 设置背景色
        plotter.set_background("royalblue", top="aliceblue")

        print("窗口已开启。使用鼠标左键旋转，中键平移，右键缩放。")
        plotter.show()

    except Exception as e:
        print(f"渲染出错: {e}")

if __name__ == "__main__":
    # 你指定的路径
    target_path = '/mnt/d/Datasets/3D_FRONT/SAVEDIR/26387a08-3ebc-4638-83cd-33150de9f17a/scene.obj'
    
    # 也可以通过命令行传入路径：python preview_obj.py 你的模型路径.obj
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        
    preview_scene_obj(target_path)