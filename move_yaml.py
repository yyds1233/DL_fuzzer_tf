import os
import shutil

def move_yaml_files(src_dirs, out_dir):
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for src_dir in src_dirs:
        if not os.path.isdir(src_dir):
            print(f"源文件夹 {src_dir} 不存在。")
            continue
        
        # 遍历文件夹中的文件
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".yaml"):
                    # 计算源文件的完整路径
                    src_file = os.path.join(root, file)
                    # 计算目标文件的完整路径
                    dst_file = os.path.join(out_dir, file)
                    
                    # 如果文件已经存在，则重命名目标文件以避免覆盖
                    if os.path.exists(dst_file):
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dst_file):
                            dst_file = os.path.join(out_dir, f"{base}_{counter}{ext}")
                            counter += 1
                    
                    # 移动文件
                    shutil.move(src_file, dst_file)
                    print(f"移动 {src_file} 到 {dst_file}")

# 示例用法
src_dirs = ["/root/yaml_list_1/08_final", "/root/yaml_list_2/08_final", "/root/yaml_list_3/08_final", "/root/yaml_list_4/08_final"]  # 传入多个源文件夹
out_dir = "/root/yaml/"  # 输出文件夹
move_yaml_files(src_dirs, out_dir)