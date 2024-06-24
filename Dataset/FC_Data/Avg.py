import os
import numpy as np
import scipy.io

# 定义根目录
root_dir = "/work/users/y/i/yifzhang/zhengwu/EPIC_Data/fitbit_Data/"

# 初始化变量
sum_matrix = None
count = 0

# 遍历目录并加载每个 fc_avg_ico4.mat 文件
for batch_dir in os.listdir(root_dir):
    batch_dir_path = os.path.join(root_dir, batch_dir)
    if os.path.isdir(batch_dir_path):  # 确保是目录
        for sub_dir in os.listdir(batch_dir_path):
            sub_dir_path = os.path.join(batch_dir_path, sub_dir)
            if os.path.isdir(sub_dir_path):  # 确保是目录
                file_path = os.path.join(sub_dir_path, "fc_avg_ico4.mat")
                if os.path.exists(file_path):
                    try:
                        mat_contents = scipy.io.loadmat(file_path)
                        print(f"Variables in {file_path}: {mat_contents.keys()}")
                        if 'fc' in mat_contents:
                            mat = mat_contents['fc']
                            if sum_matrix is None:
                                sum_matrix = np.zeros_like(mat)
                            sum_matrix += mat
                            count += 1
                        else:
                            print(f"'fc' not found in {file_path}")
                    except Exception as e:
                        print(f"Failed to load {file_path} due to an error: {e}")

# 计算平均值
if count > 0:
    average_matrix = sum_matrix / count
    print("Average matrix calculated:")
    print(average_matrix)
else:
    average_matrix = None
    print("No matrices found.")

# 保存平均矩阵到一个新的 MATLAB 文件
output_file = "/work/users/y/i/yifzhang/zhengwu/EPIC_Data/fitbit_Data/average_fc_avg_ico4.mat"
if average_matrix is not None:
    try:
        scipy.io.savemat(output_file, {'average_fc_avg_ico4': average_matrix})
        print(f"Average matrix saved to {output_file}")
    except Exception as e:
        print(f"Failed to save the average matrix due to an error: {e}")
