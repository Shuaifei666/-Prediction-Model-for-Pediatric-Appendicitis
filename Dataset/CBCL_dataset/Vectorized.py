import numpy as np
from scipy.io import loadmat
import os
import pandas as pd

# 父目录路径，包含多个sbci_finished_和sbci_finished_newbatch_文件夹
parent_dir = '/nas/longleaf/home/yifzhang/zhengwu/EPIC_Project/CBCL_dataset'

# 遍历父目录下的所有sbci_finished_开头的子目录
for batch_dir_name in os.listdir(parent_dir):
    if batch_dir_name.startswith('sbci_finished_'):
        batch_dir_path = os.path.join(parent_dir, batch_dir_name)
        
        # 遍历每个子目录下的ID文件夹
        for id_folder_name in os.listdir(batch_dir_path):
            id_folder_path = os.path.join(batch_dir_path, id_folder_name)
            
            # 在ID文件夹内寻找 ses-2YearFollowUpYArm1 子目录
            session_folder_name = 'ses-2YearFollowUpYArm1'
            session_folder_path = os.path.join(id_folder_path, session_folder_name)
            
            # 检查会话目录是否存在
            if os.path.isdir(session_folder_path):
                # 遍历会话目录中的.mat文件
                for mat_file_name in os.listdir(session_folder_path):
                    if mat_file_name.endswith('.mat'):
                        mat_file_path = os.path.join(session_folder_path, mat_file_name)
                        
                        # 加载.mat文件
                        mat_contents = loadmat(mat_file_path)
                        variable_name = [name for name in mat_contents if not name.startswith('__')][0]
                        matrix = mat_contents[variable_name]
                        
                        # 确保矩阵为方阵
                        assert matrix.shape[0] == matrix.shape[1], f"The matrix in {mat_file_path} is not square."
                        
                        # 提取上三角矩阵并向量化
                        n = matrix.shape[0]
                        upper_triangular = matrix[np.triu_indices(n, k=1)]
                        vectorized = upper_triangular.flatten()
                        
                        # 创建包含ID的向量化数据
                        vectorized_data = [id_folder_name] + vectorized.tolist()
                        
                        # 将数据转换为DataFrame
                        df = pd.DataFrame([vectorized_data])
                        
                        # 保存为CSV文件
                        csv_file_name = f"{id_folder_name}_{mat_file_name.replace('.mat', '_vectorized.csv')}"
                        csv_file_path = os.path.join(session_folder_path, csv_file_name)
                        df.to_csv(csv_file_path, index=False, header=False)
                        print(f"Processed and saved: {csv_file_path}")
