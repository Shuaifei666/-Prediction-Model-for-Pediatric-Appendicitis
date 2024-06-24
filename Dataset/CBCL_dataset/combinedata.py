import pandas as pd
import os

# 父目录路径，包含多个sbci_finished_和sbci_finished_newbatch_文件夹
parent_dir = '/nas/longleaf/home/yifzhang/zhengwu/EPIC_Project/CBCL_dataset'

# 初始化一个空列表，用于存储所有DataFrame
dataframes = []

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
                # 遍历会话目录中的_vectorized.csv文件
                for csv_file_name in os.listdir(session_folder_path):
                    if csv_file_name.endswith('_vectorized.csv'):
                        csv_file_path = os.path.join(session_folder_path, csv_file_name)
                        
                        # 读取CSV文件并将DataFrame添加到列表
                        df = pd.read_csv(csv_file_path, header=None)
                        dataframes.append(df)

# 使用concat合并所有的DataFrame
all_data = pd.concat(dataframes, ignore_index=True)
new_column_names = ['ID'] + [f'V{i}' for i in range(1, df.shape[1])]
all_data.columns = new_column_names
# 保存合并后的大DataFrame到CSV文件
output_csv_path = os.path.join(parent_dir, 'all_vectorized_data_CBCL.csv')
all_data.to_csv(output_csv_path, index=False, header=True)
print(f"All data has been merged and saved to: {output_csv_path}")

