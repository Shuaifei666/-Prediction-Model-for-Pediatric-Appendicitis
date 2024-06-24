import os
import shutil
import pandas as pd

# 定义源目录和目标目录
source_dir = "/overflow/zzhanglab/SBCI_Finished_ABCD_Data/psc_connectome"
target_dir = "/nas/longleaf/home/yifzhang/zhengwu/EPIC_Project/fitbit_dataset"

# 读取CSV文件以获取ID列表
id_file = "/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/label/xtrm_label/fitbit_label.csv"
df = pd.read_csv(id_file)
ids = df.iloc[:, 0].tolist()  # 假设ID位于第一列

# 对于每个ID，检查并复制.mat文件
for id in ids:
    found = False  # 标记是否找到了对应的文件
    # 遍历源目录下的所有子目录，例如sbci_finished_batch1等
    for batch_dir in os.listdir(source_dir):
        batch_dir_path = os.path.join(source_dir, batch_dir)
        # 构造特定会话目录的路径
        session_dir_path = os.path.join(batch_dir_path, id, "ses-2YearFollowUpYArm1")
        if os.path.isdir(session_dir_path):
            # 构造.mat文件的完整路径
            mat_file_name = "ABCD_desikan_partbrain_subcort_cm_count_processed.mat"
            mat_file_path = os.path.join(session_dir_path, "psc_connectome", mat_file_name)
            if os.path.exists(mat_file_path):
                # 找到文件后，复制到目标目录
                target_path = os.path.join(target_dir, batch_dir, id, "ses-2YearFollowUpYArm1")
                os.makedirs(target_path, exist_ok=True)
                shutil.copy(mat_file_path, target_path)
                print(f"Copied: {mat_file_path} to {target_path}")
                found = True
                break  # 找到后跳出循环
    if not found:
        print(f"File for ID {id} not found.")

print("Done.")
