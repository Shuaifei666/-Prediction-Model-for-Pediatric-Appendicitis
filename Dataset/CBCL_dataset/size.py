import pandas as pd
data = pd.read_csv('/nas/longleaf/home/yifzhang/zhengwu/EPIC_Project/CBCL_dataset/ABCD_destrieux_partbrain_subcort_cm_count_processed_vectorized_CBCL.csv')
data = data.dropna()
original_row_count = data.shape[0]
print(f"Original number of rows: {original_row_count}")