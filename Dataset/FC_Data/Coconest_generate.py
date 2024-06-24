import os
import numpy as np
import scipy.io

def extract_indices(parcellation_path):
    try:
        mat_contents = scipy.io.loadmat(parcellation_path)
        diagonal_indices = np.diag(mat_contents['parc']).astype(int) - 1
        return diagonal_indices
    except Exception as e:
        print(f"Error loading or processing parcellation file {parcellation_path}: {e}")
        return None

def create_upper_triangular_matrix(fc_matrix, indices):
    if indices is not None:
        n = len(indices)
        new_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                new_matrix[i, j] = fc_matrix[indices[i], indices[j]]
        return new_matrix
    else:
        return None

def process_fc_matrices_for_parcellation(fc_data_dir, parcellation_path, output_dir):
    indices = extract_indices(parcellation_path)
    if indices is None:
        return

    parcellation_name = os.path.splitext(os.path.basename(parcellation_path))[0]
    parcellation_output_dir = os.path.join(output_dir, parcellation_name)
    os.makedirs(parcellation_output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(fc_data_dir):
        for file in files:
            if file.endswith('.mat'):
                fc_path = os.path.join(root, file)
                try:
                    fc_data = scipy.io.loadmat(fc_path)['fc']
                    new_matrix = create_upper_triangular_matrix(fc_data, indices)
                    if new_matrix is not None:
                        subID = os.path.basename(root)
                        sub_output_dir = os.path.join(parcellation_output_dir, subID)
                        os.makedirs(sub_output_dir, exist_ok=True)
                        output_path = os.path.join(sub_output_dir, "new_fc_matrix.mat")
                        scipy.io.savemat(output_path, {'new_fc_matrix': new_matrix})
                        print(f"New matrix saved: {output_path}")
                except IOError as e:
                    print(f"Failed to read file {fc_path}: {e}")
                except KeyError as e:
                    print(f"Key error in file {fc_path}: {e}")

def process_all_parcellations(parcellation_dir, fc_data_dir, output_dir):
    for parcellation_file in os.listdir(parcellation_dir):
        if parcellation_file.startswith("CoCoNest") and parcellation_file.endswith(".mat"):
            parcellation_path = os.path.join(parcellation_dir, parcellation_file)
            process_fc_matrices_for_parcellation(fc_data_dir, parcellation_path, output_dir)

# Define paths
parcellation_dir = '/work/users/y/i/yifzhang/zhengwu/EPIC_Data/fitbit_Data/output/TreeResults/parcellations'
fc_data_dir = '/work/users/y/i/yifzhang/zhengwu/EPIC_Data/fitbit_Data'
output_dir = '/work/users/y/i/yifzhang/zhengwu/EPIC_Data/fitbit_Data/CoCo_data'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all CoCoNest files
process_all_parcellations(parcellation_dir, fc_data_dir, output_dir)
