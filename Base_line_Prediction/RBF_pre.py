import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# 定义数据目录
directory = '/nas/longleaf/home/yifzhang/zhengwu/EPIC_Project/vectorized_cbcl/'

# 获取所有CSV文件
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# 准备记录AUC分数的字典
auc_scores = {}

# 遍历所有文件
for file in files:
    # 加载数据
    data = pd.read_csv(file)
    X = data.drop(['subject_id', 'CBCL'], axis=1).select_dtypes(include=[np.number])
    y = data['CBCL']

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 计算类权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = dict(enumerate(class_weights))

    # 定义pipeline
    pipeline = Pipeline([
        ('svm', SVC(kernel='rbf', class_weight=weights_dict, probability=True))
    ])

    # 定义网格搜索参数
    param_grid = {
        'svm__C': [ 1, 10, 100,1000],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100]
    }

    # 执行网格搜索
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=3)
    grid_search.fit(X_train, y_train)

    # 输出最佳参数和分类报告
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类的概率
    auc_score = roc_auc_score(y_test, y_pred_proba)
    auc_scores[os.path.basename(file)] = auc_score

    print(f"File: {os.path.basename(file)}")
    print("Best parameters:", grid_search.best_params_)
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {auc_score}")

# 绘制AUC分数图
plt.figure(figsize=(10, 8))
plt.barh(list(auc_scores.keys()), list(auc_scores.values()), color='skyblue')
plt.xlabel('AUC Score')
plt.title('AUC Scores for Different CSV Files')
plt.show()
