from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 加载数据
data = pd.read_csv('/nas/longleaf/home/yifzhang/zhengwu/EPIC_Project/vectorized_cbcl/updated_flattened_fc_matrices_level_125.csv')
X = data.drop(['subject_id', 'CBCL'], axis=1)
X = X.select_dtypes(include=[np.number])
y = data['CBCL']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算类权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = dict(enumerate(class_weights))

# 定义pipeline
pipeline = Pipeline([
   # ('scaler', StandardScaler()),  # 取消注释以启用特征标准化
  # ('pca', PCA(n_components=0.95)),  # 取消注释以启用PCA
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
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
