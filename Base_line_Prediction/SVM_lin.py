import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

data = pd.read_csv('/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/final_data/merged_dataset_cbcl.csv')
X = data.drop(['ID', 'CBCL'], axis=1)

X = X.select_dtypes(include=[np.number])

y = data['CBCL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=0.95)), 
    ('svm', SVC(kernel='linear',probability=True)) 
  #  ('svm', SVC(kernel='linear',probability=True)) 
])

param_grid = {
    'svm__C': [0.01,0.05,0.1, 1, 10, 100]
}
print("without SS")
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy',verbose=2)

grid_search.fit(X_train, y_train)

print("Best parameter: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1] 

print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

