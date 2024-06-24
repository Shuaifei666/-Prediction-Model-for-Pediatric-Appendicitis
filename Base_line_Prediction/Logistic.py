import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load your dataset
data = pd.read_csv('/nas/longleaf/home/yifzhang/zhengwu/EPIC_Project/CBCL_dataset/ABCD_destrieux_partbrain_subcort_cm_count_processed_vectorized_CBCL.csv')
original_row_count = data.shape[0]
print(f"Original number of rows: {original_row_count}")
data = data.dropna()
X = data.drop(['ID', 'CBCL'], axis=1)
y = data['CBCL']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

# Creating transformers for preprocessing
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = Pipeline(steps=[
    ('pca', PCA(n_components=0.9))
])


# Combining transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Creating the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('logreg', LogisticRegression(solver='liblinear',max_iter=100000))
])

# Setting up the parameter grid
param_grid = {
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'logreg__penalty': ['l1', 'l2']
}

# Running GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Displaying the best parameters
print("Best parameter: ", grid_search.best_params_)

# Predicting on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Printing the classification report
print(classification_report(y_test, y_pred))

