
"""HR Promotion Prediction - runnable script
Usage: python3 hr_promotion.py
This script trains a RandomForest pipeline and produces submission_hr_promotion.csv in /mnt/data
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib

train = pd.read_csv('/mnt/data/train.csv')
test = pd.read_csv('/mnt/data/test.csv')
sample = pd.read_csv('/mnt/data/sample_submission.csv')

X = train.drop(['is_promoted','employee_id'], axis=1)
y = train['is_promoted']
X_test = test.drop(['employee_id'], axis=1)

numeric_cols = []
categorical_cols = []
for c in X.columns:
    if X[c].dtype in ['int64','float64']:
        if X[c].nunique() <= 10 and c not in ['avg_training_score','age','length_of_service']:
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    else:
        categorical_cols.append(c)

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols),('cat', categorical_transformer, categorical_cols)])

clf = Pipeline(steps=[('preprocessor', preprocessor),('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])

clf.fit(X, y)
test_preds = clf.predict(X_test)
sample['is_promoted'] = test_preds
sample.to_csv('/mnt/data/submission_hr_promotion.csv', index=False)
joblib.dump(clf, '/mnt/data/hr_promotion_pipeline.joblib')
print('Done. Submission and pipeline saved in /mnt/data')
