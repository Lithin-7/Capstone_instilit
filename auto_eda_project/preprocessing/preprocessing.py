
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize

#  Clean categorical typos (e.g., job_title)

def clean_categorical_typos(df):
    job_title_map = {
        'data scntist': 'data scientist',
        'data scienist': 'data scientist',
        'dt scientist': 'data scientist',
        'ml engr': 'ml engineer',
        'ml enginer': 'ml engineer',
        'machine learning engr': 'ml engineer',
        'software engr': 'software engineer',
        'softwre engineer': 'software engineer',
        'sofware engneer': 'software engineer'
    }

    if 'job_title' in df.columns:
        df['job_title'] = df['job_title'].str.lower().str.strip().replace(job_title_map)
    return df

# Drop fully missing columns (e.g., all NaN cols)

def drop_fully_missing_columns(df):
    return df.dropna(axis=1, how='all')


#  Log transform the target for skew handling

def log_transform_target(y):
    return np.log1p(y)

# Custom Winsorizer for outlier treatment

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits
        self.columns = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
        else:
            self.columns = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns)
        X_winsorized = X.copy()
        for col in self.columns:
            X_winsorized[col] = winsorize(X_winsorized[col], limits=self.limits)
        return X_winsorized.values  # Return numpy array for pipeline

#  Get Preprocessing Pipeline (based on current dataset)

def get_preprocessor(df):
    # Drop irrelevant or leakage columns (if any exist)
    drop_cols = ['education', 'skills', 'company_location', 'salary_currency']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Identify feature types
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Remove target from numerical columns
    target = 'adjusted_total_usd'
    if target in num_cols:
        num_cols.remove(target)

    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='median')),
        ('winsor', Winsorizer()),
        ('scaler', MinMaxScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combined ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor

