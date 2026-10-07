import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

class DaysEmployedAnomalyHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_out = X.copy()
        # Ensure it's DataFrame for column access, though inside Pipeline it might be numpy array.
        # It's better to use this before ColumnTransformer, or pass df directly.
        if isinstance(X_out, pd.DataFrame) and 'APP_DAYS_EMPLOYED' in X_out.columns:
            X_out.loc[X_out['APP_DAYS_EMPLOYED'] == 365243, 'APP_DAYS_EMPLOYED'] = np.nan
        elif isinstance(X_out, np.ndarray):
            # Assuming it's applied correctly to that specific column
            X_out[X_out == 365243] = np.nan
        return X_out

def load_and_split_data(filepath, random_state=42):
    df = pd.read_csv(filepath)
    app_features = [
        'APP_INCOME_TOTAL', 'APP_CREDIT_AMOUNT', 'APP_ANNUITY',
        'APP_DAYS_BIRTH', 'APP_DAYS_EMPLOYED', 
        'APP_EXT_SOURCE_1', 'APP_EXT_SOURCE_2', 'APP_EXT_SOURCE_3',
        'APP_CREDIT_INCOME_RATIO', 'APP_ANNUITY_INCOME_RATIO'
    ]
    X = df[['SK_ID_CURR'] + app_features].copy()
    y = df['TARGET'].copy()
    
    # Stratified Split 70/15/15
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )
    
    # To get 15% out of 85%, we need 15/85 = 0.17647
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15/0.85), stratify=y_train_val, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_baseline_pipeline():
    # We want indicator for EXT_SOURCE missingness. SimpleImputer can do add_indicator.
    app_features = [
        'APP_INCOME_TOTAL', 'APP_CREDIT_AMOUNT', 'APP_ANNUITY',
        'APP_DAYS_BIRTH', 'APP_DAYS_EMPLOYED', 
        'APP_EXT_SOURCE_1', 'APP_EXT_SOURCE_2', 'APP_EXT_SOURCE_3',
        'APP_CREDIT_INCOME_RATIO', 'APP_ANNUITY_INCOME_RATIO'
    ]
    
    # For EXT sources, we add indicator
    ext_cols = ['APP_EXT_SOURCE_1', 'APP_EXT_SOURCE_2', 'APP_EXT_SOURCE_3']
    other_num_cols = [c for c in app_features if c not in ext_cols]
    
    ext_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
        ('scaler', StandardScaler())
    ])
    
    other_transformer = Pipeline([
        ('anomaly', DaysEmployedAnomalyHandler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('ext', ext_transformer, ext_cols),
        ('other', other_transformer, other_num_cols)
    ], remainder='drop')
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    return pipeline

def get_constant_baseline_predictions(y_train, n_samples):
    base_rate = y_train.mean()
    return np.full(n_samples, base_rate)
