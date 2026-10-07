import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import matplotlib.pyplot as plt
import os

def evaluate_model(y_true, y_prob):
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob)
    }
    return metrics

def plot_roc_curve(y_true, y_prob, save_path=None):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_prob):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_pr_curve(y_true, y_prob, save_path=None):
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'PR-AUC = {average_precision_score(y_true, y_prob):.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_prob_distribution(y_true, y_prob, save_path=None):
    import seaborn as sns
    df = pd.DataFrame({'Target': y_true, 'Probability': y_prob})
    plt.figure()
    sns.kdeplot(data=df[df['Target']==0]['Probability'], label='TARGET=0', fill=True)
    sns.kdeplot(data=df[df['Target']==1]['Probability'], label='TARGET=1', fill=True)
    plt.xlabel('Predicted Probability')
    plt.title('Predicted Probability Distribution')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def extract_coefficients(pipeline):
    clf = pipeline.named_steps['clf']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names from ColumnTransformer
    ext_cols = ['APP_EXT_SOURCE_1', 'APP_EXT_SOURCE_2', 'APP_EXT_SOURCE_3']
    ext_missing = [f"{c}_missing_indicator" for c in ext_cols]
    
    # SimpleImputer add_indicator puts the indicators AFTER the original features
    # Wait, the shape of ext_transformer output will be (N, 3 + number_of_missing_cols).
    # Since all 3 have missing values, there are 3 indicators.
    ext_feat_names = ext_cols + ext_missing
    
    other_num_cols = [
        'APP_INCOME_TOTAL', 'APP_CREDIT_AMOUNT', 'APP_ANNUITY',
        'APP_DAYS_BIRTH', 'APP_DAYS_EMPLOYED', 
        'APP_CREDIT_INCOME_RATIO', 'APP_ANNUITY_INCOME_RATIO'
    ]
    
    feature_names = ext_feat_names + other_num_cols
    coefs = clf.coef_[0]
    
    df_coef = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'odds_ratio': np.exp(coefs)
    })
    df_coef['abs_coef'] = df_coef['coefficient'].abs()
    return df_coef.sort_values('abs_coef', ascending=False).drop('abs_coef', axis=1)
