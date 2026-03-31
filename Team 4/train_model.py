import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import time
import os
import warnings

warnings.filterwarnings('ignore')

def feature_engineering_global(df):
    """Features that do not aggregate across rows, safe to apply globally."""
    df['req_rate_ratio_6h_24h'] = df['req_rate_6h'] / (df['req_rate_24h'] + 1e-5)
    df['req_rate_ratio_24h_4w'] = df['req_rate_24h'] / (df['req_rate_4w'] + 1e-5)
    df['addr_diff'] = df['curr_addr_months'] - df['prev_addr_months']
    df['device_email_ratio'] = df['device_email_cnt_8w'] / (df['dob_email_count_4w'] + 1e-5)
    df['income_per_age'] = df['yearly_income'] / (df['age_bucket'] + 1e-5)
    
    # Missing value flags
    df['prev_addr_missing'] = df['prev_addr_months'].isna().astype(int)
    df['bank_months_missing'] = df['bank_months_count'].isna().astype(int)
    
    df['req_rate_acceleration'] = df['req_rate_6h'] - (df['req_rate_24h'] / 4)
    df['total_reqs'] = df['zip_req_count_4w'] + df['branch_req_count_8w'] + df['dob_email_count_4w']

    return df

def make_cv_features(X_train, y_train, X_valid, X_test):
    """Features that aggregate across rows. MUST be fitted only on X_train to prevent leakage."""
    X_tr = X_train.copy()
    X_va = X_valid.copy()
    X_te = X_test.copy()
    
    X_tr['target_TEMP'] = y_train.values
    
    # 1. Target Encoding with Smoothing
    global_mean = X_tr['target_TEMP'].mean()
    te_cols = ['payment_type', 'employment_status', 'device_os', 'housing_status', 'application_source']
    
    for col in te_cols:
        agg = X_tr.groupby(col, observed=False)['target_TEMP'].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        weight = 100 # Smoothing weight (prevents overfitting to rare categories)
        smoothed = (counts * means + weight * global_mean) / (counts + weight)
        
        # safely map categorical variables to float
        X_tr[f'{col}_te'] = X_tr[col].astype(str).map(smoothed).fillna(global_mean).astype(float)
        X_va[f'{col}_te'] = X_va[col].astype(str).map(smoothed).fillna(global_mean).astype(float)
        X_te[f'{col}_te'] = X_te[col].astype(str).map(smoothed).fillna(global_mean).astype(float)
        
    # 2. GroupBy Statistics (e.g. Income by age and employment)
    group_col = ['age_bucket', 'employment_status']
    income_mean = X_tr.groupby(group_col, observed=False)['yearly_income'].mean().reset_index().rename(columns={'yearly_income': 'group_income_mean'})
    
    X_tr = X_tr.merge(income_mean, on=group_col, how='left')
    X_va = X_va.merge(income_mean, on=group_col, how='left')
    X_te = X_te.merge(income_mean, on=group_col, how='left')
    
    global_inc_mean = X_tr['group_income_mean'].mean()
    X_tr['group_income_mean'] = X_tr['group_income_mean'].fillna(global_inc_mean)
    X_va['group_income_mean'] = X_va['group_income_mean'].fillna(global_inc_mean)
    X_te['group_income_mean'] = X_te['group_income_mean'].fillna(global_inc_mean)
    
    for df in [X_tr, X_va, X_te]:
        df['income_vs_group'] = df['yearly_income'] / (df['group_income_mean'] + 1e-5)
    
    # 3. Mean credit score by age
    credit_mean = X_tr.groupby('age_bucket')['credit_risk_score'].mean().to_dict()
    global_credit_mean = X_tr['credit_risk_score'].mean()
    
    for df in [X_tr, X_va, X_te]:
        df['group_credit_mean'] = df['age_bucket'].map(credit_mean).fillna(global_credit_mean)
        df['credit_vs_group'] = df['credit_risk_score'] - df['group_credit_mean']
        
    X_tr.drop(columns=['target_TEMP'], inplace=True)
    return X_tr, X_va, X_te

def main():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sample_sub = pd.read_csv('sample_submission.csv')

    TARGET = 'fraud_bool'
    features = [c for c in train.columns if c not in ['id', TARGET]]

    print("Preprocessing data (replacing -1 with NaN)...")
    for c in features:
        if pd.api.types.is_numeric_dtype(train[c]):
            train[c] = train[c].replace(-1, np.nan)
            test[c] = test[c].replace(-1, np.nan)

    print("Engineering global features...")
    train = feature_engineering_global(train)
    test = feature_engineering_global(test)
    
    features = [c for c in train.columns if c not in ['id', TARGET]]
    categorical_cols = train[features].select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    
    for c in categorical_cols:
        train[c] = train[c].astype(str).fillna('missing').astype('category')
        test[c] = test[c].astype(str).fillna('missing').astype('category')

    X = train[features]
    y = train[TARGET]
    X_test = test[features]

    # LightGBM Params 
    lgb_params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': 8,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 50,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # CatBoost Params
    cat_params = {
        'iterations': 4000,
        'learning_rate': 0.05,
        'depth': 8,            # Increased depth for catching complex interactions
        'eval_metric': 'PRAUC',
        'random_seed': 42,
        'task_type': 'CPU',
        'verbose': 0,
        'early_stopping_rounds': 150
    }

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(len(train))
    test_preds_lgb = np.zeros(len(test))
    
    oof_preds_cat = np.zeros(len(train))
    test_preds_cat = np.zeros(len(test))

    print("Starting cross-validation training...")
    start_time = time.time()

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\\n------------- Fold {fold+1} / {n_splits} -------------")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # Apply CV-dependent engineered features (Leakage-Free!)
        X_train_fe, X_valid_fe, X_test_fe = make_cv_features(X_train, y_train, X_valid, X_test)
        
        # Ensure category types are preserved after merge for LightGBM
        for c in categorical_cols:
            X_train_fe[c] = X_train_fe[c].astype('category')
            X_valid_fe[c] = X_valid_fe[c].astype('category')
            X_test_fe[c] = X_test_fe[c].astype('category')

        # 1. ---------------- LightGBM ----------------
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(
            X_train_fe, y_train,
            eval_set=[(X_valid_fe, y_valid)],
            categorical_feature=categorical_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
        )

        valid_preds_lgb = model_lgb.predict_proba(X_valid_fe)[:, 1]
        oof_preds_lgb[valid_idx] = valid_preds_lgb
        test_preds_lgb += model_lgb.predict_proba(X_test_fe)[:, 1] / n_splits
        
        # 2. ---------------- CatBoost ----------------
        X_train_cat = X_train_fe.copy()
        X_valid_cat = X_valid_fe.copy()
        X_test_cat = X_test_fe.copy()
        
        # CatBoost needs them as raw strings natively
        for c in categorical_cols:
            X_train_cat[c] = X_train_cat[c].astype(str)
            X_valid_cat[c] = X_valid_cat[c].astype(str)
            X_test_cat[c] = X_test_cat[c].astype(str)
            
        train_pool = Pool(X_train_cat, y_train, cat_features=categorical_cols)
        valid_pool = Pool(X_valid_cat, y_valid, cat_features=categorical_cols)
        
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(train_pool, eval_set=valid_pool)
        
        valid_preds_cat = model_cat.predict_proba(valid_pool)[:, 1]
        oof_preds_cat[valid_idx] = valid_preds_cat
        test_preds_cat += model_cat.predict_proba(Pool(X_test_cat, cat_features=categorical_cols))[:, 1] / n_splits

        # Fold evaluation
        fold_ap_lgb = average_precision_score(y_valid, valid_preds_lgb)
        fold_ap_cat = average_precision_score(y_valid, valid_preds_cat)
        fold_ap_ens = average_precision_score(y_valid, (valid_preds_lgb + valid_preds_cat) / 2)
        
        print(f"Fold {fold+1} Validation AP | LGB: {fold_ap_lgb:.5f} | CAT: {fold_ap_cat:.5f} | ENS: {fold_ap_ens:.5f}")

    cv_ap_lgb = average_precision_score(y, oof_preds_lgb)
    cv_ap_cat = average_precision_score(y, oof_preds_cat)
    cv_ap_ens = average_precision_score(y, (oof_preds_lgb + oof_preds_cat) / 2)
    
    print(f"\\n================ SUMMARY ================")
    print(f"Overall CV AP - LightGBM: {cv_ap_lgb:.5f}")
    print(f"Overall CV AP - CatBoost: {cv_ap_cat:.5f}")
    print(f"Overall CV AP - Ensemble: {cv_ap_ens:.5f}")
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Save submission
    print("Saving submission...")
    sub = sample_sub.copy()
    sub['fraud'] = (test_preds_lgb + test_preds_cat) / 2
    sub.to_csv('submission.csv', index=False)
    
    sub_cat = sample_sub.copy()
    sub_cat['fraud'] = test_preds_cat
    sub_cat.to_csv('submission_cat.csv', index=False)
    print("Done! Submission saved to submission.csv (Ensemble) and submission_cat.csv (CatBoost Only)")

if __name__ == '__main__':
    main()
