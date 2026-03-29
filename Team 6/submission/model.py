
import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

COMP_DIR = './data'

TRAIN_PATH = os.path.join(COMP_DIR, "train.csv")
TEST_PATH = os.path.join(COMP_DIR, "test.csv")
SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")

OUT_DIR = "./output" 
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "fraud_bool"
ID_COL = "id"
MONTH_COL = "month_idx"
SUB_COL = "fraud"

def reduce_mem_usage(df):
    for col in df.columns:
        dt = df[col].dtype
        
        if not pd.api.types.is_numeric_dtype(dt):
            continue
            
        cmin, cmax = df[col].min(), df[col].max()
        
        if pd.api.types.is_integer_dtype(dt):
            if cmin >= np.iinfo(np.int8).min and cmax <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif cmin >= np.iinfo(np.int16).min and cmax <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif cmin >= np.iinfo(np.int32).min and cmax <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        
        else:
            df[col] = df[col].astype(np.float32)
            
    return df

train_df = reduce_mem_usage(pd.read_csv(TRAIN_PATH))
test_df = reduce_mem_usage(pd.read_csv(TEST_PATH))
sub_df = pd.read_csv(SUB_PATH)

for col in [MONTH_COL, "age_bucket"]:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

feature_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET]]

for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(train_df[col]):
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

print(f"사용 피처 개수: {len(feature_cols)}개")
print(f"범주형 피처로 변환된 컬럼들: {[c for c in feature_cols if train_df[c].dtype.name == 'category']}")

months = sorted(train_df[MONTH_COL].astype(int).unique())
valid_months = months[-2:] # 마지막 2개월
SEEDS = [42, 52]

oof = np.zeros(len(train_df), dtype=np.float32)
test_pred = np.zeros(len(test_df), dtype=np.float32)

for seed in SEEDS:
    seed_oof = np.zeros(len(train_df), dtype=np.float32)
    seed_test = np.zeros(len(test_df), dtype=np.float32)

    for m in valid_months:
        tr_mask = train_df[MONTH_COL].astype(int) < m
        va_mask = train_df[MONTH_COL].astype(int) == m
        
        X_tr = train_df.loc[tr_mask, feature_cols]
        y_tr = train_df.loc[tr_mask, TARGET]
        X_va = train_df.loc[va_mask, feature_cols]
        y_va = train_df.loc[va_mask, TARGET]
        
        clf = xgb.XGBClassifier(
            tree_method='hist',
            device='cuda',
            enable_categorical=True,
            eval_metric='aucpr',
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=6,     
            subsample=0.8,             
            colsample_bytree=0.8,    
            min_child_weight=10,    
            random_state=seed,
            early_stopping_rounds=150
        )
        
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=250
        )
        
        va_pred = clf.predict_proba(X_va)[:, 1]
        te_pred = clf.predict_proba(test_df[feature_cols])[:, 1]
        
        seed_oof[va_mask] = va_pred
        seed_test += te_pred / len(valid_months)
        
        ap = average_precision_score(y_va, va_pred)
        print(f"[XGBoost][Seed {seed}][Month {m}] AP: {ap:.6f}")
        
        del clf, X_tr, y_tr, X_va, y_va
        gc.collect()

    oof += seed_oof / len(SEEDS)
    test_pred += seed_test / len(SEEDS)

valid_mask = train_df[MONTH_COL].astype(int).isin(valid_months)
final_ap = average_precision_score(train_df.loc[valid_mask, TARGET], oof[valid_mask])
sub_df[SUB_COL] = test_pred
sub_df.to_csv(f"{OUT_DIR}/submission_xgboost_raw.csv", index=False)

