import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder
from tqdm.auto import tqdm

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample_sub = pd.read_csv('./data/sample_submission.csv')

def engineer_features(df):
    df_new = df.copy()
    
    df_new['req_rate_spike'] = df_new['req_rate_6h'] / (df_new['req_rate_24h'] + 0.001)
    df_new['device_email_ratio'] = df_new['device_email_cnt_8w'] / (df_new['session_length_min'] + 0.001)
    df_new['total_addr_months'] = df_new['curr_addr_months'] + df_new['prev_addr_months']
    df_new['valid_phone_count'] = df_new['is_home_phone_valid'] + df_new['is_mobile_valid']
    df_new['is_high_risk_email'] = ((df_new['name_email_sim'] < 0.3) & (df_new['is_free_email'] == 1)).astype(int)
    df_new['credit_to_income'] = df_new['req_credit_limit'] / (df_new['yearly_income'] + 0.001)
    df_new['foreign_unstable_risk'] = ((df_new['is_foreign_req'] == 1) & (df_new['is_session_persistent'] == 0)).astype(int)
    df_new['limit_diff_from_age_mean'] = df_new['req_credit_limit'] - df_new.groupby('age_bucket')['req_credit_limit'].transform('mean')
    df_new['income_diff_from_housing_mean'] = df_new['yearly_income'] - df_new.groupby('housing_status')['yearly_income'].transform('mean')
    
    return df_new

print("파생 변수를 생성 중입니다...")
train = engineer_features(train)
test = engineer_features(test)

drop_cols = ['id', 'device_fraud_history', 'is_bot_speed']
X = train.drop(columns=['fraud_bool'] + [c for c in drop_cols if c in train.columns])
y = train['fraud_bool']
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])

cat_features = ['housing_status', 'device_os', 'payment_type', 'employment_status', 'application_source', 'month_idx']

for col in cat_features:
    X[col] = X[col].astype(str).fillna('Missing')
    X_test[col] = X_test[col].astype(str).fillna('Missing')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

cat_features = ['housing_status', 'device_os', 'payment_type', 'employment_status', 'application_source', 'month_idx']

print(f"🚀 {n_splits}-Fold 교차 검증을 시작합니다.")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold + 1} / {n_splits} =====")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

    te = TargetEncoder(cols=cat_features, smoothing=10)
    X_train_te = te.fit_transform(X_train, y_train)
    X_valid_te = te.transform(X_valid)
    curr_test_te = te.transform(X_test)

    lgb_params = {
        'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt',
        'learning_rate': 0.015, 'num_leaves': 95, 'max_depth': 10, 'min_data_in_leaf': 600,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
        'scale_pos_weight': 5, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    
    train_set = lgb.Dataset(X_train_te, label=y_train)
    valid_set = lgb.Dataset(X_valid_te, label=y_valid, reference=train_set)
    
    lgb_model = lgb.train(
        lgb_params, train_set, num_boost_round=5000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(period=200)]
    )
    
    oof_lgb[val_idx] = lgb_model.predict(X_valid_te, num_iteration=lgb_model.best_iteration)
    test_preds_lgb += lgb_model.predict(curr_test_te, num_iteration=lgb_model.best_iteration) / n_splits

    cat_model = CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=6,
        eval_metric='PRAUC', scale_pos_weight=5,
        cat_features=cat_features, random_seed=42, verbose=200, early_stopping_rounds=100
    )
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    
    oof_cat[val_idx] = cat_model.predict_proba(X_valid)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test)[:, 1] / n_splits


best_ap = 0
best_w = 0.5

for w in np.arange(0, 1.05, 0.05):
    # $$P_{OOF\_final} = w \cdot OOF_{LGBM} + (1-w) \cdot OOF_{CatBoost}$$
    combined_oof = (oof_lgb * w) + (oof_cat * (1 - w))
    current_ap = average_precision_score(y, combined_oof)
    
    if current_ap > best_ap:
        best_ap = current_ap
        best_w = w

print(f"🏆 최적 가중치: LightGBM({best_w:.2f}) : CatBoost({1-best_w:.2f})")
print(f"⭐ 전체 데이터(OOF) 최고 AP 점수: {best_ap:.4f}")

sample_sub['fraud'] = (test_preds_lgb * best_w) + (test_preds_cat * (1 - best_w))
sample_sub.to_csv('./data/submission.csv', index=False)
