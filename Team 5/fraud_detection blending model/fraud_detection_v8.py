"""
================================================================================
v8 — Seed Averaging + TabNet
================================================================================
전략 1: XGB/CAT Seed Averaging — 분산 감소 (guaranteed improvement)
전략 2: TabNet — 트리와 근본적으로 다른 NN → 앙상블 다양성 (breakthrough potential)

MLP 실험에서 확인된 팩트:
  - NN-트리 상관: 0.867 (다양성 충분)
  - MLP 개별 AP: 0.1446 (너무 약해서 앙상블 기여 못함)
  → TabNet으로 개별 AP를 0.165+로 올리면 앙상블 breakthrough 가능

실행 구조:
  Phase 1: TabNet 설치 확인
  Phase 2: XGB 5-Seed × 5-Fold + CAT 3-Seed × 5-Fold
  Phase 3: TabNet 5-Fold
  Phase 4: 앙상블 (Seed Avg 효과 + TabNet 기여 각각 추적)

실행 결과:
======================================================================
v8 — SEED AVERAGING + TABNET
======================================================================

[0] TabNet 설치 확인
  pytorch-tabnet 이미 설치됨

[1] 데이터 로드 + 전처리 + 피처 엔지니어링
  train: (700000, 33), test: (300000, 32), fraud: 0.010917
  피처 엔지니어링 완료

[2] 피처 선택
  피처 수: 94

======================================================================
PHASE 2: XGB Seed Averaging (5 seeds)
======================================================================

--- XGB Seed 42 (1/5) ---
  Seed 42 OOF AP: 0.177471 (86s)

--- XGB Seed 43 (2/5) ---
  Seed 43 OOF AP: 0.177139 (82s)

--- XGB Seed 44 (3/5) ---
  Seed 44 OOF AP: 0.176570 (80s)

--- XGB Seed 45 (4/5) ---
  Seed 45 OOF AP: 0.177392 (87s)

--- XGB Seed 46 (5/5) ---
  Seed 46 OOF AP: 0.177070 (88s)

[XGB Seed Averaging 결과]
  단일 seed (42):  0.177471
  5-seed 평균:     0.178302
  Seed Avg 효과:   +0.000830

======================================================================
PHASE 2B: CAT Seed Averaging (3 seeds)
======================================================================

[CAT Seed Averaging 결과]
  단일 seed (42):  0.174014
  3-seed 평균:     0.174974
  Seed Avg 효과:   +0.000959

======================================================================
PHASE 3: TabNet
======================================================================

--- Fold 1/5 ---

Early stopping occurred at epoch 52 with best_epoch = 37 and best_val_0_auc = 0.88307
  AP=0.158220 | 645s

--- Fold 2/5 ---

Early stopping occurred at epoch 44 with best_epoch = 29 and best_val_0_auc = 0.89308
  AP=0.160974 | 576s

--- Fold 3/5 ---

Early stopping occurred at epoch 53 with best_epoch = 38 and best_val_0_auc = 0.88897
  AP=0.158021 | 654s

--- Fold 4/5 ---

Early stopping occurred at epoch 36 with best_epoch = 21 and best_val_0_auc = 0.88087
  AP=0.152172 | 450s

--- Fold 5/5 ---

Early stopping occurred at epoch 52 with best_epoch = 37 and best_val_0_auc = 0.88751
  AP=0.155847 | 619s

[TabNet] OOF AP: 0.150402
  TabNet-XGB 상관: 0.8789
  TabNet-CAT 상관: 0.8907
  XGB-CAT 상관:    0.9686

======================================================================
PHASE 4: 앙상블
======================================================================

[4-1] Seed Averaging 효과 (XGB+CAT만)
  단일seed XGB+CAT: 0.178939 (XGB=0.60)
  SeedAvg XGB+CAT:  0.179280 (XGB=0.65)
  Seed Avg 앙상블 효과: +0.000341

[4-2] TabNet 기여도
  XGB+CAT+TabNet: 0.179350 (XGB=0.65, CAT=0.30, Tab=0.05)
  TabNet 기여:    +0.000070

[4-3] Rank 가중 평균
  Rank XGB+CAT+Tab: 0.178631 (XGB=0.70, CAT=0.25, Tab=0.05)

[4-4] Stacking
  Stacking AP: 0.176739

======================================================================
PHASE 5: 최종 결과
======================================================================

  방식                                AP
  ----------------------------------------
  XGB+CAT+Tab_WA            0.179350 ← BEST
  XGB+CAT_seedavg           0.179280
  XGB+CAT_single            0.178939
  XGB+CAT+Tab_Rank          0.178631
  XGB_5seed                 0.178302
  XGB_single                0.177471
  XGB+CAT+Tab_Stack         0.176739
  CAT_3seed                 0.174974
  CAT_single                0.174014
  TabNet                    0.150402

  [역대 AP]
  v6:  0.177950 (LB 0.17972)
  v7:  0.178483
  v8:  0.179350 (XGB+CAT+Tab_WA)
  v8 vs v7: +0.000867

  [효과 분해]
  Seed Averaging:  +0.000341
  TabNet 기여:     +0.000070
  TabNet-XGB 상관: 0.8789
  TabNet-CAT 상관: 0.8907

[제출 파일] submission_v8.csv 저장 완료 (방식: XGB+CAT+Tab_WA)
  예측값 범위: [0.000049, 0.855741]

총 실행 시간: 92.7분

======================================================================
v8 완료!
======================================================================

================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
import os
import time
import subprocess
import sys

warnings.filterwarnings('ignore')

DATA_DIR = './'
OUTPUT_DIR = './'
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)
start_time = time.time()

print("=" * 70)
print("v8 — SEED AVERAGING + TABNET")
print("=" * 70)


# ============================================================================
# Phase 0: TabNet 설치 확인
# ============================================================================

print("\n[0] TabNet 설치 확인")
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("  pytorch-tabnet 이미 설치됨")
    TABNET_AVAILABLE = True
except ImportError:
    print("  pytorch-tabnet 설치 중...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               'pytorch-tabnet', '--quiet'])
        from pytorch_tabnet.tab_model import TabNetClassifier
        print("  pytorch-tabnet 설치 완료")
        TABNET_AVAILABLE = True
    except Exception as e:
        print(f"  ⚠ pytorch-tabnet 설치 실패: {e}")
        print("  → TabNet 없이 Seed Averaging만 진행")
        TABNET_AVAILABLE = False


# ============================================================================
# 1. 데이터 로드 + 전처리 + v7 확정 피처
# ============================================================================

print("\n[1] 데이터 로드 + 전처리 + 피처 엔지니어링")
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

TARGET = 'fraud_bool'
ID_COL = 'id'
print(f"  train: {train.shape}, test: {test.shape}, fraud: {train[TARGET].mean():.6f}")

train['_is_train'] = 1
test['_is_train'] = 0
test[TARGET] = -1
df = pd.concat([train, test], axis=0, ignore_index=True)

df.drop(columns=['device_fraud_history'], inplace=True)

sentinel_cols = ['prev_addr_months', 'curr_addr_months', 'bank_months_count',
                 'device_email_cnt_8w', 'session_length_min']
for col in sentinel_cols:
    df[f'{col}_missing'] = (df[col] == -1).astype(np.int8)
    df[f'{col}_clean'] = df[col].replace(-1, np.nan)

cat_cols = ['payment_type', 'employment_status', 'housing_status',
            'application_source', 'device_os']
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_le'] = le.fit_transform(df[col].astype(str))

# --- v6 확정 피처 (전부 유지, 생략 없이) ---
df['risk_flag_count'] = (df['is_free_email'] + df['is_foreign_req'] +
    (df['init_transfer_amt'] < 0).astype(int) + (df['application_source'] == 'TELEAPP').astype(int))
df['safe_flag_count'] = (df['is_home_phone_valid'] + df['is_mobile_valid'] +
    df['has_other_cards'] + df['is_session_persistent'])
df['net_risk'] = df['risk_flag_count'] - df['safe_flag_count']
df['trust_score'] = (df['is_home_phone_valid'] + df['is_mobile_valid'] + df['has_other_cards'] +
    (1 - df['is_free_email']) + (1 - df['is_foreign_req']) + df['is_session_persistent'])
df['missing_count'] = sum(df[f'{c}_missing'] for c in sentinel_cols)
df['addr_stability'] = df['curr_addr_months_clean'] / (df['prev_addr_months_clean'] + 1)
df['total_addr_months'] = df['curr_addr_months_clean'].fillna(0) + df['prev_addr_months_clean'].fillna(0)
df['long_curr_addr'] = (df['curr_addr_months'] > 100).astype(np.int8)
df['income_credit_ratio'] = df['req_credit_limit'] / (df['yearly_income'] + 0.01)
df['risk_x_limit'] = df['credit_risk_score'] * df['req_credit_limit']
df['age_income'] = df['age_bucket'] * df['yearly_income']
df['init_transfer_negative'] = (df['init_transfer_amt'] < 0).astype(np.int8)
df['rate_6h_24h_ratio'] = df['req_rate_6h'] / (df['req_rate_24h'] + 1)
for col in ['req_rate_6h', 'req_rate_24h']:
    df[f'{col}_month_rank'] = df.groupby('month_idx')[col].rank(pct=True)
df['branch_intensity'] = df['branch_req_count_8w'] / (df['zip_req_count_4w'] + 1)
df['dob_email_intensity'] = df['dob_email_count_4w'] / (df['device_email_cnt_8w'].replace(-1, 1) + 1)
df['high_risk_combo'] = ((df['credit_risk_score'] > df['credit_risk_score'].quantile(0.75)).astype(int) +
    (df['req_credit_limit'] >= 1000).astype(int) + df['is_free_email'])
df['transient_foreign'] = (1 - df['is_session_persistent']) * df['is_foreign_req']
df['mature_new_account'] = ((df['age_bucket'] >= 40).astype(int) *
    (df['yearly_income'] >= 0.6).astype(int) * df['bank_months_count_missing'])
df['low_sim_free'] = ((df['name_email_sim'] < df['name_email_sim'].quantile(0.25)).astype(int) * df['is_free_email'])
df['is_teleapp'] = (df['application_source'] == 'TELEAPP').astype(np.int8)
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)
df['req_credit_limit_freq'] = df['req_credit_limit'].map(df['req_credit_limit'].value_counts(normalize=True))
df['velocity_sum'] = df['req_rate_6h'] + df['req_rate_24h'] + df['zip_req_count_4w']
df['velocity_std'] = df[['req_rate_6h', 'req_rate_24h']].std(axis=1)

train_only = df[df['_is_train'] == 1]
df['housing_credit_avg'] = df['housing_status'].map(train_only.groupby('housing_status')['credit_risk_score'].mean())
df['housing_limit_avg'] = df['housing_status'].map(train_only.groupby('housing_status')['req_credit_limit'].mean())
df['credit_vs_housing_avg'] = df['credit_risk_score'] - df['housing_credit_avg']
df['limit_vs_housing_avg'] = df['req_credit_limit'] - df['housing_limit_avg']
df['credit_x_age'] = df['credit_risk_score'] * df['age_bucket']
df['credit_x_income'] = df['credit_risk_score'] * df['yearly_income']
df['age_x_addr'] = df['age_bucket'] * df['curr_addr_months']
df['sim_x_rate6h'] = df['name_email_sim'] * df['req_rate_6h']

df['employ_credit_avg'] = df['employment_status'].map(train_only.groupby('employment_status')['credit_risk_score'].mean())
df['employ_limit_avg'] = df['employment_status'].map(train_only.groupby('employment_status')['req_credit_limit'].mean())
df['credit_vs_employ_avg'] = df['credit_risk_score'] - df['employ_credit_avg']
df['limit_vs_employ_avg'] = df['req_credit_limit'] - df['employ_limit_avg']
df['payment_credit_avg'] = df['payment_type'].map(train_only.groupby('payment_type')['credit_risk_score'].mean())
df['payment_limit_avg'] = df['payment_type'].map(train_only.groupby('payment_type')['req_credit_limit'].mean())
df['credit_vs_payment_avg'] = df['credit_risk_score'] - df['payment_credit_avg']
df['os_credit_avg'] = df['device_os'].map(train_only.groupby('device_os')['credit_risk_score'].mean())
df['credit_vs_os_avg'] = df['credit_risk_score'] - df['os_credit_avg']

tc = train_only.copy()
tc['_he'] = tc['housing_status'].astype(str) + '_' + tc['employment_status'].astype(str)
he_stats = tc.groupby('_he')['credit_risk_score'].mean()
df['_he'] = df['housing_status'].astype(str) + '_' + df['employment_status'].astype(str)
df['he_credit_avg'] = df['_he'].map(he_stats)
df['credit_vs_he_avg'] = df['credit_risk_score'] - df['he_credit_avg'].fillna(df['credit_risk_score'].mean())
df.drop(columns=['_he'], inplace=True)

df['limit_to_income_extreme'] = df['req_credit_limit'] / (df['yearly_income'] * 1000 + 1)
df['addr_change_volatility'] = np.abs(df['curr_addr_months_clean'].fillna(0) - df['prev_addr_months_clean'].fillna(0))
df['risk_per_bank_month'] = df['credit_risk_score'] / (df['bank_months_count'].replace(-1, 0) + 1)

# v7 피처 (IF + Time)
from sklearn.ensemble import IsolationForest
if_input_cols = ['yearly_income', 'name_email_sim', 'prev_addr_months', 'curr_addr_months',
    'age_bucket', 'days_since_req', 'init_transfer_amt', 'zip_req_count_4w',
    'req_rate_6h', 'req_rate_24h', 'req_rate_4w', 'branch_req_count_8w',
    'dob_email_count_4w', 'credit_risk_score', 'bank_months_count',
    'req_credit_limit', 'session_length_min', 'device_email_cnt_8w']
if_data = df[if_input_cols].fillna(-999).values
iso = IsolationForest(n_estimators=200, max_samples=0.5, contamination=0.011, random_state=SEED, n_jobs=-1)
iso.fit(if_data)
df['if_anomaly_score'] = -iso.decision_function(if_data)

for col in ['credit_risk_score', 'req_credit_limit', 'req_rate_6h', 'init_transfer_amt']:
    month_mean = train_only.groupby('month_idx')[col].mean()
    df[f'{col}_vs_month'] = df[col] - df['month_idx'].map(month_mean)
df['is_recent_month'] = (df['month_idx'] >= 6).astype(np.int8)
df['month_progress'] = df['month_idx'] / 7.0

print("  피처 엔지니어링 완료")


# ============================================================================
# 2. 피처 선택 + 데이터 준비
# ============================================================================

print("\n[2] 피처 선택")

exclude_cols = [TARGET, '_is_train', ID_COL,
    'payment_type', 'employment_status', 'housing_status', 'application_source', 'device_os',
    'prev_addr_months_clean', 'curr_addr_months_clean', 'bank_months_count_clean',
    'device_email_cnt_8w_clean', 'session_length_min_clean']
exclude_cols = [c for c in exclude_cols if c in df.columns]
feature_cols = [c for c in df.columns if c not in exclude_cols]

for col in feature_cols:
    if df[col].dtype in ['float64', 'float32']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

print(f"  피처 수: {len(feature_cols)}")

train_df = df[df['_is_train'] == 1].reset_index(drop=True)
test_df = df[df['_is_train'] == 0].reset_index(drop=True)
y = train_df[TARGET].values.astype(np.int32)
test_ids = test_df[ID_COL].values

# TE 정의
te_target_cols = [
    ('payment_type', 'payment_type_le'), ('employment_status', 'employment_status_le'),
    ('housing_status', 'housing_status_le'), ('device_os', 'device_os_le'),
    ('application_source', 'application_source_le'), ('age_bucket', 'age_bucket'),
    ('yearly_income', 'yearly_income'), ('req_credit_limit', 'req_credit_limit')]
te_cols = [(o, f) for o, f in te_target_cols if f in feature_cols]
te_names = [f'{o}_te' for o, _ in te_cols]
all_feat_names = feature_cols + te_names


def target_encode_fold(tr_df, val_df, test_full, col_orig, y_fold, smoothing=20):
    gm = y_fold.mean()
    tmp = tr_df.copy(); tmp['_y'] = y_fold
    s = tmp.groupby(col_orig)['_y'].agg(['count', 'mean'])
    s['sm'] = (s['count'] * s['mean'] + smoothing * gm) / (s['count'] + smoothing)
    m = s['sm'].to_dict()
    return (tr_df[col_orig].map(m).fillna(gm).values,
            val_df[col_orig].map(m).fillna(gm).values,
            test_full[col_orig].map(m).fillna(gm).values)


def build_te(tr_idx, val_idx):
    X_b = train_df[feature_cols].values.astype(np.float32)
    X_tb = test_df[feature_cols].values.astype(np.float32)
    trf, valf = train_df.iloc[tr_idx].copy(), train_df.iloc[val_idx].copy()
    yt = y[tr_idx]
    ttr, tvl, tte = [], [], []
    for o, _ in te_cols:
        a, b, c = target_encode_fold(trf, valf, test_df, o, yt, 20)
        ttr.append(a); tvl.append(b); tte.append(c)
    return (np.column_stack([X_b[tr_idx]] + ttr),
            np.column_stack([X_b[val_idx]] + tvl),
            np.column_stack([X_tb] + tte),
            yt, y[val_idx])


# ============================================================================
# Phase 2: XGB Seed Averaging (5 seeds × 5-Fold)
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 2: XGB Seed Averaging (5 seeds)")
print("=" * 70)

XGB_SEEDS = [42, 43, 44, 45, 46]

xgb_params_base = {
    'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'tree_method': 'hist', 'learning_rate': 0.03,
    'max_depth': 7, 'min_child_weight': 100,
    'scale_pos_weight': 1, 'subsample': 0.75,
    'colsample_bytree': 0.6, 'reg_alpha': 0.3,
    'reg_lambda': 2.0, 'gamma': 0.1,
    'nthread': -1, 'verbosity': 0,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 단일 seed OOF (비교용)
oof_xgb_single = np.zeros(len(y))
pred_xgb_single = np.zeros(len(test_ids))

# 다중 seed 평균
oof_xgb_avg = np.zeros(len(y))
pred_xgb_avg = np.zeros(len(test_ids))

for seed_idx, xgb_seed in enumerate(XGB_SEEDS):
    print(f"\n--- XGB Seed {xgb_seed} ({seed_idx+1}/{len(XGB_SEEDS)}) ---")
    t0 = time.time()

    params = xgb_params_base.copy()
    params['seed'] = xgb_seed

    oof_this_seed = np.zeros(len(y))
    pred_this_seed = np.zeros(len(test_ids))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
        X_tr, X_val, X_te, y_tr, y_val = build_te(tr_idx, val_idx)

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=all_feat_names)
        dvalid = xgb.DMatrix(X_val, label=y_val, feature_names=all_feat_names)
        dtest = xgb.DMatrix(X_te, feature_names=all_feat_names)

        model = xgb.train(
            params, dtrain, num_boost_round=15000,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=300, verbose_eval=0)

        oof_this_seed[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        pred_this_seed += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / N_FOLDS

    ap = average_precision_score(y, oof_this_seed)
    print(f"  Seed {xgb_seed} OOF AP: {ap:.6f} ({time.time()-t0:.0f}s)")

    oof_xgb_avg += oof_this_seed / len(XGB_SEEDS)
    pred_xgb_avg += pred_this_seed / len(XGB_SEEDS)

    if seed_idx == 0:
        oof_xgb_single = oof_this_seed.copy()
        pred_xgb_single = pred_this_seed.copy()

ap_single = average_precision_score(y, oof_xgb_single)
ap_avg = average_precision_score(y, oof_xgb_avg)
print(f"\n[XGB Seed Averaging 결과]")
print(f"  단일 seed (42):  {ap_single:.6f}")
print(f"  5-seed 평균:     {ap_avg:.6f}")
print(f"  Seed Avg 효과:   {ap_avg - ap_single:+.6f}")


# ============================================================================
# Phase 2B: CAT Seed Averaging (3 seeds × 5-Fold)
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 2B: CAT Seed Averaging (3 seeds)")
print("=" * 70)

CAT_SEEDS = [42, 43, 44]

cb_exclude = [TARGET, '_is_train', ID_COL,
    'prev_addr_months_clean', 'curr_addr_months_clean', 'bank_months_count_clean',
    'device_email_cnt_8w_clean', 'session_length_min_clean']
cb_exclude += [c + '_le' for c in cat_cols]
cb_exclude = [c for c in cb_exclude if c in df.columns]
cb_feature_cols = [c for c in df.columns if c not in cb_exclude]
cb_cat_indices = [cb_feature_cols.index(c) for c in cat_cols if c in cb_feature_cols]

X_cb = train_df[cb_feature_cols].copy()
X_test_cb = test_df[cb_feature_cols].copy()
for col in cat_cols:
    if col in X_cb.columns:
        X_cb[col] = X_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

oof_cat_avg = np.zeros(len(y))
pred_cat_avg = np.zeros(len(test_ids))
oof_cat_single = np.zeros(len(y))
pred_cat_single = np.zeros(len(test_ids))

for seed_idx, cat_seed in enumerate(CAT_SEEDS):
    print(f"\n--- CAT Seed {cat_seed} ({seed_idx+1}/{len(CAT_SEEDS)}) ---")
    t0 = time.time()

    oof_this = np.zeros(len(y))
    pred_this = np.zeros(len(test_ids))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cb, y)):
        model = CatBoostClassifier(
            iterations=10000, learning_rate=0.02, depth=8,
            l2_leaf_reg=3.0, scale_pos_weight=1, rsm=0.75,
            cat_features=cb_cat_indices, eval_metric='PRAUC',
            random_seed=cat_seed, verbose=0,
            early_stopping_rounds=200, task_type='CPU')

        model.fit(X_cb.iloc[tr_idx], y[tr_idx],
                  eval_set=(X_cb.iloc[val_idx], y[val_idx]),
                  use_best_model=True)

        oof_this[val_idx] = model.predict_proba(X_cb.iloc[val_idx])[:, 1]
        pred_this += model.predict_proba(X_test_cb)[:, 1] / N_FOLDS

    ap = average_precision_score(y, oof_this)
    print(f"  Seed {cat_seed} OOF AP: {ap:.6f} ({time.time()-t0:.0f}s)")

    oof_cat_avg += oof_this / len(CAT_SEEDS)
    pred_cat_avg += pred_this / len(CAT_SEEDS)

    if seed_idx == 0:
        oof_cat_single = oof_this.copy()
        pred_cat_single = pred_this.copy()

ap_cat_single = average_precision_score(y, oof_cat_single)
ap_cat_avg = average_precision_score(y, oof_cat_avg)
print(f"\n[CAT Seed Averaging 결과]")
print(f"  단일 seed (42):  {ap_cat_single:.6f}")
print(f"  3-seed 평균:     {ap_cat_avg:.6f}")
print(f"  Seed Avg 효과:   {ap_cat_avg - ap_cat_single:+.6f}")


# ============================================================================
# Phase 3: TabNet (5-Fold)
# ============================================================================

if TABNET_AVAILABLE:
    print("\n" + "=" * 70)
    print("PHASE 3: TabNet")
    print("=" * 70)

    oof_tabnet = np.zeros(len(y))
    pred_tabnet = np.zeros(len(test_ids))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        t0 = time.time()

        X_tr, X_val, X_te, y_tr, y_val = build_te(tr_idx, val_idx)

        # TabNet은 float64 필요
        X_tr = X_tr.astype(np.float64)
        X_val = X_val.astype(np.float64)
        X_te = X_te.astype(np.float64)

        # NaN 제거
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        tabnet = TabNetClassifier(
            n_d=32, n_a=32,            # decision/attention 차원
            n_steps=5,                  # attention step 수
            gamma=1.5,                  # feature reuse coefficient
            n_independent=2,            # independent GLU layers
            n_shared=2,                 # shared GLU layers
            lambda_sparse=1e-4,         # sparsity regularization
            momentum=0.3,
            clip_value=2.0,
            optimizer_fn=None,          # default Adam
            optimizer_params=dict(lr=0.02),
            scheduler_params={"step_size": 20, "gamma": 0.9},
            scheduler_fn=None,
            mask_type='entmax',
            seed=SEED,
            verbose=0,
        )

        # class weight를 sample_weight로 전달
        # weight=1이 최적이므로 균등 weight 사용
        tabnet.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=['average_precision'],
            max_epochs=100,
            patience=15,
            batch_size=4096,
            virtual_batch_size=512,
            drop_last=False,
        )

        oof_tabnet[val_idx] = tabnet.predict_proba(X_val)[:, 1]
        pred_tabnet += tabnet.predict_proba(X_te)[:, 1] / N_FOLDS

        fold_ap = average_precision_score(y_val, oof_tabnet[val_idx])
        print(f"  AP={fold_ap:.6f} | {time.time()-t0:.0f}s")

    oof_ap_tabnet = average_precision_score(y, oof_tabnet)
    print(f"\n[TabNet] OOF AP: {oof_ap_tabnet:.6f}")

    # 다양성 확인
    tab_xgb_corr = np.corrcoef(oof_tabnet, oof_xgb_avg)[0, 1]
    tab_cat_corr = np.corrcoef(oof_tabnet, oof_cat_avg)[0, 1]
    xgb_cat_corr = np.corrcoef(oof_xgb_avg, oof_cat_avg)[0, 1]
    print(f"  TabNet-XGB 상관: {tab_xgb_corr:.4f} (MLP 참고: 0.867)")
    print(f"  TabNet-CAT 상관: {tab_cat_corr:.4f} (MLP 참고: 0.884)")
    print(f"  XGB-CAT 상관:    {xgb_cat_corr:.4f}")
else:
    print("\n[TabNet 사용 불가 — Seed Averaging 결과만으로 진행]")
    oof_tabnet = None
    pred_tabnet = None
    oof_ap_tabnet = 0
    tab_xgb_corr = 0
    tab_cat_corr = 0
    xgb_cat_corr = np.corrcoef(oof_xgb_avg, oof_cat_avg)[0, 1]


# ============================================================================
# Phase 4: 앙상블 (효과 추적 포함)
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 4: 앙상블")
print("=" * 70)

# --- 4-1. Seed Averaging 효과 (TabNet 없이) ---
print("\n[4-1] Seed Averaging 효과 (XGB+CAT만)")

# 단일 seed XGB+CAT
best_ap_single = 0
best_w_single = (0.5, 0.5)
for w in np.arange(0.0, 1.05, 0.05):
    ap = average_precision_score(y, w * oof_xgb_single + (1-w) * oof_cat_single)
    if ap > best_ap_single:
        best_ap_single = ap
        best_w_single = (w, 1-w)
print(f"  단일seed XGB+CAT: {best_ap_single:.6f} (XGB={best_w_single[0]:.2f})")

# Seed Avg XGB+CAT
best_ap_seedavg = 0
best_w_seedavg = (0.5, 0.5)
for w in np.arange(0.0, 1.05, 0.05):
    ap = average_precision_score(y, w * oof_xgb_avg + (1-w) * oof_cat_avg)
    if ap > best_ap_seedavg:
        best_ap_seedavg = ap
        best_w_seedavg = (w, 1-w)
print(f"  SeedAvg XGB+CAT:  {best_ap_seedavg:.6f} (XGB={best_w_seedavg[0]:.2f})")
print(f"  Seed Avg 앙상블 효과: {best_ap_seedavg - best_ap_single:+.6f}")

# --- 4-2. TabNet 기여도 ---
if TABNET_AVAILABLE and oof_tabnet is not None:
    print("\n[4-2] TabNet 기여도")

    # XGB(SeedAvg) + CAT(SeedAvg) + TabNet 가중 평균
    best_ap_3 = 0
    best_w_3 = (1/3, 1/3, 1/3)
    for w1 in np.arange(0.0, 1.0, 0.05):
        for w2 in np.arange(0.0, 1.0 - w1, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0: continue
            blend = w1 * oof_xgb_avg + w2 * oof_cat_avg + w3 * oof_tabnet
            ap = average_precision_score(y, blend)
            if ap > best_ap_3:
                best_ap_3 = ap
                best_w_3 = (w1, w2, w3)

    print(f"  XGB+CAT+TabNet: {best_ap_3:.6f} (XGB={best_w_3[0]:.2f}, CAT={best_w_3[1]:.2f}, Tab={best_w_3[2]:.2f})")
    print(f"  TabNet 기여:    {best_ap_3 - best_ap_seedavg:+.6f}")

    # Rank 가중 평균
    print("\n[4-3] Rank 가중 평균")
    r_xgb = rankdata(oof_xgb_avg) / len(oof_xgb_avg)
    r_cat = rankdata(oof_cat_avg) / len(oof_cat_avg)
    r_tab = rankdata(oof_tabnet) / len(oof_tabnet)

    best_ap_3r = 0
    best_w_3r = (1/3, 1/3, 1/3)
    for w1 in np.arange(0.0, 1.0, 0.05):
        for w2 in np.arange(0.0, 1.0 - w1, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0: continue
            blend = w1 * r_xgb + w2 * r_cat + w3 * r_tab
            ap = average_precision_score(y, blend)
            if ap > best_ap_3r:
                best_ap_3r = ap
                best_w_3r = (w1, w2, w3)

    print(f"  Rank XGB+CAT+Tab: {best_ap_3r:.6f} (XGB={best_w_3r[0]:.2f}, CAT={best_w_3r[1]:.2f}, Tab={best_w_3r[2]:.2f})")

    # Stacking
    print("\n[4-4] Stacking")
    meta_X = np.column_stack([oof_xgb_avg, oof_cat_avg, oof_tabnet])
    meta_test = np.column_stack([pred_xgb_avg, pred_cat_avg, pred_tabnet])

    oof_stack = np.zeros(len(y))
    pred_stack = np.zeros(len(test_ids))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(meta_X, y)):
        mm = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        mm.fit(meta_X[tr_idx], y[tr_idx])
        oof_stack[val_idx] = mm.predict_proba(meta_X[val_idx])[:, 1]
        pred_stack += mm.predict_proba(meta_test)[:, 1] / N_FOLDS
    stack_ap = average_precision_score(y, oof_stack)
    print(f"  Stacking AP: {stack_ap:.6f}")

else:
    best_ap_3 = best_ap_seedavg
    best_w_3 = (best_w_seedavg[0], best_w_seedavg[1], 0)
    best_ap_3r = best_ap_seedavg
    best_w_3r = best_w_3
    stack_ap = best_ap_seedavg
    pred_stack = pred_xgb_avg * best_w_seedavg[0] + pred_cat_avg * best_w_seedavg[1]


# ============================================================================
# Phase 5: 최종 결과
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 5: 최종 결과")
print("=" * 70)

# 제출 예측값 계산
pred_single = best_w_single[0]*pred_xgb_single + best_w_single[1]*pred_cat_single
pred_sa = best_w_seedavg[0]*pred_xgb_avg + best_w_seedavg[1]*pred_cat_avg

results = {
    'XGB_single':          (ap_single, pred_xgb_single),
    'XGB_5seed':           (ap_avg, pred_xgb_avg),
    'CAT_single':          (ap_cat_single, pred_cat_single),
    'CAT_3seed':           (ap_cat_avg, pred_cat_avg),
    'XGB+CAT_single':      (best_ap_single, pred_single),
    'XGB+CAT_seedavg':     (best_ap_seedavg, pred_sa),
}

if TABNET_AVAILABLE and oof_tabnet is not None:
    pred_3wa = best_w_3[0]*pred_xgb_avg + best_w_3[1]*pred_cat_avg + best_w_3[2]*pred_tabnet

    pr_xgb = rankdata(pred_xgb_avg) / len(pred_xgb_avg)
    pr_cat = rankdata(pred_cat_avg) / len(pred_cat_avg)
    pr_tab = rankdata(pred_tabnet) / len(pred_tabnet)
    pred_3r = best_w_3r[0]*pr_xgb + best_w_3r[1]*pr_cat + best_w_3r[2]*pr_tab

    results['TabNet'] = (oof_ap_tabnet, pred_tabnet)
    results['XGB+CAT+Tab_WA'] = (best_ap_3, pred_3wa)
    results['XGB+CAT+Tab_Rank'] = (best_ap_3r, pred_3r)
    results['XGB+CAT+Tab_Stack'] = (stack_ap, pred_stack)

best_name = max(results, key=lambda k: results[k][0])
best_ap_overall = results[best_name][0]

print(f"\n  {'방식':<24s}  {'AP':>10s}")
print(f"  {'-'*40}")
for name, (ap, _) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<24s}  {ap:.6f}{marker}")

print(f"\n  [역대 AP]")
print(f"  v2:  0.174014 (LB 0.17557)")
print(f"  v5:  0.177913")
print(f"  v6:  0.177950 (LB 0.17972)")
print(f"  v7:  0.178483")
print(f"  v8:  {best_ap_overall:.6f} ({best_name})")
print(f"  v8 vs v7: {best_ap_overall - 0.178483:+.6f}")

print(f"\n  [효과 분해]")
print(f"  Seed Averaging:   {best_ap_seedavg - best_ap_single:+.6f}")
if TABNET_AVAILABLE and oof_tabnet is not None:
    print(f"  TabNet 기여:      {best_ap_3 - best_ap_seedavg:+.6f}")
    print(f"  TabNet-XGB 상관:  {tab_xgb_corr:.4f}")
    print(f"  TabNet-CAT 상관:  {tab_cat_corr:.4f}")

# 제출 파일
best_pred = results[best_name][1]
submission = pd.DataFrame({'id': test_ids.astype(int), 'fraud': best_pred})
output_path = os.path.join(OUTPUT_DIR, 'submission_v8.csv')
submission.to_csv(output_path, index=False)

print(f"\n[제출 파일]")
print(f"  방식: {best_name}")
print(f"  파일: {output_path}")
print(f"  예측값 범위: [{best_pred.min():.6f}, {best_pred.max():.6f}]")
print(f"  예측값 평균: {best_pred.mean():.6f}")

# 모든 방식 파일
for name, (ap, pred) in results.items():
    if pred is not None:
        safe = name.replace('+', '_').replace(' ', '_')
        pd.DataFrame({'id': test_ids.astype(int), 'fraud': pred}).to_csv(
            os.path.join(OUTPUT_DIR, f'submission_v8_{safe}.csv'), index=False)

total = time.time() - start_time
print(f"\n총 실행 시간: {total/60:.1f}분")

print("\n" + "=" * 70)
print("v8 완료!")
print("=" * 70)
print(f"\n[판단 가이드]")
print(f"  v8 BEST: {best_name} (AP={best_ap_overall:.6f})")
print(f"  v7 BEST: 0.178483")
print(f"  → v8 > v7이면 제출")
print(f"  → 출력 전체를 Claude Opus에 붙여넣기")
