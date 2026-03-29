"""
================================================================================
v13 — MODEL ZOO: 다양한 관점의 앙상블
================================================================================
핵심: 같은 모델을 seed만 바꾸는 게 아니라,
      하이퍼파라미터 자체를 바꿔서 "다른 안경"으로 보게 한다.

Model Zoo 구성 (6타입 × 3seed = 18모델):
  1. XGB_shallow: depth=5, lr=0.05   (거친 패턴, 과적합 강함)
  2. XGB_standard: depth=7, lr=0.03  (현재 최고, 검증됨)
  3. XGB_deep: depth=9, lr=0.02, mcw=250 (미세 패턴, 강한 정규화)
  4. XGB_blind: depth=7, top피처 제거  (2차 패턴 전담)
  5. CAT: depth=8, lr=0.02           (범주형 특화)
  6. LGB: leaves=63, lr=0.03         (Leaf-wise 다양성)

앙상블: scipy.optimize로 AP 최대화 가중치 역산
최종: v13 Rank + v8 Rank Blend

실행 결과:
======================================================================
v13 — MODEL ZOO ENSEMBLE
======================================================================

[1] 데이터 + 피처
  train: (700000, 33), test: (300000, 32)
  피처 완료
  전체 피처: 97개
  Blind 피처: 95개 (제거: ['net_risk', 'he_credit_avg'])

======================================================================
MODEL ZOO 학습
======================================================================

--- XGB_shallow (3 seeds) ---
  Seed 42: AP=0.176240 (62s)
  Seed 43: AP=0.177067 (66s)
  Seed 44: AP=0.176070 (67s)
  XGB_shallow 3seed AP: 0.177785 (195s)

--- XGB_standard (3 seeds) ---
  Seed 42: AP=0.178065 (93s)
  Seed 43: AP=0.177525 (89s)
  Seed 44: AP=0.177488 (90s)
  XGB_standard 3seed AP: 0.178611 (272s)

--- XGB_deep (3 seeds) ---
  Seed 42: AP=0.175852 (146s)
  Seed 43: AP=0.176017 (147s)
  Seed 44: AP=0.175639 (145s)
  XGB_deep 3seed AP: 0.176213 (439s)

--- XGB_blind (3 seeds) ---
  Seed 42: AP=0.176915 (87s)
  Seed 43: AP=0.178309 (91s)
  Seed 44: AP=0.177912 (88s)
  XGB_blind 3seed AP: 0.178661 (266s)

--- CAT (3 seeds) ---
  Seed 42: AP=0.173813 (3119s)
  Seed 43: AP=0.174252 (3658s)
  Seed 44: AP=0.173627 (3417s)
  CAT 3seed AP: 0.175116 (10194s)

--- LGB (3 seeds) ---
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[304]	valid_0's average_precision: 0.169129
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[311]	valid_0's average_precision: 0.178007
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[247]	valid_0's average_precision: 0.173805
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[205]	valid_0's average_precision: 0.167001
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[458]	valid_0's average_precision: 0.165445
  Seed 42: AP=0.169848 (325s)
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[207]	valid_0's average_precision: 0.167857
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[323]	valid_0's average_precision: 0.177035
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[341]	valid_0's average_precision: 0.177812
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[335]	valid_0's average_precision: 0.167516
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[302]	valid_0's average_precision: 0.164712
  Seed 43: AP=0.170076 (322s)
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[222]	valid_0's average_precision: 0.167393
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[348]	valid_0's average_precision: 0.178943
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[201]	valid_0's average_precision: 0.174638
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[374]	valid_0's average_precision: 0.168791
Training until validation scores don't improve for 300 rounds
Early stopping, best iteration is:
[281]	valid_0's average_precision: 0.165055
  Seed 44: AP=0.170022 (315s)
  LGB 3seed AP: 0.172725 (962s)

======================================================================
다양성 분석
======================================================================

                   XGB_shal  XGB_stan  XGB_deep  XGB_blin       CAT       LGB
      XGB_shallow    1.0000    0.9940    0.9791    0.9920    0.9701    0.9803
     XGB_standard    0.9940    1.0000    0.9853    0.9958    0.9707    0.9815
         XGB_deep    0.9791    0.9853    1.0000    0.9848    0.9607    0.9691
        XGB_blind    0.9920    0.9958    0.9848    1.0000    0.9705    0.9796
              CAT    0.9701    0.9707    0.9607    0.9705    1.0000    0.9661
              LGB    0.9803    0.9815    0.9691    0.9796    0.9661    1.0000

======================================================================
최적 가중치 역산 (scipy.optimize)
======================================================================

  [최적 Rank 가중치]
      XGB_shallow: 0.1650
     XGB_standard: 0.1448
         XGB_deep: 0.2182
        XGB_blind: 0.2387
              CAT: 0.2332
              LGB: 0.0002

  최적 Rank Blend AP: 0.179625
  ⚠ LGB 가중치 < 0.01 → 앙상블에서 거의 기여 없음

  [최적 WA 가중치]
      XGB_shallow: 0.1610
     XGB_standard: 0.1533
         XGB_deep: 0.2007
        XGB_blind: 0.2272
              CAT: 0.2578
              LGB: 0.0001
  최적 WA AP: 0.179770

  XGB_std+CAT baseline: 0.179350 (XGB=0.65)

======================================================================
Test 예측 + v8 Blend
======================================================================

  방식                                OOF AP
  --------------------------------------------
  Zoo_WA                        0.179770 ← BEST
  Zoo_Rank                      0.179625
  XGB_std+CAT                   0.179350
  XGB_blind                     0.178661
  XGB_standard                  0.178611
  XGB_shallow                   0.177785
  XGB_deep                      0.176213
  CAT                           0.175116
  LGB                           0.172725
  Zoo(95)+v8(5)                    N/A
  Zoo(90)+v8(9)                    N/A
  Zoo(85)+v8(15)                   N/A
  Zoo(80)+v8(19)                   N/A
  Zoo(70)+v8(30)                   N/A
  Zoo(60)+v8(40)                   N/A
  Zoo(50)+v8(50)                   N/A
  ZooWA(90)+v8(9)                  N/A
  ZooWA(80)+v8(19)              0.181704(Public) / 0.177871(Private) -> 최종 제출
  ZooWA(70)+v8(30)                 N/A
  ZooWA(60)+v8(40)                 N/A
  ZooWA(50)+v8(50)                 N/A

  [역대 AP]
  v8:  0.179350 (LB 0.18134) ← 현재 LB 최고
  v13: 0.179770 (Zoo_WA)
  v13 vs v8: +0.000420

  Zoo 다양성 효과: +0.000419 (vs XGB_std+CAT)

[제출 파일]
  메인: submission_v13.csv (Zoo_WA)

[Output 파일 확인]
  ✓ submission_v13.csv (8215KB)
  ✓ submission_v13_CAT.csv (8214KB)
  ✓ submission_v13_LGB.csv (8200KB)
  ✓ submission_v13_XGB_blind.csv (8216KB)
  ✓ submission_v13_XGB_deep.csv (8216KB)
  ✓ submission_v13_XGB_shallow.csv (8219KB)
  ✓ submission_v13_XGB_standard.csv (8217KB)
  ✓ submission_v13_XGB_std_CAT.csv (8214KB)
  ✓ submission_v13_Zoo50_v850.csv (6833KB)
  ✓ submission_v13_Zoo60_v840.csv (6898KB)
  ✓ submission_v13_Zoo70_v830.csv (6911KB)
  ✓ submission_v13_Zoo80_v819.csv (6911KB)
  ✓ submission_v13_Zoo85_v815.csv (6960KB)
  ✓ submission_v13_Zoo90_v89.csv (6928KB)
  ✓ submission_v13_Zoo95_v85.csv (6976KB)
  ✓ submission_v13_ZooWA50_v850.csv (6828KB)
  ✓ submission_v13_ZooWA60_v840.csv (6896KB)
  ✓ submission_v13_ZooWA70_v830.csv (6915KB)
  ✓ submission_v13_ZooWA80_v819.csv (6905KB)
  ✓ submission_v13_ZooWA90_v89.csv (6931KB)
  ✓ submission_v13_Zoo_Rank.csv (7587KB)
  ✓ submission_v13_Zoo_WA.csv (8215KB)

총 실행 시간: 208.6분

======================================================================
v13 완료!
======================================================================

================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from scipy.stats import rankdata
from scipy.optimize import minimize
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings
import os
import time

warnings.filterwarnings('ignore')

DATA_DIR = '/kaggle/input/datasets/parkhyenwoong/cuai-data/'
OUTPUT_DIR = '/kaggle/working/'
SEED = 42
N_FOLDS = 5
N_SEEDS = 3  # 타입당 seed 수
np.random.seed(SEED)
start_time = time.time()

print("=" * 70)
print("v13 — MODEL ZOO ENSEMBLE")
print("=" * 70)


# ============================================================================
# 1. 데이터 + 피처 (v8 동일, 97개)
# ============================================================================

print("\n[1] 데이터 + 피처")
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

TARGET = 'fraud_bool'
ID_COL = 'id'
print(f"  train: {train.shape}, test: {test.shape}")

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

# v8 피처 전체 + 가속도
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

df['rate_acceleration_6h'] = df['req_rate_6h'] - (df['req_rate_24h'] / 4)
df['credit_limit_per_age'] = df['req_credit_limit'] / (df['age_bucket'] + 1)
df['session_per_transfer'] = df['session_length_min'].replace(-1, np.nan).fillna(0) / (np.abs(df['init_transfer_amt']) + 1)

print("  피처 완료")

# 피처 선택
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

# 피처 배깅용: top 2 피처 제거 버전
blind_remove = ['net_risk', 'he_credit_avg']
feature_cols_blind = [c for c in feature_cols if c not in blind_remove]

print(f"  전체 피처: {len(feature_cols)}개")
print(f"  Blind 피처: {len(feature_cols_blind)}개 (제거: {blind_remove})")

train_df = df[df['_is_train'] == 1].reset_index(drop=True)
test_df = df[df['_is_train'] == 0].reset_index(drop=True)
y = train_df[TARGET].values.astype(np.int32)
test_ids = test_df[ID_COL].values

te_target_cols = [
    ('payment_type', 'payment_type_le'), ('employment_status', 'employment_status_le'),
    ('housing_status', 'housing_status_le'), ('device_os', 'device_os_le'),
    ('application_source', 'application_source_le'), ('age_bucket', 'age_bucket'),
    ('yearly_income', 'yearly_income'), ('req_credit_limit', 'req_credit_limit')]


def target_encode_fold(tr_df, val_df, test_full, col_orig, y_fold, smoothing=20):
    gm = y_fold.mean()
    tmp = tr_df.copy(); tmp['_y'] = y_fold
    s = tmp.groupby(col_orig)['_y'].agg(['count', 'mean'])
    s['sm'] = (s['count'] * s['mean'] + smoothing * gm) / (s['count'] + smoothing)
    m = s['sm'].to_dict()
    return (tr_df[col_orig].map(m).fillna(gm).values,
            val_df[col_orig].map(m).fillna(gm).values,
            test_full[col_orig].map(m).fillna(gm).values)


def build_te(tr_idx, val_idx, feat_list=None):
    if feat_list is None:
        feat_list = feature_cols
    te_valid = [(o, f) for o, f in te_target_cols if f in feat_list]
    te_n = [f'{o}_te' for o, _ in te_valid]

    X_b = train_df[feat_list].values.astype(np.float32)
    X_tb = test_df[feat_list].values.astype(np.float32)
    trf, valf = train_df.iloc[tr_idx].copy(), train_df.iloc[val_idx].copy()
    yt = y[tr_idx]
    ttr, tvl, tte = [], [], []
    for o, _ in te_valid:
        a, b, c = target_encode_fold(trf, valf, test_df, o, yt, 20)
        ttr.append(a); tvl.append(b); tte.append(c)
    fn = feat_list + te_n
    return (np.column_stack([X_b[tr_idx]] + ttr),
            np.column_stack([X_b[val_idx]] + tvl),
            np.column_stack([X_tb] + tte), yt, y[val_idx], fn)


# ============================================================================
# 2. MODEL ZOO 학습
# ============================================================================

print("\n" + "=" * 70)
print("MODEL ZOO 학습")
print("=" * 70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 모델 타입별 파라미터 정의
model_configs = {
    'XGB_shallow': {
        'type': 'xgb',
        'params': {
            'objective': 'binary:logistic', 'eval_metric': 'aucpr',
            'tree_method': 'hist', 'device': 'cuda',
            'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 80,
            'scale_pos_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0.05,
            'nthread': -1, 'verbosity': 0,
        },
        'features': 'full',
    },
    'XGB_standard': {
        'type': 'xgb',
        'params': {
            'objective': 'binary:logistic', 'eval_metric': 'aucpr',
            'tree_method': 'hist', 'device': 'cuda',
            'learning_rate': 0.03, 'max_depth': 7, 'min_child_weight': 100,
            'scale_pos_weight': 1, 'subsample': 0.75, 'colsample_bytree': 0.6,
            'reg_alpha': 0.3, 'reg_lambda': 2.0, 'gamma': 0.1,
            'nthread': -1, 'verbosity': 0,
        },
        'features': 'full',
    },
    'XGB_deep': {
        'type': 'xgb',
        'params': {
            'objective': 'binary:logistic', 'eval_metric': 'aucpr',
            'tree_method': 'hist', 'device': 'cuda',
            'learning_rate': 0.02, 'max_depth': 9, 'min_child_weight': 250,
            'scale_pos_weight': 1, 'subsample': 0.7, 'colsample_bytree': 0.5,
            'reg_alpha': 0.5, 'reg_lambda': 3.0, 'gamma': 0.2,
            'nthread': -1, 'verbosity': 0,
        },
        'features': 'full',
    },
    'XGB_blind': {
        'type': 'xgb',
        'params': {
            'objective': 'binary:logistic', 'eval_metric': 'aucpr',
            'tree_method': 'hist', 'device': 'cuda',
            'learning_rate': 0.03, 'max_depth': 7, 'min_child_weight': 100,
            'scale_pos_weight': 1, 'subsample': 0.75, 'colsample_bytree': 0.6,
            'reg_alpha': 0.3, 'reg_lambda': 2.0, 'gamma': 0.1,
            'nthread': -1, 'verbosity': 0,
        },
        'features': 'blind',  # top 피처 제거
    },
    'CAT': {
        'type': 'cat',
        'features': 'full',
    },
    'LGB': {
        'type': 'lgb',
        'params': {
            'objective': 'binary', 'metric': 'average_precision',
            'learning_rate': 0.03, 'num_leaves': 63,
            'min_child_samples': 150, 'scale_pos_weight': 1,
            'feature_fraction': 0.7, 'bagging_fraction': 0.75,
            'bagging_freq': 1, 'reg_alpha': 0.3, 'reg_lambda': 2.0,
            'verbosity': -1,
        },
        'features': 'full',
    },
}

SEEDS = [42, 43, 44]

# 결과 저장
zoo_oof = {}   # model_name → oof array
zoo_pred = {}  # model_name → test pred array

for model_name, config in model_configs.items():
    print(f"\n--- {model_name} ({N_SEEDS} seeds) ---")
    t_model = time.time()

    oof_avg = np.zeros(len(y))
    pred_avg = np.zeros(len(test_ids))

    feat_list = feature_cols if config['features'] == 'full' else feature_cols_blind

    for seed_idx, seed in enumerate(SEEDS):
        t0 = time.time()
        oof_this = np.zeros(len(y))
        pred_this = np.zeros(len(test_ids))

        if config['type'] == 'xgb':
            params = config['params'].copy()
            params['seed'] = seed

            for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
                X_tr, X_val, X_te, y_tr, y_val, fn = build_te(tr_idx, val_idx, feat_list)
                dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=fn)
                dvalid = xgb.DMatrix(X_val, label=y_val, feature_names=fn)
                dtest = xgb.DMatrix(X_te, feature_names=fn)

                m = xgb.train(params, dtrain, num_boost_round=15000,
                               evals=[(dvalid, 'v')],
                               early_stopping_rounds=300, verbose_eval=0)

                oof_this[val_idx] = m.predict(dvalid, iteration_range=(0, m.best_iteration + 1))
                pred_this += m.predict(dtest, iteration_range=(0, m.best_iteration + 1)) / N_FOLDS

        elif config['type'] == 'cat':
            cb_exclude = [TARGET, '_is_train', ID_COL,
                'prev_addr_months_clean', 'curr_addr_months_clean', 'bank_months_count_clean',
                'device_email_cnt_8w_clean', 'session_length_min_clean']
            cb_exclude += [c + '_le' for c in cat_cols]
            cb_exclude = [c for c in cb_exclude if c in df.columns]
            cb_feat = [c for c in df.columns if c not in cb_exclude]
            cb_cat_idx = [cb_feat.index(c) for c in cat_cols if c in cb_feat]

            X_c = train_df[cb_feat].copy()
            X_tc = test_df[cb_feat].copy()
            for col in cat_cols:
                if col in X_c.columns:
                    X_c[col] = X_c[col].astype(str)
                    X_tc[col] = X_tc[col].astype(str)

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_c, y)):
                try:
                    m = CatBoostClassifier(
                        iterations=10000, learning_rate=0.02, depth=8,
                        l2_leaf_reg=3.0, scale_pos_weight=1, rsm=0.75,
                        cat_features=cb_cat_idx, eval_metric='PRAUC',
                        random_seed=seed, verbose=0,
                        early_stopping_rounds=200, task_type='GPU')
                    m.fit(X_c.iloc[tr_idx], y[tr_idx],
                          eval_set=(X_c.iloc[val_idx], y[val_idx]), use_best_model=True)
                except Exception:
                    m = CatBoostClassifier(
                        iterations=10000, learning_rate=0.02, depth=8,
                        l2_leaf_reg=3.0, scale_pos_weight=1, rsm=0.75,
                        cat_features=cb_cat_idx, eval_metric='PRAUC',
                        random_seed=seed, verbose=0,
                        early_stopping_rounds=200, task_type='CPU')
                    m.fit(X_c.iloc[tr_idx], y[tr_idx],
                          eval_set=(X_c.iloc[val_idx], y[val_idx]), use_best_model=True)

                oof_this[val_idx] = m.predict_proba(X_c.iloc[val_idx])[:, 1]
                pred_this += m.predict_proba(X_tc)[:, 1] / N_FOLDS

        elif config['type'] == 'lgb':
            params = config['params'].copy()

            for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
                X_tr, X_val, X_te, y_tr, y_val, fn = build_te(tr_idx, val_idx, feat_list)

                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

                p = params.copy()
                p['seed'] = seed

                m = lgb.train(p, dtrain, num_boost_round=15000,
                               valid_sets=[dvalid], callbacks=[
                                   lgb.early_stopping(300),
                                   lgb.log_evaluation(0)])

                oof_this[val_idx] = m.predict(X_val)
                pred_this += m.predict(X_te) / N_FOLDS

        ap = average_precision_score(y, oof_this)
        print(f"  Seed {seed}: AP={ap:.6f} ({time.time()-t0:.0f}s)")

        oof_avg += oof_this / N_SEEDS
        pred_avg += pred_this / N_SEEDS

    ap_avg = average_precision_score(y, oof_avg)
    print(f"  {model_name} 3seed AP: {ap_avg:.6f} ({time.time()-t_model:.0f}s)")

    zoo_oof[model_name] = oof_avg
    zoo_pred[model_name] = pred_avg


# ============================================================================
# 3. 다양성 분석
# ============================================================================

print("\n" + "=" * 70)
print("다양성 분석")
print("=" * 70)

model_names = list(zoo_oof.keys())
print(f"\n  {'':>15s}", end='')
for n in model_names:
    print(f"  {n[:8]:>8s}", end='')
print()

for i, n1 in enumerate(model_names):
    print(f"  {n1:>15s}", end='')
    for j, n2 in enumerate(model_names):
        if i == j:
            print(f"  {'1.0000':>8s}", end='')
        else:
            corr = np.corrcoef(zoo_oof[n1], zoo_oof[n2])[0, 1]
            print(f"  {corr:>8.4f}", end='')
    print()


# ============================================================================
# 4. scipy.optimize로 최적 가중치 역산
# ============================================================================

print("\n" + "=" * 70)
print("최적 가중치 역산 (scipy.optimize)")
print("=" * 70)

# OOF를 Rank로 변환
oof_ranks = {}
for name in model_names:
    oof_ranks[name] = rankdata(zoo_oof[name]) / len(y)

# AP를 최대화하는 가중치 찾기 (Nelder-Mead)
def neg_ap_rank(weights):
    """가중치로 Rank blend한 뒤 -AP 반환 (minimize용)"""
    w = np.abs(weights)  # 음수 방지
    w = w / w.sum()      # 합 1로 정규화

    blend = np.zeros(len(y))
    for i, name in enumerate(model_names):
        blend += w[i] * oof_ranks[name]

    return -average_precision_score(y, blend)

# 초기값: 균등
w0 = np.ones(len(model_names)) / len(model_names)

# 최적화
result = minimize(neg_ap_rank, w0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})

w_opt = np.abs(result.x)
w_opt = w_opt / w_opt.sum()

print(f"\n  [최적 Rank 가중치]")
for name, w in zip(model_names, w_opt):
    print(f"  {name:>15s}: {w:.4f}")

ap_opt_rank = -result.fun
print(f"\n  최적 Rank Blend AP: {ap_opt_rank:.6f}")

# 가중치가 0.01 미만인 모델 식별
for name, w in zip(model_names, w_opt):
    if w < 0.01:
        print(f"  ⚠ {name} 가중치 < 0.01 → 앙상블에서 거의 기여 없음")

# 일반 가중 평균도 최적화
def neg_ap_wa(weights):
    w = np.abs(weights)
    w = w / w.sum()
    blend = np.zeros(len(y))
    for i, name in enumerate(model_names):
        blend += w[i] * zoo_oof[name]
    return -average_precision_score(y, blend)

result_wa = minimize(neg_ap_wa, w0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})
w_opt_wa = np.abs(result_wa.x)
w_opt_wa = w_opt_wa / w_opt_wa.sum()

ap_opt_wa = -result_wa.fun
print(f"\n  [최적 WA 가중치]")
for name, w in zip(model_names, w_opt_wa):
    print(f"  {name:>15s}: {w:.4f}")
print(f"  최적 WA AP: {ap_opt_wa:.6f}")

# XGB+CAT만 (baseline 비교)
oof_xc = np.zeros(len(y))
pred_xc = np.zeros(len(test_ids))
best_ap_xc = 0
best_w_xc = 0.5
for w in np.arange(0.0, 1.005, 0.01):
    blend = w * zoo_oof['XGB_standard'] + (1-w) * zoo_oof['CAT']
    ap = average_precision_score(y, blend)
    if ap > best_ap_xc:
        best_ap_xc = ap
        best_w_xc = w
print(f"\n  XGB_std+CAT baseline: {best_ap_xc:.6f} (XGB={best_w_xc:.2f})")


# ============================================================================
# 5. Test 예측 생성 + v8 Blend
# ============================================================================

print("\n" + "=" * 70)
print("Test 예측 + v8 Blend")
print("=" * 70)

# 최적 Rank Blend test 예측
pred_ranks = {}
for name in model_names:
    pred_ranks[name] = rankdata(zoo_pred[name]) / len(test_ids)

pred_opt_rank = np.zeros(len(test_ids))
for i, name in enumerate(model_names):
    pred_opt_rank += w_opt[i] * pred_ranks[name]

# 최적 WA test 예측
pred_opt_wa = np.zeros(len(test_ids))
for i, name in enumerate(model_names):
    pred_opt_wa += w_opt_wa[i] * zoo_pred[name]

# v8 Blend
v8_path = os.path.join(DATA_DIR, 'submission_v8.csv')
results = {
    'Zoo_Rank': (ap_opt_rank, pred_opt_rank),
    'Zoo_WA': (ap_opt_wa, pred_opt_wa),
    'XGB_std+CAT': (best_ap_xc, best_w_xc * zoo_pred['XGB_standard'] + (1-best_w_xc) * zoo_pred['CAT']),
}

# 개별 모델
for name in model_names:
    ap = average_precision_score(y, zoo_oof[name])
    results[name] = (ap, zoo_pred[name])

if os.path.exists(v8_path):
    v8_pred = pd.read_csv(v8_path)['fraud'].values
    r_v8 = rankdata(v8_pred) / len(v8_pred)
    r_zoo = rankdata(pred_opt_rank) / len(pred_opt_rank)

    for ratio in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        blended = ratio * r_zoo + (1 - ratio) * r_v8
        name = f'Zoo({int(ratio*100)})+v8({int((1-ratio)*100)})'
        results[name] = (None, blended)

    # v8 + Zoo_WA blend
    r_zoo_wa = rankdata(pred_opt_wa) / len(pred_opt_wa)
    for ratio in [0.90, 0.80, 0.70, 0.60, 0.50]:
        blended = ratio * r_zoo_wa + (1 - ratio) * r_v8
        name = f'ZooWA({int(ratio*100)})+v8({int((1-ratio)*100)})'
        results[name] = (None, blended)

    V8_AVAILABLE = True
else:
    V8_AVAILABLE = False

# 최종 결과
best_name = max((k for k, v in results.items() if v[0] is not None),
                key=lambda k: results[k][0])
best_ap = results[best_name][0]

print(f"\n  {'방식':<28s}  {'OOF AP':>10s}")
print(f"  {'-'*44}")
for name, (ap, _) in sorted(results.items(), key=lambda x: (x[1][0] or 0), reverse=True):
    ap_str = f"{ap:.6f}" if ap is not None else "   N/A"
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<28s}  {ap_str}{marker}")

print(f"\n  [역대 AP]")
print(f"  v8:  0.179350 (LB 0.18134) ← 현재 LB 최고")
print(f"  v13: {best_ap:.6f} ({best_name})")
print(f"  v13 vs v8: {best_ap - 0.179350:+.6f}")

zoo_effect = best_ap - best_ap_xc
print(f"\n  Zoo 다양성 효과: {zoo_effect:+.6f} (vs XGB_std+CAT)")

# 제출 파일
for name, (ap, pred) in results.items():
    if pred is not None:
        safe = name.replace('+', '_').replace('(', '').replace(')', '').replace(' ', '')
        fname = f'submission_v13_{safe}.csv'
        pd.DataFrame({'id': test_ids.astype(int), 'fraud': pred}).to_csv(
            os.path.join(OUTPUT_DIR, fname), index=False)

# 메인 제출
best_pred = results[best_name][1]
pd.DataFrame({'id': test_ids.astype(int), 'fraud': best_pred}).to_csv(
    os.path.join(OUTPUT_DIR, 'submission_v13.csv'), index=False)

print(f"\n[제출 파일]")
print(f"  메인: submission_v13.csv ({best_name})")

# Output 확인
print(f"\n[Output 파일 확인]")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.csv'):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  ✓ {f} ({size/1024:.0f}KB)")

total = time.time() - start_time
print(f"\n총 실행 시간: {total/60:.1f}분")

print("\n" + "=" * 70)
print("v13 완료!")
print("=" * 70)
print(f"\n[내일 제출 전략]")
print(f"  1회: submission_v13.csv (Zoo 최적)")
print(f"  2회: Zoo+v8 blend (80/20 또는 70/30)")
print(f"  3회: XGB_std+CAT (안전한 baseline)")
print(f"  4~5회: 1~3회 결과 보고 미세 조정")
print(f"\n  ★ Save Version → Save & Run All!")
