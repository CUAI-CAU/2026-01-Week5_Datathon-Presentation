import pandas as pd
import numpy as np
import os

# 1. 설정
OUT_DIR = "./output"
files = ["submission_xgboost_raw.csv", "submission.csv"]
weights = [0.5, 0.5] 
p = 2

dfs = [pd.read_csv(os.path.join(OUT_DIR, f)).sort_values('id') for f in files]

ensemble_df = dfs[0][['id']].copy()

weighted_sum = 0
for df, w in zip(dfs, weights):
    weighted_sum += (df['fraud'] ** p) * w

final_pred = (weighted_sum / sum(weights)) ** (1/p)

ensemble_df['fraud'] = final_pred

ensemble_df.to_csv(f"{OUT_DIR}/super_ensemble_submission.csv", index=False)
