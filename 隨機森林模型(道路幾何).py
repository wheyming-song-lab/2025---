# -*- coding: utf-8 -*-
"""
五個隨機 block RF（僅測試集指標）+ 平均特徵重要性（中文顯示）
輸出：
- rf_5blocks_outputs/metrics_5blocks.csv
- rf_5blocks_outputs/avg_feature_importance.csv（含中文名稱）
- rf_5blocks_outputs/avg_importance_topk.png（前 TOPK 名，中文標籤）
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import platform

# --- Font settings for plot ---
if platform.system() != "Windows":
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans", "sans-serif"]
else:
    plt.rcParams["font.family"] = "Microsoft JhengHei"


# -*- coding: utf-8 -*-
"""
五個隨機 block RF（僅測試集指標，平均+標準誤）+ 平均特徵重要性（中文顯示）
輸出：
- rf_5blocks_outputs/metrics_5blocks.csv
- rf_5blocks_outputs/avg_feature_importance.csv（含中文名稱）
- rf_5blocks_outputs/avg_importance_topk.png（前 TOPK 名，中文標籤）
"""

# ===== 可調參數 =====
DATA_PATH     = "鶯歌到高原道路幾何資料.csv"
TARGET        = "tti_sipt_normalized"
N_ESTIMATORS  = 500
N_BLOCKS      = 5
RANDOM_SEEDS  = [42, 43, 44, 45, 46]  # 每個 block 的隨機種子
DECIMALS      = 6                     # CSV 保留位數
TOPK          = 10                     # 顯示前幾名平均重要性
MAKE_PLOT     = True

# 要排除的欄位（依你先前設定）
EXCLUDE = [
    'Num', 'direction', 'Mileage', 'tti_sipt', TARGET,
    'tti_sipt_adjusted', 'count', 'Curvature_Severity', 'Lane_Density',
    'Lane_Count', 'Total_Road_Width'
]

# ===== 英文→中文 對照（缺的會自動回退英文）=====
NAME_MAP = {
    'Road_Width': '路幅寬', 'Total_Road_Width': '總道路寬度',
    'Lane_Count': '車道數', 'Total_Lanes': '總車道數',
    'Aux_Lane_Count': '輔助車道數', 'Main_Lane_Count': '主線車道數',
    'Channelized_Area_Width': '導引島寬度', 'Inner_Shoulder_Width': '內路肩寬',
    'Outer_Shoulder_Width': '外路肩寬', 'Curvature_Radius': '曲率半徑',
    'Longitudinal_Slope': '縱向坡度', 'Cross_Slope': '橫向坡度',
    'speed': '速限', 'Has_Ramp': '是否鄰近匝道',
    'Total_Lane_Width': '總車道寬', 'Total_Shoulder_Width': '總路肩寬',
    'Lane_Density': '車道密度', 'Shoulder_Ratio': '路肩比例',
    'Effective_Ratio': '有效通行比', 'Curvature_Severity': '反曲率半徑',
    'Pavement_Type_flexible': '鋪面_柔性', 'Pavement_Type_rigid': '鋪面_剛性',
    'Channelized_Area_no': '導引島_無', 'Channelized_Area_yes': '導引島_有',
    'Inner_Shoulder_no': '內路肩_無', 'Inner_Shoulder_yes': '內路肩_有',
    'Outer_Shoulder_no': '外路肩_無', 'Outer_Shoulder_yes': '外路肩_有',
    'Aux_Lane1_acceleration_lane': '輔助車道1_加速', 'Aux_Lane1_auxiliary_lane': '輔助車道1_輔助',
    'Aux_Lane1_deceleration_lane': '輔助車道1_減速', 'Aux_Lane1_no': '輔助車道1_無',
    'Aux_Lane2_acceleration_lane': '輔助車道2_加速', 'Aux_Lane2_auxiliary_lane': '輔助車道2_輔助',
    'Aux_Lane2_deceleration_lane': '輔助車道2_減速', 'Aux_Lane2_no': '輔助車道2_無',
    'Aux_Lane3_acceleration_lane': '輔助車道3_加速', 'Aux_Lane3_auxiliary_lane': '輔助車道3_輔助',
    'Aux_Lane3_deceleration_lane': '輔助車道3_減速', 'Aux_Lane3_no': '輔助車道3_無',
    'Pull_Over_Area_no': '臨停區_無', 'Pull_Over_Area_yes': '臨停區_有'
}

# ===== 讀資料 =====
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=EXCLUDE, errors='ignore').select_dtypes(include=[np.number]).copy()
y = df[TARGET].copy()
if X.shape[1] == 0:
    raise ValueError("沒有可用的數值特徵。請確認資料與 EXCLUDE 設定。")

# ===== 評估函式 =====
def metrics_on_test(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# ===== 主流程 =====
out_dir = Path("rf_5blocks_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

metrics_rows = []
feat_imps = []

for bi, seed in enumerate(RANDOM_SEEDS[:N_BLOCKS], start=1):
    # 第一次切 60%/40%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed
    )
    # 40% 再對半 -> 20% 驗證 / 20% 測試（只用測試）
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )

    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 測試集表現
    y_test_pred = rf.predict(X_test)
    m = metrics_on_test(y_test, y_test_pred)
    metrics_rows.append({
        "Block": bi,
        "MAE":  round(m["MAE"],  DECIMALS),
        "MSE":  round(m["MSE"],  DECIMALS),
        "RMSE": round(m["RMSE"], DECIMALS),
        "R2":   round(m["R2"],   DECIMALS),
    })

    # 特徵重要性
    fnames = list(rf.feature_names_in_) if hasattr(rf, "feature_names_in_") else X.columns.tolist()
    fi = pd.Series(rf.feature_importances_, index=fnames, name=f"block{bi}")
    feat_imps.append(fi)

# ===== 指標 CSV（平均 + 標準誤）=====
metrics_df = pd.DataFrame(metrics_rows).sort_values("Block").reset_index(drop=True)
n = len(metrics_df)

mean_row = {
    "Block": "Average",
    "MAE":  round(metrics_df["MAE"].mean(),  DECIMALS),
    "MSE":  round(metrics_df["MSE"].mean(),  DECIMALS),
    "RMSE": round(metrics_df["RMSE"].mean(), DECIMALS),
    "R2":   round(metrics_df["R2"].mean(),   DECIMALS),
}
se_row = {
    "Block": "SE",
    "MAE":  round(metrics_df["MAE"].std(ddof=1) / np.sqrt(n),  DECIMALS),
    "MSE":  round(metrics_df["MSE"].std(ddof=1) / np.sqrt(n),  DECIMALS),
    "RMSE": round(metrics_df["RMSE"].std(ddof=1) / np.sqrt(n), DECIMALS),
    "R2":   round(metrics_df["R2"].std(ddof=1) / np.sqrt(n),   DECIMALS),
}

metrics_out = pd.concat([metrics_df, pd.DataFrame([mean_row, se_row])], ignore_index=True)
metrics_csv_path = out_dir / "metrics_5blocks.csv"
metrics_out.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
print(f"已輸出: {metrics_csv_path}")

# ===== 特徵重要性平均（含中文名稱）=====
feat_imp_df = pd.concat(feat_imps, axis=1)  # index=英文特徵名, col=block
feat_imp_df["avg_importance"] = feat_imp_df.mean(axis=1)
feat_imp_df = feat_imp_df.sort_values("avg_importance", ascending=False)
feat_imp_df.insert(0, "中文名稱", [NAME_MAP.get(en, en) for en in feat_imp_df.index])

avg_imp_csv_path = out_dir / "avg_feature_importance.csv"
feat_imp_df.to_csv(avg_imp_csv_path, encoding="utf-8-sig")
print(f"已輸出: {avg_imp_csv_path}")

# ===== 圖（前 TOPK，中文顯示）=====
if MAKE_PLOT and TOPK > 0:
    import platform, matplotlib
    if platform.system() != "Windows":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topk_df = feat_imp_df[["中文名稱", "avg_importance"]].head(TOPK).iloc[::-1]
    plt.figure(figsize=(6, 3.8))
    plt.barh(topk_df["中文名稱"], topk_df["avg_importance"], height=0.5)
    plt.xlabel("平均重要性分數", fontsize=12)
    plt.title(f"前 {TOPK} 名特徵（5 blocks 平均）", fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plot_path = out_dir / "avg_importance_topk.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"已輸出: {plot_path}")

