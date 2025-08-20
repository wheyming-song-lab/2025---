# -*- coding: utf-8 -*-
"""
XGB, LSTM, STGCN (GCN版) — 同一份由 speed/flow 現算特徵、同一套 10-block(6/2/2) 切分
"""

import os, random, math, numpy as np, pandas as pd

# ----------------- Global -----------------
CSV_PATH        = "正確版門架車速資料.csv"
EDGES_PATH      = "edges.csv"
CSV_ENCODING    = "big5"   # 若有編碼錯誤，可改 "big5-hkscs"
SEED            = 9999
TIME_STEPS      = 12       # 12*5min = 60min 歷史窗
EPOCHS          = 50
BATCH_SIZE_LSTM = 32
BATCH_SIZE_STG  = 64
LR_STG          = 1e-3
HIDDEN_CHANNELS = 64

# ----------------- Reproducibility -----------------
os.environ["PYTHONHASHSEED"]=str(SEED)
import tensorflow as tf
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

random.seed(SEED); np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Helpers -----------------
def find_col(df, keywords, required=True):
    for k in keywords:
        for c in df.columns:
            if k.lower() in str(c).lower():
                return c
    if required:
        raise KeyError(f"找不到欄位：{keywords}\n實際欄位：{list(df.columns)}")
    return None

def fmt_num(v: float) -> str:
    """四捨五入到第2位；若為 .00 則顯示到第3位。NaN -> 空字串"""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    x = float(v)
    s2 = f"{round(x, 2):.2f}"
    if s2.endswith("00"):
        return f"{x:.3f}"
    return s2

def add_avg_se(rows, order=("MAE","MSE","RMSE","R2")):
    """
    將每個 Block 的指標數值格式化，並在最後新增一列 '平均數 (標準誤)'。
    預設輸出欄位順序：MAE, MSE, RMSE, R2
    """
    dfm = pd.DataFrame(rows)
    for k in order:
        if k not in dfm.columns:
            dfm[k] = np.nan

    # 計算平均與標準誤
    def se(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0

    means = {k: float(np.nanmean(dfm[k].astype(float))) for k in order}
    ses   = {k: se(dfm[k].astype(float)) for k in order}

    # 格式化逐列
    out = dfm.copy()
    for k in order:
        out[k] = out[k].map(fmt_num)

    # 最後一列：平均數 (標準誤)
    last = {"Block": "平均數 (標準誤)"}
    for k in order:
        last[k] = f"{fmt_num(means[k])} ({fmt_num(ses[k])})"
    out = pd.concat([out, pd.DataFrame([last])], ignore_index=True)

    cols = ["Block"] + list(order)
    return out[cols]

# 線性回歸斜率 (每分鐘)，x 用實際分鐘刻度，y 用 km/h
def rolling_lr_slope_per_min(series, win_steps, step_min=5):
    # series: 等頻（5min）序列
    # 回傳與 series 對齊的斜率（單位 km/h/分鐘），前 (win_steps-1) 位置為 NaN（因不足窗）
    y = series.to_numpy(dtype=float)
    n = len(y)
    out = np.full(n, np.nan, float)
    # 時間向量，最後一步對齊當下點：[-(win-1), ..., -1, 0] * step_min
    t = np.arange(-(win_steps-1), 1) * step_min
    t = t - t.mean()
    denom = (t*t).sum()
    for i in range(win_steps-1, n):
        yy = y[i-win_steps+1:i+1]
        slope = ((t * (yy - yy.mean())).sum()) / denom
        out[i] = slope
    return pd.Series(out, index=series.index, dtype=float)

# ----------------- Load CSV -----------------
df = pd.read_csv(CSV_PATH, encoding=CSV_ENCODING, low_memory=False)
edges_df = pd.read_csv(EDGES_PATH)

# 必要欄位
col_pair   = find_col(df, ["門架編號"])
col_date   = find_col(df, ["日期"])
col_time   = find_col(df, ["時間"])
col_speed  = find_col(df, ["平均車速"])
col_flow   = find_col(df, ["車輛總數","車流量","車流","vehicle_count","traffic_count","flow"])
col_target = find_col(df, ["未來20分鐘平均車速","未來20分鐘車速","未來20分鐘"])

# 幾何欄位（預留；目前空）
geometry_candidates = ["總車道寬","曲線半徑","內路肩寬","外路肩寬","橫坡","縱坡","車道數","路面寬","超高","視距","平曲線半徑","縱坡度"]
geo_cols = [c for c in geometry_candidates if c in df.columns]  # 若要啟用靜態幾何，將 geometry_candidates 中存在於 df 的欄位加入此清單

# 時間處理
df["datetime"] = pd.to_datetime(df[col_date].astype(str) + " " + df[col_time].astype(str), errors="coerce")
df.sort_values([col_pair, "datetime"], inplace=True)
df["hour"] = df["datetime"].dt.hour
df["date"] = df["datetime"].dt.date

# 數值化
for c in [col_speed, col_flow] + geo_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 僅保留出現在 edges 的節點
nodes_in_edges = sorted(set(edges_df["src"]).union(set(edges_df["dst"])))
df = df[df[col_pair].isin(nodes_in_edges)].copy()

# ----------------- 由 speed/flow 現算動態特徵（對齊「當下列」） -----------------
def compute_dynamic_features(df, pair_col, speed_col, flow_col):
    dfs = []
    for g, gdf in df.groupby(pair_col, sort=False):
        gdf = gdf.copy()
        gdf["past_1h_mean_speed"] = gdf[speed_col].rolling(window=12, min_periods=12).mean()
        gdf["past_1h_mean_flow"]  = gdf[flow_col].rolling(window=12, min_periods=12).mean()
        gdf["slope_5min_per_min"] = (gdf[speed_col] - gdf[speed_col].shift(1)) / 5.0
        gdf["slope_30min_lr_per_min"] = rolling_lr_slope_per_min(gdf[speed_col], win_steps=6, step_min=5)
        dfs.append(gdf)
    out = pd.concat(dfs, axis=0).sort_values([pair_col, "datetime"])
    return out

df = compute_dynamic_features(df, col_pair, col_speed, col_flow)

# 特徵集合（動態）
features_dyn = [
    col_speed,                 # 平均車速
    col_flow,                  # 車輛總數 / 車流量
    "hour",                    # 小時
    "past_1h_mean_speed",      # 過去 1 小時平均車速
    "slope_5min_per_min",      # 5 分鐘差分斜率（每分鐘）
    "slope_30min_lr_per_min"   # 30 分鐘線回歸斜率（每分鐘）
]

# one-hot（XGB/LSTM 使用）
dummies = pd.get_dummies(df[col_pair], prefix="SEG", dtype=float)
df = pd.concat([df, dummies], axis=1)
features_xgb_lstm = features_dyn + list(dummies.columns) + geo_cols
target = col_target

# 共用基表
use_cols_common = ["datetime","date",col_pair] + features_xgb_lstm + [target]
df_base = df[use_cols_common].dropna().reset_index(drop=True)

# ----------------- 10 blocks，6/2/2（以「日」為單位，三模型共用） -----------------
all_dates = sorted(df_base["date"].unique())
n_total = len(all_dates)
n_train = int(n_total * 0.6)
n_valid = int(n_total * 0.2)

def build_blocks(dates, ntr, nva, seed):
    blocks=[]
    for k in range(1, 11):
        rng = np.random.RandomState(seed + k)
        d = list(dates); rng.shuffle(d)
        train=d[:ntr]; valid=d[ntr:ntr+nva]; test=d[ntr+nva:]
        blocks.append({"fold":k, "train":train, "valid":valid, "test":test})
    return blocks

blocks = build_blocks(all_dates, n_train, n_valid, SEED)
print(f"[INFO] blocks: total_days={n_total} -> train:{n_train}, valid:{n_valid}, test:{n_total-n_train-n_valid}")

# =========================================================
# XGB（吃同一套現算特徵；為一致性，也用 z-score）
# =========================================================
from copy import deepcopy
xgb_metrics=[]; imps=[]
xgb_params = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
                  subsample=1.0, colsample_bytree=1.0, random_state=SEED)

for b in blocks:
    tr=df_base[df_base["date"].isin(b["train"])].reset_index(drop=True)
    va=df_base[df_base["date"].isin(b["valid"])].reset_index(drop=True)
    te=df_base[df_base["date"].isin(b["test" ])].reset_index(drop=True)

    sx = StandardScaler(); sy = StandardScaler()
    Xtr = sx.fit_transform(tr[features_xgb_lstm]); ytr = sy.fit_transform(tr[[target]]).ravel()
    Xva = sx.transform(va[features_xgb_lstm]);     yva = sy.transform(va[[target]]).ravel()
    Xte = sx.transform(te[features_xgb_lstm]);     yte = sy.transform(te[[target]]).ravel()

    xgb = XGBRegressor(**deepcopy(xgb_params))
    xgb.fit(Xtr, ytr, eval_set=[(Xtr,ytr),(Xva,yva)], eval_metric="rmse", verbose=False)

    y_pred = sy.inverse_transform(xgb.predict(Xte).reshape(-1,1)).ravel()
    y_true = te[target].to_numpy(float).ravel()

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)  # 直接由 MSE 開根號
    r2   = r2_score(y_true, y_pred)

    xgb_metrics.append({
        "Block": f"Block {b['fold']}",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })
    imps.append(xgb.feature_importances_)

# 輸出 XGB 成果
xgb_out = add_avg_se(xgb_metrics, order=("MAE","MSE","RMSE","R2"))
xgb_out.to_csv("xgb_10block_metrics.csv", index=False, encoding="utf-8-sig")

imp_arr=np.vstack(imps); fn=list(features_xgb_lstm)
rows=[{"Block":f"Block {i}", "feature":f, "importance":float(v)} for i,imp in enumerate(imp_arr,1) for f,v in zip(fn,imp)]
pd.DataFrame(rows).sort_values(["Block","importance"], ascending=[True,False])\
  .to_csv("xgb_feature_importance_by_block.csv", index=False, encoding="utf-8-sig")
mean_imp=imp_arr.mean(axis=0); std_imp=imp_arr.std(axis=0,ddof=1) if imp_arr.shape[0]>1 else np.zeros_like(mean_imp)
pd.DataFrame({"feature":fn,"mean_importance":mean_imp,"std_importance":std_imp}).sort_values("mean_importance",ascending=False)\
  .assign(rank=lambda d: np.arange(1,len(d)+1)).to_csv("xgb_feature_importance_avg.csv", index=False, encoding="utf-8-sig")
order_idx=np.argsort(mean_imp)[::-1]
plt.figure(figsize=(10,5)); plt.bar(range(len(fn)),mean_imp[order_idx]); plt.xticks(range(len(fn)),[fn[i] for i in order_idx],rotation=35,ha="right")
plt.ylabel("Mean feature importance (XGBoost)"); plt.title("XGBoost Feature Importance (mean over 10 blocks)")
plt.tight_layout(); plt.savefig("xgb_feature_importance_avg.png", dpi=150)
print("[Done] XGB files")

# =========================================================
# LSTM（窗尾對齊 + z-score + EarlyStopping）
# =========================================================
def make_seq_lstm(df_sorted, feat_cols, tgt_col, L, group_col):
    Xs, ys = [], []
    for _, gdf in df_sorted.groupby(group_col, sort=False):
        X = gdf[feat_cols].to_numpy(float); y = gdf[[tgt_col]].to_numpy(float)
        if len(X) < L: continue
        # 從第 L 筆開始切，窗 i..i+L-1 → y[i+L-1]
        for i in range(L-1, len(X)):
            Xs.append(X[i-L+1:i+1])
            ys.append(y[i])
    return (np.array(Xs), np.array(ys)) if Xs else (np.empty((0,L,len(feat_cols))), np.empty((0,1)))

lstm_metrics=[]
for b in blocks:
    tr=df_base[df_base["date"].isin(b["train"])].sort_values([col_pair,"datetime"]).reset_index(drop=True)
    va=df_base[df_base["date"].isin(b["valid"])].sort_values([col_pair,"datetime"]).reset_index(drop=True)
    te=df_base[df_base["date"].isin(b["test" ])].sort_values([col_pair,"datetime"]).reset_index(drop=True)

    sx, sy = StandardScaler(), StandardScaler()
    Xtr = sx.fit_transform(tr[features_xgb_lstm]); ytr = sy.fit_transform(tr[[target]])
    Xva = sx.transform(va[features_xgb_lstm]);     yva = sy.transform(va[[target]])
    Xte = sx.transform(te[features_xgb_lstm]);     yte = sy.transform(te[[target]])

    tr_s, va_s, te_s = tr.copy(), va.copy(), te.copy()
    tr_s[features_xgb_lstm]=Xtr; tr_s[target]=ytr
    va_s[features_xgb_lstm]=Xva; va_s[target]=yva
    te_s[features_xgb_lstm]=Xte; te_s[target]=yte

    Xtr_seq,ytr_seq = make_seq_lstm(tr_s,features_xgb_lstm,target,TIME_STEPS,col_pair)
    Xva_seq,yva_seq = make_seq_lstm(va_s,features_xgb_lstm,target,TIME_STEPS,col_pair)
    Xte_seq,yte_seq = make_seq_lstm(te_s,features_xgb_lstm,target,TIME_STEPS,col_pair)
    if len(Xtr_seq)==0 or len(Xva_seq)==0 or len(Xte_seq)==0:
        print(f"[WARN] LSTM Block {b['fold']} 序列不足，略過。"); continue

    model = Sequential([LSTM(64, activation="tanh", input_shape=(TIME_STEPS, len(features_xgb_lstm))), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=EPOCHS, batch_size=BATCH_SIZE_LSTM, shuffle=False, callbacks=[es], verbose=0)

    y_pred = sy.inverse_transform(model.predict(Xte_seq, verbose=0))
    y_true = sy.inverse_transform(yte_seq)

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)  # 由 MSE 開根號
    r2   = r2_score(y_true, y_pred)

    lstm_metrics.append({
        "Block": f"Block {b['fold']}",
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2":  float(r2)
    })

pd.DataFrame(add_avg_se(lstm_metrics, order=("MAE","MSE","RMSE","R2"))) \
  .to_csv("lstm_10block_metrics.csv", index=False, encoding="utf-8-sig")
print("[Done] LSTM files")

# =========================================================
# STGCN（GCN 版、Â 正規化、z-score、靜態節點融合、窗尾對齊）
# =========================================================
# 建圖（雙向 + 自迴路），再做 Â = D^{-1/2}(A+I)D^{-1/2}
nodes = nodes_in_edges; N=len(nodes); idx={n:i for i,n in enumerate(nodes)}
A = np.zeros((N,N), float)
for _, r in edges_df.iterrows():
    i, j = idx.get(r["src"]), idx.get(r["dst"])
    if i is None or j is None: continue
    w = float(r.get("weight", 1.0))
    A[i,j] = max(A[i,j], w); A[j,i] = max(A[j,i], w)
A = A + np.eye(N)
D = np.diag(A.sum(axis=1))
with np.errstate(divide='ignore'):
    D_is = np.diag(1.0/np.sqrt(np.maximum(np.diag(D), 1e-12)))
A_hat = D_is @ A @ D_is
A_hat_t = torch.tensor(A_hat, dtype=torch.float32, device=device)

# 動態 panel（只用現算動態特徵）
feature_cols_panel_dyn = features_dyn
use_cols_panel = ["datetime","date",col_pair] + feature_cols_panel_dyn + [target] + geo_cols
df_panel = df[use_cols_panel].dropna().reset_index(drop=True)

def build_panel_dynamic(df_part, nodes, feature_cols, target_col, pair_col):
    # 僅保留「所有節點同時有值」的時間戳
    cnt = df_part.groupby("datetime")[pair_col].nunique()
    full_times = cnt[cnt == len(nodes)].index
    if len(full_times) == 0:
        return None, None
    d = df_part[df_part["datetime"].isin(full_times)].sort_values(["datetime", pair_col])
    # X: (T,N,F_dyn); y: (T,N)
    X_list=[]
    for f in feature_cols:
        pv = d.pivot_table(index="datetime", columns=pair_col, values=f).reindex(columns=nodes)
        X_list.append(pv.values[:, :, None])
    X_all = np.concatenate(X_list, axis=2)
    y_pv = d.pivot_table(index="datetime", columns=pair_col, values=target).reindex(columns=nodes)
    y_all = y_pv.values
    return X_all, y_all

# 靜態節點（幾何取中位數，z-score）
def build_node_static(df_all, nodes, pair_col, geo_cols):
    if not geo_cols: return None
    med = df_all.groupby(pair_col)[geo_cols].median().reindex(nodes)
    return med.to_numpy(dtype=np.float32)

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out, bias=False)
    def forward(self, x, A_hat):
        # x: (B, C_in, T, N)；先把特徵轉到節點
        B, C, T, N = x.shape
        x_perm = x.permute(0,2,3,1).reshape(B*T, N, C)        # (BT, N, C)
        x_lin  = self.lin(x_perm)                             # (BT, N, c_out)
        x_gcn  = torch.einsum("nm, bmc -> bnc", A_hat, x_lin) # Â X Θ
        x_out  = x_gcn.reshape(B, T, N, -1).permute(0,3,1,2)  # (B, c_out, T, N)
        return x_out

class TemporalConvGLU(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__(); pad=(k-1)
        self.conv = nn.Conv2d(c_in, c_out*2, kernel_size=(k,1), padding=(pad,0))
        self.res  = nn.Conv2d(c_in, c_out, kernel_size=1)
    def forward(self, x):
        y = self.conv(x); T = x.size(2); y = y[:, :, -T:, :]
        P, Q = torch.chunk(y, 2, dim=1)
        return self.res(x) + P * torch.sigmoid(Q)

class STGCN_GCN(nn.Module):
    def __init__(self, f_in, hidden, out=1, static_dim=0):
        super().__init__()
        self.t1 = TemporalConvGLU(f_in, hidden, 3)
        self.g1 = GraphConv(hidden, hidden)
        self.t2 = TemporalConvGLU(hidden, hidden, 3)
        self.use_static = static_dim > 0
        if self.use_static:
            self.proj = nn.Linear(static_dim, hidden)
            self.fuse = nn.Conv2d(hidden*2, hidden, 1)
        self.readout = nn.Conv2d(hidden, out, 1)
        self.drop = nn.Dropout(0.1)
    def forward(self, x, A_hat, S=None):   # x:(B,F,T,N), S:(N,D_static)
        x = self.t1(x)
        x = F.relu(self.g1(x, A_hat))
        x = self.drop(self.t2(x))
        if self.use_static and S is not None:
            B,C,T,N = x.shape
            s = torch.tanh(self.proj(S)).t().unsqueeze(0).unsqueeze(2).expand(B,-1,T,-1)
            x = self.fuse(torch.cat([x, s], dim=1))
        y = self.readout(x)[:, :, -1, :]   # 取窗尾
        return y

def make_sequences_stgcn(X_all, y_all, L):
    T = X_all.shape[0]
    if T < L: return None, None
    Xs, ys = [], []
    for i in range(L-1, T):
        X_win = X_all[i-L+1:i+1]                    # (L,N,F)
        y_tgt = y_all[i]                             # (N,)
        Xs.append(np.transpose(X_win, (2,0,1)))      # -> (F,L,N)
        ys.append(y_tgt)
    return np.stack(Xs,0), np.stack(ys,0)

stg_metrics=[]

# 準備靜態節點矩陣（每節點幾何中位數 → z-score）
S_static = build_node_static(df, nodes, col_pair, geo_cols)
if S_static is not None:
    s_scaler = StandardScaler()
    S_static = s_scaler.fit_transform(S_static)
    print(f"[INFO] Static features z-scored: mean~{S_static.mean():.3f}, std~{S_static.std():.3f}")
S_static_t = torch.tensor(S_static, dtype=torch.float32, device=device) if S_static is not None else None
static_dim = 0 if S_static is None else S_static.shape[1]

for b in blocks:
    tr_df = df_panel[df_panel["date"].isin(b["train"])]
    va_df = df_panel[df_panel["date"].isin(b["valid"])]
    te_df = df_panel[df_panel["date"].isin(b["test" ])]

    Xtr_all, ytr_all = build_panel_dynamic(tr_df, nodes, feature_cols_panel_dyn, target, col_pair)
    Xva_all, yva_all = build_panel_dynamic(va_df, nodes, feature_cols_panel_dyn, target, col_pair)
    Xte_all, yte_all = build_panel_dynamic(te_df, nodes, feature_cols_panel_dyn, target, col_pair)
    if Xtr_all is None or Xva_all is None or Xte_all is None:
        print(f"[WARN] STGCN Block {b['fold']} 缺完整對齊的時段，略過。"); continue

    # z-score（fit 僅用訓練集；在 T*N 維度上 per-feature）
    Ttr, Nn, Ff = Xtr_all.shape
    tr2d = Xtr_all.reshape(Ttr*Nn, Ff)
    sx = StandardScaler(); sy = StandardScaler()
    sx.fit(tr2d)
    Xtr_all = sx.transform(Xtr_all.reshape(Ttr*Nn, Ff)).reshape(Ttr,Nn,Ff)
    Xva_all = sx.transform(Xva_all.reshape(-1,Ff)).reshape(Xva_all.shape)
    Xte_all = sx.transform(Xte_all.reshape(-1,Ff)).reshape(Xte_all.shape)

    ytr_all = sy.fit_transform(ytr_all.reshape(-1,1)).reshape(ytr_all.shape)
    yva_all = sy.transform(yva_all.reshape(-1,1)).reshape(yva_all.shape)
    yte_all = sy.transform(yte_all.reshape(-1,1)).reshape(yte_all.shape)

    # 切窗（窗尾對齊）
    Xtr_seq,ytr_seq = make_sequences_stgcn(Xtr_all, ytr_all, TIME_STEPS)
    Xva_seq,yva_seq = make_sequences_stgcn(Xva_all, yva_all, TIME_STEPS)
    Xte_seq,yte_seq = make_sequences_stgcn(Xte_all, yte_all, TIME_STEPS)
    if Xtr_seq is None or Xva_seq is None or Xte_seq is None:
        print(f"[WARN] STGCN Block {b['fold']} 序列不足，略過。"); continue

    # tensors
    Xtr = torch.tensor(Xtr_seq, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr_seq, dtype=torch.float32, device=device).unsqueeze(1)
    Xva = torch.tensor(Xva_seq, dtype=torch.float32, device=device)
    yva = torch.tensor(yva_seq, dtype=torch.float32, device=device).unsqueeze(1)
    Xte = torch.tensor(Xte_seq, dtype=torch.float32, device=device)

    model = STGCN_GCN(f_in=Xtr.shape[1], hidden=HIDDEN_CHANNELS, out=1, static_dim=static_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_STG)
    crit  = nn.MSELoss()
    best=float("inf"); best_state=None

    for ep in range(1, EPOCHS+1):
        model.train(); opt.zero_grad()
        pred = model(Xtr, A_hat_t, S_static_t)
        loss = crit(pred, ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xva, A_hat_t, S_static_t), yva).item()
        if vloss < best:
            best = vloss
            best_state = {k: v.detach().clone() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(Xte, A_hat_t, S_static_t).squeeze(1).cpu().numpy()  # (B,N)

    # 還原
    y_pred = sy.inverse_transform(pred_scaled.reshape(-1,1)).reshape(pred_scaled.shape)
    y_true = sy.inverse_transform(yte_seq.reshape(-1,1)).reshape(yte_seq.shape)

    mse  = float(mean_squared_error(y_true.ravel(), y_pred.ravel()))
    mae  = float(mean_absolute_error(y_true.ravel(), y_pred.ravel()))
    rmse = float(math.sqrt(mse))  # 由 MSE 開根號
    r2   = float(r2_score(y_true.ravel(), y_pred.ravel()))
    stg_metrics.append({"Block":f"Block {b['fold']}", "MAE":mae, "MSE":mse, "RMSE":rmse, "R2":r2})
    print(f"[STGCN Block {b['fold']}] MSE={mse:.4f} MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

pd.DataFrame(add_avg_se(stg_metrics, order=("MAE","MSE","RMSE","R2"))) \
  .to_csv("stgcn_10block_metrics.csv", index=False, encoding="utf-8-sig")

print("\n=== 完成 ===")
print("輸出：xgb_10block_metrics.csv, lstm_10block_metrics.csv, stgcn_10block_metrics.csv")
print("特徵重要性：xgb_feature_importance_by_block.csv, xgb_feature_importance_avg.csv, xgb_feature_importance_avg.png")
