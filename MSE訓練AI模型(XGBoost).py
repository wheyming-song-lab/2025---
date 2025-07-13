# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# === 字體設定（Windows 請使用微軟正黑體） ===
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# === Step 1: 讀取資料 ===
df = pd.read_csv("2024_saturday_with_avg_future_speed.csv", encoding="big5")

# === Step 2: 特徵與目標欄位 ===
features = ["平均車速(km/h)", "過去1小時平均車速"]
target = "未來20分鐘平均車速"
X = df[features]
y = df[target]

# === Step 3: 切分資料 ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === Step 4: 建立與訓練模型 ===
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# === Step 5: 預測與評估 ===
y_pred = model.predict(X_test)
residuals = y_pred - y_test

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 壅塞樣本分析（未來20分鐘車速 < 60 km/h）
mask_cong = y_test < 60
mse_cong = mean_squared_error(y_test[mask_cong], y_pred[mask_cong])
mae_cong = mean_absolute_error(y_test[mask_cong], y_pred[mask_cong])

# === Step 6: 預測 vs 實際（前100筆） ===
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="實際車速", marker="o")
plt.plot(y_pred[:100], label="XGBoost 預測", linestyle="--")
plt.axhline(y=60, color='red', linestyle='--', label='壅塞門檻 60 km/h')

plt.text(105, max(max(y_test[:100]), max(y_pred[:100])) * 0.95,
         f"=== 整體測試集評估 ===\nMSE: {mse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}\n\n"
         f"=== 壅塞樣本（< 60 km/h）===\nMSE: {mse_cong:.2f}\nMAE: {mae_cong:.2f}",
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.title("XGBoost 預測未來20分鐘車速（前100筆）")
plt.xlabel("樣本索引")
plt.ylabel("車速 (km/h)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === Step 7: 殘差圖 ===
plt.figure(figsize=(10, 5))
plt.scatter(y_test, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', label="零誤差線")
plt.xlabel("實際車速 (km/h)")
plt.ylabel("預測誤差（預測 - 實際）")
plt.title("XGBoost 殘差圖")
plt.grid(True)
plt.legend()

# ✅ 負號顯示修正
def fix_minus(x, _):
    return str(int(x)).replace("-", "-")

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fix_minus))
plt.tight_layout()
plt.show()

# === Step 8: 實際 vs 預測 散佈圖 ===
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("實際 vs 預測 車速")
plt.xlabel("實際車速 (km/h)")
plt.ylabel("預測車速 (km/h)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 9: 殘差分布直方圖 ===
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=30, color='gray', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--')
plt.title("殘差分布（預測 - 實際）")
plt.xlabel("殘差值")
plt.ylabel("頻數")

# ✅ 修正負號（x 軸）
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(fix_minus))
plt.tight_layout()
plt.show()
