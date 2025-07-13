# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ====== 設定中文字型 (適用於 Windows 中文顯示) ======
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 如果用 Mac 可改成 'Heiti TC'

# Step 1: 讀取資料
df = pd.read_csv("2024_saturday_with_avg_future_speed.csv", encoding="big5")

# Step 2: 特徵與目標欄位
features = ["平均車速(km/h)", "過去1小時平均車速"]
target = "未來20分鐘平均車速"

X = df[features]
y = df[target]

# Step 3: 資料集切分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: 建立 Pipeline 模型
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipe_dt = Pipeline([
    ('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))
])

# Step 5: 模型訓練
pipe_lr.fit(X_train, y_train)
pipe_dt.fit(X_train, y_train)

# Step 6: 預測與評估
lr_preds = pipe_lr.predict(X_test)
dt_preds = pipe_dt.predict(X_test)

mse_lr = mean_squared_error(y_test, lr_preds)
mse_dt = mean_squared_error(y_test, dt_preds)

print(f"MSE - 線性回歸: {mse_lr:.2f}")
print(f"MSE - 決策樹: {mse_dt:.2f}")

# Step 7: 視覺化（中文標籤）
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="實際車速（未來20分鐘）", marker="o")
plt.plot(lr_preds[:100], label="線性回歸預測", linestyle="--")
plt.plot(dt_preds[:100], label="決策樹預測", linestyle=":")
plt.axhline(y=60, color='red', linestyle='-.', label='壅塞門檻：60 km/h')

plt.title("未來20分鐘車速預測比較（前100筆樣本）")
plt.ylabel("車速 (km/h)")
plt.xlabel("樣本編號")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
