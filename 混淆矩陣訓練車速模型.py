# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: 讀取 Big5 CSV
df = pd.read_csv("2024_saturday_features_labeled.csv", encoding="big5")

# Step 2: 指定特徵與標籤
features = ["平均車速(km/h)", "過去1小時平均車速", "時段"]
label = "未來20分鐘壅塞"

# Step 3: 刪除缺失值
df = df[features + [label]].dropna()

# Step 4: One-hot encoding for 時段
df = pd.get_dummies(df, columns=["時段"])

# Step 5: 分割 X, y
X = df.drop(columns=[label])
y = df[label]

# Step 6: 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Step 7: 標準化（給 Logistic Regression）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: 建立模型
log_model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
tree_model = DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=42)

# Step 9: 訓練
log_model.fit(X_train_scaled, y_train)
tree_model.fit(X_train, y_train)

# Step 10: 預測與報告
log_preds = log_model.predict(X_test_scaled)
tree_preds = tree_model.predict(X_test)

print("=== Logistic Regression ===")
print(classification_report(y_test, log_preds))

print("=== Decision Tree ===")
print(classification_report(y_test, tree_preds))

# Step 11: 混淆矩陣
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, log_preds), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, tree_preds), annot=True, fmt="d", cmap="Greens")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()
