import os
import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import triang
import platform
import matplotlib

# === 字型設定（確保圖片顯示中文）===
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Microsoft JhengHei"
else:
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans", "sans-serif"]

# === 設定區 ===
BASE_DIR = r"D:\eTag_no3"
EXPORT_DIR = "export/plot/"
os.makedirs(EXPORT_DIR, exist_ok=True)

TARGET_PAIR_IDS = {
    "03F0698N-03F0648N",
    "03F0648N-03F0559N",
}
VEHICLE_TYPES = {"31", "32"}
NAMESPACE = {"ns": "http://traffic.transportdata.tw/standard/traffic/schema/"}

HOLIDAYS_2023 = {
    (1, 1), (1, 2), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25),
    (2, 27), (2, 28), (4, 3), (4, 4), (4, 5),
    (6, 22), (6, 23), (9, 29), (9, 30), (10, 9), (10, 10)
}
def is_holiday(date):
    return (date.month, date.day) in HOLIDAYS_2023

# === 資料蒐集 ===
records = []
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2024, 1, 1)
current_date = start_date

while current_date < end_date:
    if current_date.weekday() == 5 and not is_holiday(current_date):  # 星期二且非假日
        folder = os.path.join(BASE_DIR, f"{current_date.year}", f"{current_date.month:02d}", f"{current_date.day:02d}")
        if not os.path.exists(folder):
            current_date += datetime.timedelta(days=1)
            continue

        for file in os.listdir(folder):
            if file.endswith(".xml"):
                try:
                    path = os.path.join(folder, file)
                    root = ET.parse(path).getroot()
                    for pair in root.findall(".//ns:ETagPairLive", NAMESPACE):
                        pid = pair.findtext("ns:ETagPairID", "", namespaces=NAMESPACE)
                        if pid not in TARGET_PAIR_IDS:
                            continue
                        stime = pair.findtext("ns:StartTime", "", namespaces=NAMESPACE).split("+")[0]
                        hour = datetime.datetime.fromisoformat(stime).hour
                        for flow in pair.findall(".//ns:Flow", NAMESPACE):
                            vtype = flow.findtext("ns:VehicleType", "", namespaces=NAMESPACE)
                            if vtype not in VEHICLE_TYPES:
                                continue
                            speed_text = flow.findtext("ns:SpaceMeanSpeed", "", namespaces=NAMESPACE)
                            if speed_text:
                                speed = float(speed_text)
                                if speed > 0:
                                    records.append((hour, speed))
                except Exception as e:
                    print(f"Error parsing {file}: {e}")
    current_date += datetime.timedelta(days=1)

print(f"Total records collected: {len(records)}")
if not records:
    print("No data found.")
    exit()

# === 分析與繪圖 ===
df = pd.DataFrame(records, columns=["hour", "speed"])
hourly_groups = df.groupby("hour")["speed"]

fig, axes = plt.subplots(6, 4, figsize=(28, 36))
axes = axes.flatten()

for hour in range(24):
    ax = axes[hour]
    speeds = hourly_groups.get_group(hour) if hour in hourly_groups.groups else []

    if len(speeds) == 0:
        ax.text(0.5, 0.5, f"{hour:02d}:00\n無資料", ha='center', va='center', fontsize=12)
        ax.axis("on")
        ax.set_xlim(0, 140)
        continue

    # 基本統計與三角分配參數
    min_speed = speeds.min()
    max_speed = speeds.max()
    mean_speed = speeds.mean()
    std_speed = speeds.std()
    count = len(speeds)

    # 找出眾數(最高頻的bin中心)
    hist_vals, bin_edges = np.histogram(speeds, bins=50)
    mode_index = np.argmax(hist_vals)
    mode_speed = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2

    # 三角分配參數
    a, b, c = min_speed, max_speed, mode_speed
    c_scaled = (c - a) / (b - a) if b > a else 0.5
    tri_dist = triang(c_scaled, loc=a, scale=b - a)
    x = np.linspace(a, b, 200)

    # 直方圖：頻數而非密度
    counts, bins, patches = ax.hist(speeds, bins=50, density=False, alpha=0.5, edgecolor='black')

    # bin寬 (理論上所有bin寬一樣)
    bin_width = bins[1] - bins[0]

    # 三角分配的理論頻數 = PDF * 樣本數 * bin寬
    tri_freq = tri_dist.pdf(x) * count * bin_width

    ax.plot(x, tri_freq, 'r--', label="三角分配擬合(頻數)")

    ax.set_title(f"{hour:02d}:00\nμ={mean_speed:.1f}, σ={std_speed:.1f}, N={count}", fontsize=10)
    ax.set_xlim(0, 140)
    ax.grid(True, alpha=0.3)

    # y軸顯示車輛數
    ax.set_ylabel("車輛數", fontsize=10)

    # x軸只有最下排顯示
    if hour // 4 == 5:
        ax.set_xlabel("速率（公里/小時）", fontsize=10)
    else:
        ax.set_xticklabels([])

    if hour == 0:
        ax.legend(fontsize=10)

plt.suptitle(
    "國道三號 鶯歌系統到高原北向 2023禮拜六車速(三角分配)（排除國定假日）",
    fontsize=18, y=0.995
)
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.5, wspace=0.3)

plot_path = os.path.join(EXPORT_DIR, "tuesday_yingge_gaoyuan_triangular_fit_freq.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"圖片已儲存：{plot_path}")
