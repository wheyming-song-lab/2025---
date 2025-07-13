import os
import datetime
import platform
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- 字體與中文處理 ---
if platform.system() != "Windows":
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans", "sans-serif"]
else:
    plt.rcParams["font.family"] = "Microsoft JhengHei"

# 修正終端輸出亂碼
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# === 基本設定 ===
BASE_DIR = r"D:\eTag_no3"
EXPORT_DIR = "export/plot/"
YEAR = 2024

# 春節日期與說明
HOLIDAY_LABELS = {
    "02/07": "除夕前二天",
    "02/08": "除夕前一天",
    "02/09": "除夕",
    "02/10": "大年初一",
    "02/11": "大年初二",
    "02/12": "大年初三",
    "02/13": "大年初四",
    "02/14": "大年初五",
}

# 國道3號特定路段
TARGET_PAIR_IDS = {
    "03F2100N-03F2078N",
    "03F2125N-03F2100N",
    "03F2153N-03F2125N",
}

# 車種
VEHICLE_TYPES = {"31", "32"}

# XML 命名空間
NAMESPACE = {"ns": "http://traffic.transportdata.tw/standard/traffic/schema/"}

# === XML 解析 ===
def parse_xml_file(file_path, target_date):
    records = []
    if not os.path.exists(file_path):
        return records
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for live in root.findall("ns:ETagPairLives/ns:ETagPairLive", NAMESPACE):
            pair_id = live.findtext("ns:ETagPairID", default="", namespaces=NAMESPACE)
            if pair_id in TARGET_PAIR_IDS:
                for flow in live.findall("ns:Flows/ns:Flow", NAMESPACE):
                    vtype = flow.findtext("ns:VehicleType", default="", namespaces=NAMESPACE)
                    if vtype in VEHICLE_TYPES:
                        try:
                            speed = float(flow.findtext("ns:SpaceMeanSpeed", default="0", namespaces=NAMESPACE))
                            if speed > 0:
                                records.append({
                                    'datetime': target_date,
                                    'speed': speed,
                                    'pair_id': pair_id,
                                    'vehicle_type': vtype
                                })
                        except:
                            continue
    except ET.ParseError:
        print(f"Warning: Failed to parse {file_path}")
    return records

# === 資料蒐集 ===
def collect_lunar_new_year_data():
    all_records = []
    for md_str in HOLIDAY_LABELS:
        mm, dd = md_str.split("/")
        folder = os.path.join(BASE_DIR, f"{YEAR}", mm, dd)
        if not os.path.exists(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        for fname in os.listdir(folder):
            if fname.startswith("ETagPairLive_") and fname.endswith(".xml"):
                time_part = fname[len("ETagPairLive_"):-4]
                try:
                    dt_str = f"{YEAR}-{mm}-{dd} {time_part[:2]}:{time_part[2:]}"
                    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
                    fpath = os.path.join(folder, fname)
                    all_records.extend(parse_xml_file(fpath, dt))
                except Exception as e:
                    print(f"Failed to parse time in {fname}: {e}")
    return all_records

# === 統計與繪圖 ===
def calculate_and_plot(records):
    if not records:
        print("No records found.")
        return

    df = pd.DataFrame(records)
    df['day'] = df['datetime'].dt.strftime("%m/%d")
    df['time_minutes'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute

    grouped = df.groupby(['day', 'time_minutes'])['speed'].agg(['mean', 'std', 'count']).reset_index()

    # === 終端印出（5 分鐘一筆） ===
    print("\n=== 5-Minute Speed Summary by Day ===")
    for day in sorted(df['day'].unique()):
        print(f"\n--- {day} ({HOLIDAY_LABELS.get(day, '')}) ---")
        day_rows = grouped[grouped['day'] == day]
        for _, row in day_rows.iterrows():
            hour = int(row['time_minutes'] // 60)
            minute = int(row['time_minutes'] % 60)
            mean_speed = row['mean']
            std_speed = row['std']
            count = int(row['count'])
            print(f"{hour:02d}:{minute:02d} - {mean_speed:.1f} ± {std_speed:.1f} km/h (n={count})")

    # === 繪圖（每 5 分鐘） ===
    plt.figure(figsize=(16, 8))
    for day in sorted(df['day'].unique()):
        day_rows = grouped[grouped['day'] == day]
        if not day_rows.empty:
            x = day_rows['time_minutes'] / 60
            y = day_rows['mean']
            label = f"{day} {HOLIDAY_LABELS.get(day, '')}"
            plt.plot(x, y, label=label, linewidth=1.5)

    plt.title("國道三號 草屯至烏日(北向) 春節每五分鐘平均車速", fontsize=16)
    plt.xlabel("時間（小時）", fontsize=12)
    plt.ylabel("平均速率（公里/小時）", fontsize=12)
    plt.xticks(range(0, 25), [f"{h:02d}:00" for h in range(25)], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(EXPORT_DIR, exist_ok=True)
    save_path = os.path.join(EXPORT_DIR, f"LunarNewYear_5minSpeed_{YEAR}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nChart saved to: {save_path}")
    if platform.system() == "Windows":
        plt.show()
    else:
        plt.close()

# === 主程式 ===
if __name__ == "__main__":
    print("Analyzing Lunar New Year vehicle speed (5-minute intervals)...")
    holiday_records = collect_lunar_new_year_data()
    calculate_and_plot(holiday_records)
