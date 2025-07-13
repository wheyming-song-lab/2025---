import os
import datetime
import platform
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --- Font settings for plot ---
if platform.system() != "Windows":
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans", "sans-serif"]
else:
    plt.rcParams["font.family"] = "Microsoft JhengHei"

# === Configuration ===
BASE_DIR = r"D:\eTag_no3"
EXPORT_DIR = "export/plot/"
YEAR = 2023

TARGET_PAIR_IDS = {
    "03F0698N-03F0648N",
    "03F0648N-03F0559N",
}
VEHICLE_TYPES = {"31", "32"}

TAIWAN_HOLIDAYS_2023 = {
    (1, 1), (1, 2), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25),
    (2, 27), (2, 28),
    (4, 3), (4, 4), (4, 5),
    (6, 22), (6, 23),
    (9, 29), (9, 30),
    (10, 9), (10, 10)
}

NAMESPACE = {"ns": "http://traffic.transportdata.tw/standard/traffic/schema/"}

def is_holiday(date):
    return (date.month, date.day) in TAIWAN_HOLIDAYS_2023

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
                        speed_text = flow.findtext("ns:SpaceMeanSpeed", default="", namespaces=NAMESPACE)
                        try:
                            speed = float(speed_text)
                            if speed > 0:
                                records.append({
                                    'date': target_date,
                                    'speed': speed,
                                    'pair_id': pair_id,
                                    'vehicle_type': vtype
                                })
                        except (ValueError, TypeError):
                            continue
    except ET.ParseError:
        print(f"Warning: Failed to parse {file_path}")
    return records

def collect_weekday_data():
    all_records = []
    start_date = datetime.date(YEAR, 1, 1)
    end_date = datetime.date(YEAR + 1, 1, 1)

    current_date = start_date
    while current_date < end_date:
        # 星期一～五且不是國定假日
        if not is_holiday(current_date) and current_date.weekday() < 5:
            for hour in range(24):
                for minute in range(0, 60, 5):
                    dt = datetime.datetime(current_date.year, current_date.month, current_date.day, hour, minute)
                    fname = f"ETagPairLive_{dt.strftime('%H%M')}.xml"
                    fpath = os.path.join(BASE_DIR, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"), fname)
                    records = parse_xml_file(fpath, dt)
                    all_records.extend(records)
        current_date += datetime.timedelta(days=1)
    return all_records

def calculate_and_plot(records):
    if not records:
        print("No records found.")
        return

    df = pd.DataFrame(records)
    df['day'] = df['date'].dt.strftime('%A')
    df['time_minutes'] = df['date'].dt.hour * 60 + df['date'].dt.minute

    grouped = df.groupby(['day', 'time_minutes'])['speed'].agg(['mean', 'std', 'count']).reset_index()

    # === Count number of weekday samples ===
    df['date_only'] = df['date'].dt.date
    weekday_n = df.groupby('day')['date_only'].nunique().to_dict()

    print("\nTotal samples by weekday (excluding holidays):")
    for day, n in weekday_n.items():
        print(f"{day}: {n} days")

    # === Plot ===
    plt.figure(figsize=(15, 8))
    weekday_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    for day_name in weekday_list:
        day_data = grouped[grouped['day'] == day_name]
        if not day_data.empty:
            x = day_data['time_minutes'] / 60
            y = day_data['mean']
            std = day_data['std']
            count = day_data['count']
            se = std / np.sqrt(count)
            label = f"{day_name} (n={weekday_n.get(day_name, 0)})"
            plt.plot(x, y, label=label, linewidth=2)
            plt.fill_between(x, y - 3 * se, y + 3 * se, alpha=0.2)

    plt.title("國道三號 鶯歌系統到高原北向 2023平日平均車速（排除國定假日）", fontsize=16)
    plt.xlabel("時間（小時）", fontsize=12)
    plt.ylabel("平均速率（公里/小時）", fontsize=12)
    plt.xticks(range(0, 25), [f"{h:02d}:00" for h in range(25)], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(EXPORT_DIR, exist_ok=True)
    save_path = os.path.join(EXPORT_DIR, f"WeekdaySpeed_{YEAR}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to: {save_path}")
    if platform.system() == "Windows":
        plt.show()
    else:
        plt.close()

# === Main ===
if __name__ == "__main__":
    print("Analyzing weekday (Mon–Fri) vehicle speed (excluding holidays)...")
    weekday_records = collect_weekday_data()
    calculate_and_plot(weekday_records)
