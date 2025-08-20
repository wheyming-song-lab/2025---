# -*- coding: utf-8 -*-
import os
import datetime
import platform
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# === Font settings ===
if platform.system() != "Windows":
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans", "sans-serif"]
else:
    plt.rcParams["font.family"] = "Microsoft JhengHei"

# === Configuration ===
BASE_DIR = r"D:\eTag_no3"  # 修改為你的資料夾路徑
EXPORT_DIR = "plot/"
YEAR = 2024

# 8 個門架路段（原 6 + 新增 2）
SEGMENTS = {
    "北向": [
        "03F0746N-03F0698N",  # 關西服務區 → 高原
        "03F0698N-03F0648N",  # 高原 → 龍潭
        "03F0648N-03F0559N",  # 龍潭 → 大溪
        "03F0559N-03F0525N",  # 大溪 → 鶯歌系統
    ],
    "南向": [
        "03F0525S-03F0559S",  # 三鶯 → 鶯歌系統
        "03F0559S-03F0648S",  # 鶯歌系統 → 大溪
        "03F0648S-03F0698S",  # 大溪 → 龍潭
        "03F0698S-03F0746S",  # 龍潭 → 高原
    ]
}

PAIR_LABELS = {
    "03F0559S-03F0648S": "鶯歌系統 → 大溪",
    "03F0648S-03F0698S": "大溪 → 龍潭",
    "03F0698S-03F0746S": "龍潭 → 高原",
    "03F0525S-03F0559S": "三鶯 → 鶯歌系統",
    "03F0746N-03F0698N": "關西服務區 → 高原",
    "03F0698N-03F0648N": "高原 → 龍潭",
    "03F0648N-03F0559N": "龍潭 → 大溪",
    "03F0559N-03F0525N": "大溪 → 鶯歌系統",
}

# 僅保留小型車（31：小客、32：小貨）
VEHICLE_TYPES = {"31", "32"}

# 2024 國定假日（用來排除）
TAIWAN_HOLIDAYS_2024 = {
    (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14),
}

# eTag XML 命名空間
NAMESPACE = {"ns": "http://traffic.transportdata.tw/standard/traffic/schema/"}

def is_holiday(date):
    return (date.month, date.day) in TAIWAN_HOLIDAYS_2024

def get_excluded_dates(pair_id):
    """依方向過濾『事故日/異常日』；可按你的清單再增修"""
    if pair_id.endswith("N"):
        return {
            datetime.date(YEAR, 1, 21), datetime.date(YEAR, 3, 31),
            datetime.date(YEAR, 9, 22), datetime.date(YEAR, 10, 27),
            datetime.date(YEAR, 12, 1), datetime.date(YEAR, 12, 8),
            datetime.date(YEAR, 12, 22), datetime.date(YEAR, 12, 29),
        }
    else:
        return {
            datetime.date(YEAR, 2, 17), datetime.date(YEAR, 4, 7),
            datetime.date(YEAR, 4, 20), datetime.date(YEAR, 5, 11),
        }

def _get_vehicle_count(flow):
    """只從 ns:VehicleCount 讀取車流量；缺值/<=0 時回傳 1"""
    txt = flow.findtext("ns:VehicleCount", default="", namespaces=NAMESPACE)
    try:
        v = float(txt)
        return v if v > 0 else 1.0
    except:
        return 1.0

def parse_xml_file(file_path):
    """讀單一 XML，回傳符合門架與車種的紀錄（排除 0 或負速；若任一 31/32 為 0，整時段略過）"""
    records = []
    if not os.path.exists(file_path):
        return records

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for live in root.findall("ns:ETagPairLives/ns:ETagPairLive", NAMESPACE):
            pair_id = live.findtext("ns:ETagPairID", default="", namespaces=NAMESPACE)
            start_time_str = live.findtext("ns:StartTime", default="", namespaces=NAMESPACE)
            try:
                actual_time = datetime.datetime.fromisoformat(start_time_str)
            except:
                continue

            all_pairs = SEGMENTS["北向"] + SEGMENTS["南向"]
            if pair_id not in all_pairs:
                continue
            if actual_time.date() in get_excluded_dates(pair_id):
                continue

            # 若任何 31/32 的速度為 0，整筆該時段的 pair 直接略過（沿用原本策略）
            has_zero_speed = False
            for flow in live.findall("ns:Flows/ns:Flow", NAMESPACE):
                vtype = flow.findtext("ns:VehicleType", default="", namespaces=NAMESPACE)
                if vtype in VEHICLE_TYPES:
                    speed_text = flow.findtext("ns:SpaceMeanSpeed", default="", namespaces=NAMESPACE)
                    try:
                        if float(speed_text) == 0:
                            has_zero_speed = True
                            break
                    except:
                        continue
            if has_zero_speed:
                continue

            # 收集每個車種的 (速度, 車流量)
            for flow in live.findall("ns:Flows/ns:Flow", NAMESPACE):
                vtype = flow.findtext("ns:VehicleType", default="", namespaces=NAMESPACE)
                if vtype in VEHICLE_TYPES:
                    speed_text = flow.findtext("ns:SpaceMeanSpeed", default="", namespaces=NAMESPACE)
                    try:
                        speed = float(speed_text)
                        if speed > 0:  # 忽略 <= 0 的值
                            cnt = _get_vehicle_count(flow)  # 只用 VehicleCount
                            records.append({
                                'date': actual_time,       # 時間戳
                                'pair_id': pair_id,        # 門架對
                                'vehicle_type': vtype,     # 車種
                                'speed': speed,            # 該車種平均速
                                'count': cnt               # 作為加權權重
                            })
                    except:
                        continue
    except ET.ParseError:
        print(f"Parse error in file: {file_path}")

    return records

def collect_all_records():
    """掃 2024 年所有週末（排除國定假日）之每 5 分鐘 XML，彙整成列表"""
    record_map = {}
    start_date = datetime.date(YEAR, 1, 1)
    end_date = datetime.date(YEAR + 1, 1, 1)

    current_date = start_date
    while current_date < end_date:
        # 非國定假日且是週末（六/日）
        if not is_holiday(current_date) and current_date.weekday() >= 5:
            for hour in range(24):
                for minute in range(0, 60, 5):
                    dt = datetime.datetime(current_date.year, current_date.month, current_date.day, hour, minute)
                    fname = f"ETagPairLive_{dt.strftime('%H%M')}.xml"
                    fpath = os.path.join(BASE_DIR, dt.strftime("%Y/%m/%d"), fname)
                    records = parse_xml_file(fpath)
                    # 以 (時間戳, pair_id, 車種) 去重（保留最後一次）
                    for rec in records:
                        key = (rec['date'], rec['pair_id'], rec['vehicle_type'])
                        record_map[key] = rec
        current_date += datetime.timedelta(days=1)
    return list(record_map.values())

def _weighted_mean(x_speed, x_count):
    """車種加權平均（以 VehicleCount 為權重）"""
    w = np.asarray(x_count, dtype=float)
    v = np.asarray(x_speed, dtype=float)
    s = w.sum()
    if s <= 0:
        return np.nan
    return np.nansum(v * w) / s

def plot_combined(records):
    """繪製 8 格子（2x4）：每個門架一格，週六/週日曲線 + 平均±3SE 陰影 + 紅線(y=60)"""
    if not records:
        print("找不到任何資料，請檢查路徑與 XML 結構是否正確。")
        return

    df = pd.DataFrame(records)
    # 基本欄位
    df['date_only'] = df['date'].dt.date
    df['day'] = df['date'].dt.strftime('%A')       # 'Saturday' / 'Sunday'
    df['time_minutes'] = df['date'].dt.hour * 60 + df['date'].dt.minute

    # 車種加權平均：同一「日×時段×門架」先合成單一代表速度
    daily_wavg = (
        df.groupby(['pair_id', 'date_only', 'time_minutes'], as_index=False)
          .apply(lambda g: pd.Series({
              'wavg_speed': _weighted_mean(g['speed'], g['count']),
              'total_count': g['count'].sum()
          }))
    )

    # 用原 df 補 day（同日同時段 day 一致）
    df_key = df.drop_duplicates(subset=['date_only', 'time_minutes'])[['date_only', 'time_minutes', 'day']]
    daily_wavg = daily_wavg.merge(df_key, on=['date_only', 'time_minutes'], how='left')

    # 對「門架×星期幾×時段」跨多個週末再取平均、標準差與標準誤
    grouped = (
        daily_wavg.groupby(['pair_id', 'day', 'time_minutes'])
                  .agg(mean=('wavg_speed', 'mean'),
                       std=('wavg_speed', 'std'),
                       n=('wavg_speed', 'count'))
                  .reset_index()
    )
    grouped['se'] = grouped['std'] / np.sqrt(grouped['n'])

    # 繪圖
    fig_rows, fig_cols = 2, 4
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(24, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    all_segments = SEGMENTS['北向'] + SEGMENTS['南向']
    for i, pair_id in enumerate(all_segments):
        ax = axes[i]
        seg_data = grouped[grouped['pair_id'] == pair_id]
        if seg_data.empty:
            ax.set_title(f"{PAIR_LABELS.get(pair_id, pair_id)} 無資料")
            ax.grid(True)
            continue

        for day_name, label in [('Saturday', '週六'), ('Sunday', '週日')]:
            day_data = seg_data[seg_data['day'] == day_name].sort_values('time_minutes')
            if not day_data.empty:
                x = day_data['time_minutes'] / 60.0
                y = day_data['mean']
                se = day_data['se'].fillna(0)
                ax.plot(x, y, label=label)
                ax.fill_between(x, y - 3 * se, y + 3 * se, alpha=0.2)

        # === 加上紅線標記車速60 ===
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1, label='速度60')

        ax.set_title(f"{PAIR_LABELS.get(pair_id, pair_id)} ({pair_id})")
        ax.set_xticks(range(0, 25))
        ax.set_xticklabels([f"{h:02d}" for h in range(25)], rotation=45)
        ax.set_yticks(np.arange(0, 121, 10))
        ax.grid(True)
        ax.legend(loc='lower left')

    fig.suptitle(
        f"國道三號 鶯歌系統-高原路段 南北向門架 {YEAR}週末車速([每五分鐘平均速率]的平均值)比較\n排除國定假日與事故日",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(EXPORT_DIR, exist_ok=True)
    out_path = os.path.join(EXPORT_DIR, f"CombinedWeekendSpeed_{YEAR}_weighted.png")
    plt.savefig(out_path, dpi=300)
    print(f"圖片儲存於：{out_path}")
    plt.show()

# === Run ===
if __name__ == "__main__":
    print("Collecting speed (weighted by VehicleCount)...")
    all_records = collect_all_records()
    plot_combined(all_records)
