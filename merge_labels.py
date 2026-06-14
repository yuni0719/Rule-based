# =============================================================================
# merge_labels.py
# 西洋劍動作標籤合併工具
#
# 功能：遍歷 temp_results/ 資料夾內所有小 JSON 檔案，
#       將它們合併成一個標準大 JSON 陣列，儲存為 final_labels.json。
#
# 使用時機：待所有影片標註完畢後手動執行此腳本。
# 僅使用 Python 內建庫：os, json, glob
# =============================================================================

import os
import json
import glob

# =============================================================================
# 路徑設定
# =============================================================================

# 暫存小 JSON 的來源資料夾（與本檔案同層）
INPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "temp_results"
)

# 最終合併輸出的大 JSON 檔案路徑（與本檔案同層）
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "final_labels.json"
)

# =============================================================================
# 欄位驗證設定（確保合併時的資料完整性）
# =============================================================================

# final_labels.json 每筆記錄必須包含的欄位
REQUIRED_FIELDS = ["video_name", "person_id", "label_upper", "label_lower"]

# =============================================================================
# 合併函式
# =============================================================================

def load_single_label(fpath: str):
    """
    讀取單一小 JSON 檔案並做基本驗證。

    Args:
        fpath: 小 JSON 檔案的完整路徑

    Returns:
        dict：驗證通過的資料物件，或 None（驗證失敗時）
    """
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [錯誤] 無法讀取 {os.path.basename(fpath)}：{e}")
        return None

    # 確認必要欄位均存在
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        print(f"  [警告] {os.path.basename(fpath)} 缺少欄位：{missing}，略過此筆")
        return None

    # 確認欄位型別正確（提供友善錯誤訊息）
    if not isinstance(data["video_name"], str):
        print(f"  [警告] {os.path.basename(fpath)} 的 video_name 應為字串，略過此筆")
        return None

    for int_field in ["person_id", "label_upper", "label_lower"]:
        if not isinstance(data[int_field], int):
            try:
                data[int_field] = int(data[int_field])
            except (ValueError, TypeError):
                print(f"  [警告] {os.path.basename(fpath)} 的 {int_field} "
                      f"無法轉換為整數，略過此筆")
                return None

    # 只保留標準欄位（丟棄標註過程中產生的額外欄位）
    clean_record = {field: data[field] for field in REQUIRED_FIELDS}

    return clean_record


def merge_all_labels():
    """
    主合併函式：
    1. 掃描 INPUT_DIR 下所有 *.json 檔案。
    2. 讀取並驗證每個小 JSON。
    3. 按照 (video_name, person_id) 排序（便於閱讀）。
    4. 輸出為格式化的 final_labels.json。
    """
    print("=" * 60)
    print("  西洋劍動作標籤合併工具 — 啟動")
    print(f"  來源資料夾：{INPUT_DIR}")
    print(f"  輸出檔案  ：{OUTPUT_FILE}")
    print("=" * 60)

    # ── 確認來源資料夾是否存在 ───────────────────────────────────────────
    if not os.path.isdir(INPUT_DIR):
        print(f"\n[錯誤] 找不到 temp_results 資料夾：{INPUT_DIR}")
        print("請先執行 auto_labeler.py 產生標註結果。")
        return

    # ── 蒐集所有小 JSON 檔案（依檔名排序確保可重現性）───────────────────
    pattern    = os.path.join(INPUT_DIR, "*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"\n[警告] temp_results/ 資料夾內找不到任何 .json 檔案。")
        print("請確認 auto_labeler.py 已正確執行並產生標註結果。")
        return

    print(f"\n  找到 {len(json_files)} 個小 JSON 檔案，開始讀取與驗證...")

    # ── 逐一讀取並驗證 ────────────────────────────────────────────────────
    all_records  = []
    success_count = 0
    skip_count    = 0

    for fpath in json_files:
        fname  = os.path.basename(fpath)
        record = load_single_label(fpath)

        if record is not None:
            all_records.append(record)
            success_count += 1
            print(f"  [OK] {fname:40s} -> video={record['video_name']}, "
                  f"person={record['person_id']}, "
                  f"upper={record['label_upper']}, "
                  f"lower={record['label_lower']}")
        else:
            skip_count += 1

    print(f"\n  讀取完成：成功 {success_count} 筆，略過 {skip_count} 筆")

    if not all_records:
        print("\n[錯誤] 沒有任何有效記錄可合併，final_labels.json 不會被建立。")
        return

    # ── 排序：先依 video_name，再依 person_id ────────────────────────────
    all_records.sort(key=lambda r: (r["video_name"], r["person_id"]))

    # ── 偵測重複記錄（相同 video_name + person_id）────────────────────────
    seen_keys = {}
    dedup_records = []
    dup_count = 0

    for record in all_records:
        key = (record["video_name"], record["person_id"])
        if key in seen_keys:
            dup_count += 1
            print(f"  [警告] 重複記錄：video={key[0]}, person={key[1]}，"
                  f"保留最後出現的版本")
            # 移除舊版
            dedup_records = [r for r in dedup_records if
                             (r["video_name"], r["person_id"]) != key]
        seen_keys[key] = True
        dedup_records.append(record)

    if dup_count:
        print(f"\n  偵測到 {dup_count} 筆重複記錄，已自動去重")

    # ── 寫出 final_labels.json ────────────────────────────────────────────
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(dedup_records, f, ensure_ascii=False, indent=4)
    except OSError as e:
        print(f"\n[錯誤] 無法寫入 {OUTPUT_FILE}：{e}")
        return

    # ── 成功摘要 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  合併完成！共 {len(dedup_records)} 筆有效標籤記錄")
    print(f"  輸出檔案：{OUTPUT_FILE}")
    print("=" * 60)

    # ── 預覽輸出結構 ──────────────────────────────────────────────────────
    preview_count = min(3, len(dedup_records))
    if preview_count > 0:
        print("\n  ── 輸出預覽（前 3 筆）──────────────────────────────────")
        for record in dedup_records[:preview_count]:
            print(f"  {json.dumps(record, ensure_ascii=False)}")
        if len(dedup_records) > preview_count:
            print(f"  ... 以及另外 {len(dedup_records) - preview_count} 筆")

    # ── 統計摘要 ──────────────────────────────────────────────────────────
    print("\n  ── 標籤分佈統計 ────────────────────────────────────────")

    upper_counts = {}
    lower_counts = {}
    for r in dedup_records:
        u = r["label_upper"]
        l = r["label_lower"]
        upper_counts[u] = upper_counts.get(u, 0) + 1
        lower_counts[l] = lower_counts.get(l, 0) + 1

    upper_names = {0: "無", 1: "直刺"}
    lower_names = {
        0: "無（靜止）",
        1: "前進",
        2: "後退",
        3: "長刺",
        4: "飛刺",
        5: "前進長刺"
    }

    print("  上半身標籤分佈：")
    for label in sorted(upper_counts):
        name  = upper_names.get(label, f"未知({label})")
        count = upper_counts[label]
        bar   = "█" * count
        print(f"    [{label}] {name:8s}：{count:4d} 筆  {bar}")

    print("  下半身標籤分佈：")
    for label in sorted(lower_counts):
        name  = lower_names.get(label, f"未知({label})")
        count = lower_counts[label]
        bar   = "█" * count
        print(f"    [{label}] {name:10s}：{count:4d} 筆  {bar}")

    print("=" * 60)


# =============================================================================
# 主程式入口
# =============================================================================

if __name__ == "__main__":
    merge_all_labels()
