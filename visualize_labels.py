import os
import sys
import json
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 強制 Windows 終端機以 UTF-8 輸出中文，防止出現亂碼
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ── 常數定義 ──────────────────────────────────────────────────────────
DETECTION_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "yolo", "ultralytics", "output_videos", "2d_detections"
)
VIDEO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "yolo", "ultralytics", "test_videos"
)
OUTPUT_VIDEO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "labeled_videos"
)

# 動作標籤中文名稱定義
UPPER_LABELS = {
    0: "無",
    1: "直刺"
}

LOWER_LABELS = {
    0: "無",
    1: "前進",
    2: "後退",
    3: "長刺",
    4: "飛刺",
    5: "前進長刺"
}

# COCO 骨架連接關係 (17個關鍵點)
SKELETON_CONNECTIONS = [
    (5, 6),             # 雙肩
    (5, 7), (7, 9),     # 左手
    (6, 8), (8, 10),    # 右手
    (5, 11), (6, 12),   # 左右側身
    (11, 12),           # 髖部
    (11, 13), (13, 15), # 左腿
    (12, 14), (14, 16)  # 右腿
]

# 關鍵點索引定義
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6

def draw_text_chinese(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在 OpenCV 影像上繪製繁體中文字，防止亂碼。
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_paths = [
        "C:\\Windows\\Fonts\\msjhbd.ttc", # 微軟正黑體 Bold
        "C:\\Windows\\Fonts\\msjh.ttc",   # 微軟正黑體
        "C:\\Windows\\Fonts\\mingliu.ttc",# 細明體
        "arial.ttf"
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_keypoint_coords(keypoints: list, idx: int):
    """
    自適應提取關鍵點坐標 (x, y)，相容 dict 與 list 兩種格式。
    """
    if not keypoints or idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    
    # ── 格式一：dict ─────────────────────────────────────────────────
    if isinstance(kp, dict):
        x = kp.get("x", kp.get("X", None))
        y = kp.get("y", kp.get("Y", None))
        conf = kp.get("score", kp.get("confidence", kp.get("prob", kp.get("visibility", 1.0))))
        if x is not None and y is not None:
            try:
                x, y, conf = float(x), float(y), float(conf)
                if conf >= 0.1:  # 信心度小於 0.1 視為無效點
                    return (int(x), int(y))
            except:
                pass
                
    # ── 格式二：list / tuple ──────────────────────────────────────────
    elif isinstance(kp, (list, tuple)):
        if len(kp) >= 2:
            try:
                x, y = float(kp[0]), float(kp[1])
                conf = float(kp[2]) if len(kp) > 2 else 1.0
                if conf >= 0.1:
                    return (int(x), int(y))
            except:
                pass
    return None

def load_frame_json(json_path):
    """
    讀取 ViTPose 關節點 JSON，相容多種格式
    """
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if "keypoints" in data:
                return data["keypoints"]
            elif "instances" in data and data["instances"] and "keypoints" in data["instances"][0]:
                return data["instances"][0]["keypoints"]
            elif "persons" in data and data["persons"] and "keypoints" in data["persons"][0]:
                return data["persons"][0]["keypoints"]
    except Exception:
        pass
    return None

def find_video_file(video_name):
    """
    在 VIDEO_DIR 中尋找該影片檔
    """
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4"]
    for ext in extensions:
        path = os.path.join(VIDEO_DIR, video_name + ext)
        if os.path.exists(path):
            return path
    return None

def process_video(video_name, labels_dict):
    """
    讀取影片並標繪骨架、框與標籤資訊
    """
    video_path = find_video_file(video_name)
    if not video_path:
        print(f"  [跳過] 找不到原始影片：{video_name}，搜尋路徑為：{VIDEO_DIR}")
        return False
        
    print(f"\n🎥 開始標註視覺化影片：{video_name}")
    print(f"  -> 影片路徑: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [錯誤] 無法開啟影片：{video_path}")
        return False
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
        
    # 建立輸出目錄與寫入器
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_VIDEO_DIR, f"labeled_{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # 找尋此影片下偵測到的所有 person_id 資料夾
    video_det_dir = os.path.join(DETECTION_DIR, video_name)
    person_ids = []
    if os.path.exists(video_det_dir):
        person_ids = [int(d) for d in os.listdir(video_det_dir) if d.isdigit()]
    person_ids.sort()
    
    if not person_ids:
        print(f"  [跳過] 偵測目錄下找不到任何 person_id 關節資料：{video_det_dir}")
        cap.release()
        out.release()
        return False
        
    print(f"  -> 偵測到選手 ID 列表: {person_ids}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 繪製半透明資訊黑底板（加大加寬，並往下移一點避開頂端邊界，配合 32 號大粗體字）
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 70), (900, 75 + len(person_ids) * 60), (0, 0, 0), -1)
        alpha = 0.55
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 走訪每位選手，繪製骨架與抓取 Bounding Box
        for idx, pid in enumerate(person_ids):
            # 讀取對應的關節 JSON
            json_name = f"frame_{frame_idx:012d}.json"
            json_path = os.path.join(video_det_dir, str(pid), json_name)
            kps = load_frame_json(json_path)
            
            # 從 dictionary 裡取得該選手此影片的標籤
            label_info = labels_dict.get((video_name, pid))
            if label_info:
                up_val  = label_info.get("label_upper", 0)
                low_val = label_info.get("label_lower", 0)
            else:
                up_val, low_val = 0, 0
                
            up_name  = UPPER_LABELS.get(up_val, "無")
            low_name = LOWER_LABELS.get(low_val, "無")
            
            # 在資訊板上印出標籤資訊 (字體顯著增大到 32，間距拉大，粗體微軟正黑體，再往上微調 10 像素)
            y_pos = 80 + idx * 58
            color_text = (235, 130, 0) if pid == 1 else (0, 165, 255) # 深天藍對 ID1, 橘色對 ID2
            info_str = f"選手 ID {pid} - 上半身: {up_val}-{up_name} | 下半身: {low_val}-{low_name}"
            frame = draw_text_chinese(frame, info_str, (35, y_pos), 32, color_text)
            
            if kps:
                # 1. 繪製骨架關節連線
                for conn in SKELETON_CONNECTIONS:
                    p1_idx, p2_idx = conn
                    pt1 = get_keypoint_coords(kps, p1_idx)
                    pt2 = get_keypoint_coords(kps, p2_idx)
                    if pt1 and pt2:
                        x1, y1 = pt1
                        x2, y2 = pt2
                        line_color = (235, 130, 0) if pid == 1 else (0, 80, 255) # 深天藍 / 橘紅
                        cv2.line(frame, (x1, y1), (x2, y2), line_color, 3) # 線條加粗到 3
                                
                # 2. 繪製關節點 (小圓點)
                for i in range(len(kps)):
                    pt = get_keypoint_coords(kps, i)
                    if pt:
                        cv2.circle(frame, pt, 3, (255, 255, 255), -1)
                            
                # 3. 繪製人體邊界框 (Bounding Box)
                valid_pts = [get_keypoint_coords(kps, i) for i in range(len(kps))]
                valid_pts = [pt for pt in valid_pts if pt is not None]
                if valid_pts:
                    xs = [pt[0] for pt in valid_pts]
                    ys = [pt[1] for pt in valid_pts]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    
                    padding = 15
                    xmin = max(0, xmin - padding)
                    ymin = max(0, ymin - padding)
                    xmax = min(width - 1, xmax + padding)
                    ymax = min(height - 1, ymax + padding)
                    
                    box_color = (235, 130, 0) if pid == 1 else (0, 165, 255)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 4) # 框線加粗到 4 (粗體)
                    
                    # 框內上方加上 ID 標籤，字體加大到 30 粗體
                    lbl_txt = f"ID {pid}"
                    frame = draw_text_chinese(frame, lbl_txt, (xmin + 8, ymin + 8), 30, box_color)
                    
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"  [OK] 視覺化影片已輸出：{out_path}")
    return True

def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    pred_path = os.path.join(workspace, "final_labels.json")
    
    if not os.path.exists(pred_path):
        print(f"[錯誤] 找不到 final_labels.json。")
        print("請先執行 python merge_labels.py 合併出最新的標記結果！")
        return
        
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
        
    # 建立快速查詢 dict: {(video_name, person_id): item}
    labels_dict = {(r["video_name"], r["person_id"]): r for r in pred_data}
    
    # 獲取所有影片的獨特名稱
    all_videos = sorted(list(set(r["video_name"] for r in pred_data)))
    
    print("=" * 60)
    print(" 🤺 西洋劍動作標註 — 標籤影片視覺化產生器 🤺")
    print("=" * 60)
    print(f" 共有 {len(all_videos)} 部有標註結果的影片。")
    print(f" 影片來源目錄：{VIDEO_DIR}")
    print(f" 輸出影片目錄：{OUTPUT_VIDEO_DIR}")
    print("-" * 60)
    
    # 命令列參數支援：可以執行 python visualize_labels.py test001 指定產生單片
    if len(sys.argv) > 1:
        target_video = sys.argv[1]
        if target_video in all_videos:
            process_video(target_video, labels_dict)
        else:
            print(f"[錯誤] 在標註清單中找不到影片 '{target_video}'")
            print("請確認名稱是否正確，或先跑 python merge_labels.py 更新。")
        return

    # 若無參數，則供使用者輸入，或是選擇全部跑前五部
    print(" 請選擇您想產出的視覺化影片方式：")
    print("  1. 輸入特定影片名稱 (例如：test030)")
    print("  2. 自動產生前 5 部影片 (預設，方便展示)")
    print("  3. 產生所有有標籤的影片 (耗時，需要幾分鐘)")
    
    choice = input("\n請輸入選項 (1/2/3，預設為 2): ").strip()
    
    if choice == "1":
        vname = input("請輸入影片名稱 (例如：test030): ").strip()
        if vname in all_videos:
            process_video(vname, labels_dict)
        else:
            print(f"[警告] 標註清單中找不到影片 '{vname}'。正在嘗試為您尋找...")
            # 模糊比對
            matches = [v for v in all_videos if vname in v]
            if matches:
                print(f"為您找到最接近的影片：{matches[0]}")
                process_video(matches[0], labels_dict)
            else:
                print("[錯誤] 找不到任何符合的影片。")
    elif choice == "3":
        print(f"\n開始批次處理所有 {len(all_videos)} 部影片...")
        success_count = 0
        for vname in all_videos:
            if process_video(vname, labels_dict):
                success_count += 1
        print(f"\n批次處理結束！成功產生 {success_count} 部影片。")
    else:
        # 預設：跑前 5 部有代表性的影片
        showcase_videos = all_videos[:5]
        print(f"\n自動為您產生前 5 部影片以供簡報展示：{showcase_videos}")
        for vname in showcase_videos:
            process_video(vname, labels_dict)
        print(f"\n產生完畢！影片存放在：{OUTPUT_VIDEO_DIR}")

if __name__ == "__main__":
    main()
