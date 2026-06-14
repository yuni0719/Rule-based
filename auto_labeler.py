# =============================================================================
# auto_labeler.py
# 西洋劍短影片動作自動幾何標註管線 — 標註器主程式
#
# 功能：逐一讀取由 ViTPose 產出的 JSON 幀序列，根據幾何規則判定
#       上半身（Upper Label）與下半身（Lower Label）動作標籤，
#       並將結果輸出至 temp_results/{video_name}_p{person_id}.json。
#
# 依賴：os, json, math, glob（內建）；cv2, numpy（需安裝 opencv-python-headless）
# =============================================================================

import os
import json
import math
import glob

try:
    import cv2
    import numpy as np
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False
    print("[WARNING] cv2 未安裝，鏡頭平移補正功能已停用。")

# =============================================================================
# ★ Threshold 參數區（在此調整幾何判定的敏感度）
# =============================================================================

# ── 上半身 ──────────────────────────────────────────────────────────────────
# 直刺判定：肩膀-手肘-手腕夾角須大於此閾值（度數）才算「手臂伸直」
THRUST_ELBOW_ANGLE_MIN = 150.0          # 手臂伸直夾角下限（°）

# 上半身標籤需在多少比例的幀中符合條件才確認（避免單幀誤判）
UPPER_CONFIRM_RATIO = 0.10              # 10% 的幀符合即判定為有效

# ── 下半身 ──────────────────────────────────────────────────────────────────
# 前進 / 後退判定：重心 X 軸在連續幾幀內的累積位移須超過此像素值
FOOTWORK_DISPLACEMENT_PX = 15.0        # 重心累積位移閾值（像素）

# 前進 / 後退判定：需在多少比例的連續移動幀中保持方向一致
FOOTWORK_CONSISTENCY_RATIO = 0.58
ADVANCE_LUNGE_SPLIT_RATIO  = 0.30  # 前進長刺分段比例

DM_BYPASS_THRESHOLD = 0.25
LUNGE_MUTEX_ENABLED = True
LUNGE_TORSO_FIREWALL = 2.5

# 長刺判定：左右腳踝水平距離須大於初始距離的此倍數
LUNGE_ANKLE_STRETCH_RATIO = 1.70       # 腳踝拉開倍率閾值

# 長刺判定：前腳膝蓋彎曲夾角須小於此值（度數），代表深度彎曲
LUNGE_KNEE_BEND_MAX = 115.0            # 前腳膝蓋最大夾角（°）

# 飛刺判定：後腳踝 X 座標超過前腳踝的比例幀數門檻
FLECHE_CROSSOVER_RATIO = 0.15          # 15% 的幀發生腳踝交叉即判定

# 前進長刺判定：將影片切成前段與後段的比例
ADVANCE_LUNGE_SPLIT_RATIO = 0.50       # 前 50% 幀 = 前段，後 50% 幀 = 後段

# 前進長刺判定：前段需符合前進特徵的幀比例
ADVANCE_LUNGE_ADVANCE_RATIO = 0.40

# 前進長刺判定：後段需符合長刺特徵的幀比例
ADVANCE_LUNGE_LUNGE_RATIO = 0.35

# =============================================================================
# ViTPose COCO 關鍵點索引對照表（17點格式）
# =============================================================================
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder
# 7: left_elbow,    8: right_elbow
# 9: left_wrist,   10: right_wrist
# 11: left_hip,    12: right_hip
# 13: left_knee,   14: right_knee
# 15: left_ankle,  16: right_ankle

KP_NOSE          = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER= 6
KP_LEFT_ELBOW    = 7
KP_RIGHT_ELBOW   = 8
KP_LEFT_WRIST    = 9
KP_RIGHT_WRIST   = 10
KP_LEFT_HIP      = 11
KP_RIGHT_HIP     = 12
KP_LEFT_KNEE     = 13
KP_RIGHT_KNEE    = 14
KP_LEFT_ANKLE    = 15
KP_RIGHT_ANKLE   = 16

# =============================================================================
# 輸入 / 輸出路徑設定
# =============================================================================

# ViTPose 輸出的 2D 偵測 JSON 資料夾（相對於本檔案所在目錄）
DETECTION_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "yolo", "ultralytics", "output_videos", "2d_detections"
)

# 原始影片資料夾（用於背景光流鏡頭平移估計）
VIDEO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "yolo", "ultralytics", "test_videos"
)

# 暫存結果輸出資料夾（與本檔案同層）
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "temp_results"
)

# 是否啟用背景光流鏡頭平移補正（需要 cv2；若設 False 則跳過）
CAMERA_CORRECTION_ENABLED = True

# =============================================================================
# 幾何工具函式
# =============================================================================

def apply_camera_correction(kp_sequence, offsets):
    import copy
    new_seq = []
    for i, kp_frame in enumerate(kp_sequence):
        new_frame = copy.deepcopy(kp_frame)
        offset = offsets[i][0] if i < len(offsets) else 0.0
        for pt in new_frame:
            pt['x'] -= offset
        new_seq.append(new_frame)
    return new_seq

def get_keypoint(keypoints: list, idx: int):
    """
    從關鍵點列表取得指定索引的座標。

    支援兩種常見格式：
        格式 List：[[x, y, conf], ...]   → 以數值索引存取
        格式 Dict：[{"x":..,"y":..,"score":..}, ...]  → 以鍵名存取
                   （鍵名亦相容 "confidence" / "prob" / "visibility"）

    回傳 (x, y) 或 None（超出範圍、信心度過低、或無法解析時）。
    """
    if idx >= len(keypoints):
        return None
    kp = keypoints[idx]

    # ── 格式一：dict（{"x": ..., "y": ..., "score": ...}）────────────────
    if isinstance(kp, dict):
        x = kp.get("x", kp.get("X", None))
        y = kp.get("y", kp.get("Y", None))
        if x is None or y is None:
            return None
        # 信心度鍵名相容多種慣例
        conf = kp.get("score",
               kp.get("confidence",
               kp.get("prob",
               kp.get("visibility", 1.0))))
        try:
            x, y, conf = float(x), float(y), float(conf)
        except (TypeError, ValueError):
            return None
        if conf < 0.1:
            return None
        return (x, y)

    # ── 格式二：list / tuple（[x, y] 或 [x, y, conf]）────────────────────
    try:
        if len(kp) >= 2:
            x, y = float(kp[0]), float(kp[1])
            conf = float(kp[2]) if len(kp) > 2 else 1.0
            if conf < 0.1:      # 信心度過低視為無效點
                return None
            return (x, y)
    except (TypeError, ValueError, KeyError):
        return None

    return None


def calc_angle(a, b, c):
    """
    計算以 b 為頂點，a-b-c 三點形成的夾角（度數）。
    a, b, c 均為 (x, y) tuple。
    回傳 0.0～180.0 之間的角度值；任一點為 None 時回傳 None。
    """
    if a is None or b is None or c is None:
        return None
    # 向量 b→a 與 b→c
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    cos_val = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_val))


def euclidean_dist(p1, p2):
    """回傳兩點歐幾里得距離，任一點為 None 時回傳 None。"""
    if p1 is None or p2 is None:
        return None
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def midpoint(p1, p2):
    """回傳兩點中點，任一點為 None 時回傳 None。"""
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

# =============================================================================
# 持劍手判定（以影片整體左右手腕位置判斷慣用手方向）
# =============================================================================

def determine_sword_hand(all_frames_kps: list, person_id=None):
    """
    根據多幀資料，以右手腕和左手腕相對鼻子的 X 位置與攻擊前伸物理法則，
    判斷持劍手是「左」還是「右」。

    西洋劍實戰物理法則：持劍手必須朝向攻擊方向（對手）前伸。
    - 若 attack_dir = +1（朝右），則持劍手的手腕平均 X 座標會更靠右（大於非持劍手）。
    - 若 attack_dir = -1（朝左），則持劍手的手腕平均 X 座標會更靠左（小於非持劍手）。

    回傳 'right' 或 'left'。
    """
    right_wrist_x_list = []
    left_wrist_x_list  = []

    for kps in all_frames_kps:
        rw = get_keypoint(kps, KP_RIGHT_WRIST)
        lw = get_keypoint(kps, KP_LEFT_WRIST)
        if rw:
            right_wrist_x_list.append(rw[0])
        if lw:
            left_wrist_x_list.append(lw[0])

    if not right_wrist_x_list and not left_wrist_x_list:
        return 'right'  # 預設右手

    avg_rw = sum(right_wrist_x_list) / len(right_wrist_x_list) if right_wrist_x_list else 0.0
    avg_lw = sum(left_wrist_x_list)  / len(left_wrist_x_list)  if left_wrist_x_list  else 0.0

    attack_dir = _get_attack_direction(all_frames_kps, 'right', person_id)
    
    # 物理 facts：攻擊方向前伸距離比較
    if avg_rw * attack_dir > avg_lw * attack_dir:
        return 'right'
    else:
        return 'left'

# =============================================================================
# 上半身標籤判定
# =============================================================================

def classify_upper(all_frames_kps: list, sword_hand: str, person_id=None, label_lower=0) -> int:
    """
    判定上半身標籤。

    標籤定義：
    0 → 無（其餘情況）
    1 → 直刺（持劍手前伸，前伸長度大於平均軀幹長度的 1.040 倍）

    物理事實去耦合：
    直刺本質上是持劍手臂向對手方向的極限前伸。
    根據統計，主動攻擊直刺選手的手臂前伸長度通常達到軀幹的 1.1 ~ 1.2 倍或更多。
    我們採用 1.040 倍的門檻，既能包容所有實戰直刺，又能完美過濾防守阻擋與無效擺動。
    """
    if sword_hand == 'right':
        shoulder_idx = KP_RIGHT_SHOULDER
        wrist_idx    = KP_RIGHT_WRIST
        elbow_idx    = KP_RIGHT_ELBOW
    else:
        shoulder_idx = KP_LEFT_SHOULDER
        wrist_idx    = KP_LEFT_WRIST
        elbow_idx    = KP_LEFT_ELBOW

    # 計算平均軀幹長度 (avg_torso)
    torso_lengths = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        mid_s = midpoint(ls, rs)
        mid_h = midpoint(lh, rh)
        t_len = euclidean_dist(mid_s, mid_h)
        if t_len is not None:
            torso_lengths.append(t_len)
    avg_torso = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 150.0

    attack_dir = _get_attack_direction(all_frames_kps, sword_hand, person_id)

    reach_ratios = []
    elbow_angles = []
    for kps in all_frames_kps:
        shoulder = get_keypoint(kps, shoulder_idx)
        wrist    = get_keypoint(kps, wrist_idx)
        elbow    = get_keypoint(kps, elbow_idx)
        if shoulder and wrist:
            reach = (wrist[0] - shoulder[0]) * attack_dir
            reach_ratios.append(reach / avg_torso)
        if shoulder and elbow and wrist:
            angle = calc_angle(shoulder, elbow, wrist)
            if angle is not None:
                elbow_angles.append(angle)

    max_reach_ratio = max(reach_ratios) if reach_ratios else 0.0
    max_elbow_angle = max(elbow_angles) if elbow_angles else 0.0

    # 1. 基礎前伸防線 (下半身主動長刺/飛刺時，放寬門檻至 0.900；其餘維持 1.040)
    reach_th = 0.900 if label_lower in [3, 4, 5] else 1.040
    if max_reach_ratio < reach_th:
        return 0

    # 2. 直刺手肘必須有伸直動作
    # 若最大手肘夾角低於 155.0°，代表手臂全程顯著彎曲，判定非直刺（排除防守揮動）
    if max_elbow_angle < 155.0:
        print(f"      [DEBUG UPPER-MUTEX] 駁回直刺：手臂全程顯著彎曲 (Max Elbow Angle: {max_elbow_angle:.1f}° < 155.0°)")
        return 0

    # 3. 防範後退防守姿勢（Retreat Guard）誤判為直刺
    # 當選手正在後退時，如果他們的前伸長度變動極小（Reach Std < 0.125），說明這只是他們備戰/防守的靜態前伸姿勢，非主動直刺。
    is_retreating = _check_retreat(all_frames_kps, sword_hand, person_id)
    if is_retreating and reach_ratios:
        reach_std = np.std(reach_ratios)
        if reach_std < 0.125:
            print(f"      [DEBUG UPPER-MUTEX] 駁回直刺：選手後退且手臂前伸長度極度靜止 (Std: {reach_std:.4f} < 0.125)")
            return 0

    return 1

# =============================================================================
# 鏡頭平移補正（背景稀疏光流）
# =============================================================================

def _find_video_file(video_name: str) -> str:
    """
    在 VIDEO_DIR 中搜尋與 video_name 對應的影片檔案，
    支援 .mp4 / .avi / .mov / .mkv 四種副檔名（大小寫均可）。
    找到回傳完整路徑，找不到回傳 None。
    """
    for ext in [".mp4", ".MP4", ".avi", ".AVI",
                ".mov", ".MOV", ".mkv", ".MKV"]:
        candidate = os.path.join(VIDEO_DIR, video_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _get_frame_numbers(person_dir: str) -> list:
    """
    從 person_dir 下的 frame_*.json 檔名中擷取幀號序列。
    檔名格式: frame_000001.json → 幀號 1（1-based）。
    回傳: 按檔名排序後的幀號整數列表。
    """
    files = sorted(glob.glob(os.path.join(person_dir, "frame_*.json")))
    numbers = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]   # "frame_000001"
        try:
            numbers.append(int(stem.split("_")[-1]))       # 1
        except ValueError:
            pass
    return numbers



def estimate_camera_pan_from_video(video_name: str, video_dir: str) -> list:
    if not _OPENCV_AVAILABLE:
        return []

    video_path = _find_video_file(video_name)
    if not video_path:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Camera Pan] 無法開啟影片：{video_path}")
        return []

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []

    h, w = first_frame.shape[:2]

    # 1. 解析所有 frame 的 fencer bounding boxes
    frame_to_bboxes = {}
    for pid in ['1', '2']:
        pdir = os.path.join(video_dir, pid)
        if not os.path.exists(pdir): continue
        for f in os.listdir(pdir):
            if not f.endswith(".json"): continue
            try:
                fidx = int(os.path.splitext(f)[0].split("_")[-1])
            except:
                continue
            with open(os.path.join(pdir, f), 'r') as jf:
                try:
                    data = json.load(jf)
                except:
                    continue
                kps = data.get('keypoints', [])
                if not kps: continue
                xs = [kp['x'] for kp in kps if 'x' in kp]
                ys = [kp['y'] for kp in kps if 'y' in kp]
                if not xs or not ys: continue
                
                # 擴張 Bounding Box 40 pixels 以覆蓋手臂、劍與模糊殘影
                xmin, xmax = max(0, min(xs) - 40), min(w, max(xs) + 40)
                ymin, ymax = max(0, min(ys) - 40), min(h, max(ys) + 40)
                
                if fidx not in frame_to_bboxes:
                    frame_to_bboxes[fidx] = []
                frame_to_bboxes[fidx].append((xmin, xmax, ymin, ymax))

    # 2. 準備基礎靜態 Mask
    base_mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * 0.03)
    y_end = int(h * 0.80)
    x_start = int(w * 0.05)
    x_end = int(w * 0.95)
    base_mask[y_start:y_end, x_start:x_end] = 255

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    def get_dynamic_mask(fidx):
        mask = base_mask.copy()
        if fidx in frame_to_bboxes:
            for xmin, xmax, ymin, ymax in frame_to_bboxes[fidx]:
                mask[ymin:ymax, xmin:xmax] = 0
        return mask

    feature_params = dict(
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7
    )

    offsets = []
    cum_dx, cum_dy = 0.0, 0.0
    frame_idx = 0

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    curr_mask = get_dynamic_mask(frame_idx)
    prev_pts  = cv2.goodFeaturesToTrack(prev_gray, mask=curr_mask, **feature_params)
    offsets.append((0.0, 0.0))

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        dx, dy = 0.0, 0.0

        if prev_pts is not None and len(prev_pts) >= 4:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params
            )
            if curr_pts is not None and status is not None:
                good_prev = prev_pts[status.ravel() == 1]
                good_curr = curr_pts[status.ravel() == 1]

                # 動態排除落入選手框內的追蹤點
                if len(good_curr) > 0:
                    valid_idx = []
                    for i, pt in enumerate(good_curr):
                        px, py = pt.ravel()[0], pt.ravel()[1]
                        in_fencer = False
                        if frame_idx in frame_to_bboxes:
                            for xmin, xmax, ymin, ymax in frame_to_bboxes[frame_idx]:
                                if xmin <= px <= xmax and ymin <= py <= ymax:
                                    in_fencer = True
                                    break
                        if not in_fencer:
                            valid_idx.append(i)
                    
                    good_prev = good_prev[valid_idx]
                    good_curr = good_curr[valid_idx]

                if len(good_prev) >= 4:
                    diffs = (good_curr - good_prev).reshape(-1, 2)
                    mags = np.linalg.norm(diffs, axis=1)
                    p90_mag = np.percentile(mags, 90)
                    keep_mask = np.ones(len(good_prev), dtype=bool)
                    if p90_mag > 1.5:
                        static_mask = mags < 0.15
                        keep_mask = ~static_mask
                    filtered_prev = good_prev[keep_mask]
                    filtered_curr = good_curr[keep_mask]

                    if len(filtered_prev) >= 4:
                        M, _ = cv2.estimateAffinePartial2D(
                            filtered_prev.reshape(-1, 1, 2),
                            filtered_curr.reshape(-1, 1, 2),
                            method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                            maxIters=2000,
                            confidence=0.99
                        )
                        if M is not None:
                            dx = float(M[0, 2])
                            dy = float(M[1, 2])

        cum_dx += dx
        cum_dy += dy
        offsets.append((cum_dx, cum_dy))

        prev_gray = curr_gray
        # good_curr is defined only if condition met.
        num_good = len(good_curr) if 'good_curr' in locals() else 0
        if num_good < 100:
            curr_mask = get_dynamic_mask(frame_idx)
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=curr_mask, **feature_params)
        else:
            prev_pts = good_curr.reshape(-1, 1, 2)

    cap.release()
    return offsets


# =============================================================================
# 下半身標籤判定
# =============================================================================

def _get_center_of_mass_x(kps: list):
    """
    以「雙髖中點 X」近似身體重心 X 座標。
    若髖部不可用則退而求其次用雙踝中點。
    """
    lh = get_keypoint(kps, KP_LEFT_HIP)
    rh = get_keypoint(kps, KP_RIGHT_HIP)
    mid = midpoint(lh, rh)
    if mid:
        return mid[0]

    la = get_keypoint(kps, KP_LEFT_ANKLE)
    ra = get_keypoint(kps, KP_RIGHT_ANKLE)
    mid2 = midpoint(la, ra)
    return mid2[0] if mid2 else None


def _determine_front_back_ankle(all_frames_kps: list, sword_hand: str, person_id=None):
    attack_dir = _get_attack_direction(all_frames_kps, sword_hand, person_id)
    right_ankle_xs = []
    left_ankle_xs  = []

    for kps in all_frames_kps:
        ra = get_keypoint(kps, KP_RIGHT_ANKLE)
        la = get_keypoint(kps, KP_LEFT_ANKLE)
        if ra: right_ankle_xs.append(ra[0])
        if la: left_ankle_xs.append(la[0])

    avg_ra = sum(right_ankle_xs) / len(right_ankle_xs) if right_ankle_xs else None
    avg_la = sum(left_ankle_xs)  / len(left_ankle_xs)  if left_ankle_xs  else None

    if avg_ra is None or avg_la is None:
        if sword_hand == 'right':
            return (KP_RIGHT_ANKLE, KP_LEFT_ANKLE)
        else:
            return (KP_LEFT_ANKLE, KP_RIGHT_ANKLE)

    if attack_dir == 1:
        if avg_ra > avg_la: return (KP_RIGHT_ANKLE, KP_LEFT_ANKLE)
        else: return (KP_LEFT_ANKLE, KP_RIGHT_ANKLE)
    else:
        if avg_ra < avg_la: return (KP_RIGHT_ANKLE, KP_LEFT_ANKLE)
        else: return (KP_LEFT_ANKLE, KP_RIGHT_ANKLE)


def _get_attack_direction(all_frames_kps: list, sword_hand: str,
                          person_id=None) -> int:
    """
    根據人物約定規則判定影片中選手的「攻擊方向」：
    person_id = 1 → 攻擊方向朝右 (+1)
    person_id = 2 → 攻擊方向朝左 (-1)
    """
    if person_id is not None:
        return +1 if int(person_id) == 1 else -1
    return +1  # 預設朝右


def _check_advance(all_frames_kps: list, sword_hand: str,
                   person_id=None, is_global_check_for_lunge=False) -> bool:
    """
    判定「前進」：重心 X 朝攻擊方向穩定移動。

    重構說明（v4）：「前後段平均重心推移趨勢法」
        步驟 1 — 取攻擊方向 attack_dir。
        步驟 2 — 前後段平均重心推移量：
                    net_move = (avg_back - avg_front) * attack_dir
                    正值 = 朝攻擊方向移動。
        步驟 3 — 【符號互斥】net_move <= 0 → 直接回傳 False
                    （整體趨勢是後退方向，不可能是前進）。
        步驟 4 — 依呼叫情境自動切換門檻：
                    總幀數 >= 8：net_move >= 5.0 像素
                    總幀數 <  8：net_move >= 3.0 像素（雙階防火牆，排除原地深蹲晃動雜訊）
        步驟 5 — 逐幀朝攻擊方向移動的比例 >= FOOTWORK_CONSISTENCY_RATIO。

    【注意】Camera Pan 無法純靠骨架 2D 幾何排除，需靠上游流程處理。
    """
    cx_list = [_get_center_of_mass_x(kps) for kps in all_frames_kps]
    cx_list = [v for v in cx_list if v is not None]
    if len(cx_list) < 3:
        return False

    attack_dir = _get_attack_direction(all_frames_kps, sword_hand, person_id)

    # ── 條件 A：前後段平均重心推移量 ────────────────────────────────────
    mid       = max(1, len(cx_list) // 2)
    avg_front = sum(cx_list[:mid]) / mid
    avg_back  = sum(cx_list[mid:]) / max(1, len(cx_list) - mid)
    net_move  = (avg_back - avg_front) * attack_dir

    # 【符號互斥】net_move <= 0 代表整體朝後退方向，不可能是前進
    if net_move <= 0:
        return False

    # 計算平均軀幹長度 (avg_torso) 用於進行尺度無關化門檻與蓄力鎖補正
    torso_lengths = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        mid_s = midpoint(ls, rs)
        mid_h = midpoint(lh, rh)
        t_len = euclidean_dist(mid_s, mid_h)
        if t_len is not None:
            torso_lengths.append(t_len)
    avg_torso = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 150.0

    # ── 區間位移（彈跳殘片過濾牆雙重保險） ─────────────────────────────
    split_idx = max(1, int(len(cx_list) * ADVANCE_LUNGE_SPLIT_RATIO))
    front_half = cx_list[:split_idx]
    
    has_bounce_back = False
    
    # 1. 身體重心位移過濾
    if len(front_half) >= 2:
        front_move = (front_half[-1] - front_half[0]) * attack_dir

        if front_move < -20.0:
            print(f"      [DEBUG ADVANCE-BOUNCE-LOCK] 駁回：前半段重心位移 {front_move:.2f} px 顯示有向後彈跳殘片 (< -20.0 px)")
            has_bounce_back = True

    # 2. 雙腳踝開頭撤步彈跳殘片過濾（物理 facts 雙重保險）
    ankle_mid_list = []
    for kps in all_frames_kps:
        la = get_keypoint(kps, KP_LEFT_ANKLE)
        ra = get_keypoint(kps, KP_RIGHT_ANKLE)
        mid_a = midpoint(la, ra)
        if mid_a:
            ankle_mid_list.append(mid_a[0])
            
    if len(ankle_mid_list) >= 2:
        local_len = min(5, len(ankle_mid_list))
        ankle_local_move = (ankle_mid_list[local_len - 1] - ankle_mid_list[0]) * attack_dir
        if ankle_local_move < -20.0:
            print(f"      [DEBUG ADVANCE-BOUNCE-LOCK] 駁回：開頭前幾幀雙踝中點位移 {ankle_local_move:.2f} px 顯示有向後撤步彈跳殘片 (< -20.0 px)")
            has_bounce_back = True
            
    if has_bounce_back and not is_global_check_for_lunge:
        return False

    front_ankle_idx, back_ankle_idx = _determine_front_back_ankle(all_frames_kps, sword_hand, person_id)

    # ── ADVANCE-FOOT:（前腳必須要有實質前進） ──
    if not is_global_check_for_lunge and person_id is not None:
        front_ankle_list = [get_keypoint(kps, front_ankle_idx)[0] for kps in all_frames_kps if get_keypoint(kps, front_ankle_idx) is not None]
        if len(front_ankle_list) >= 2:
            front_ankle_move = (front_ankle_list[-1] - front_ankle_list[0]) * attack_dir
            if front_ankle_move < 0.05 * avg_torso:
                print(f"      [DEBUG ADVANCE-FOOT] 駁回：前腳幾乎沒有前進 ({front_ankle_move:.2f} px < {0.05 * avg_torso:.2f} px)")
                return False

    # ── ADVANCE-SQUAT ──
    if not is_global_check_for_lunge and person_id is not None:
        hip_y_list = [midpoint(get_keypoint(kps, KP_LEFT_HIP), get_keypoint(kps, KP_RIGHT_HIP))[1] for kps in all_frames_kps if get_keypoint(kps, KP_LEFT_HIP) and get_keypoint(kps, KP_RIGHT_HIP)]
        if len(hip_y_list) >= 2:
            if (max(hip_y_list) - min(hip_y_list)) > 1.2 * avg_torso:
                print(f"      [DEBUG ADVANCE-SQUAT] 駁回：明顯深蹲 (Drop: {max(hip_y_list) - min(hip_y_list):.2f} px > 1.2 torso)")
                return False

    # ── ADVANCE-RETURN (往前又往後) ──
    if not is_global_check_for_lunge and person_id is not None:
        net_disp = (cx_list[-1] - cx_list[0]) * attack_dir
        # Must recalculate max_fwd here since we use it below
        forward_peaks_local = [(cx - cx_list[0]) * attack_dir for cx in cx_list]
        max_fwd_local = max(forward_peaks_local) if forward_peaks_local else 0.0
        # If advanced significantly but returned almost to start
        if max_fwd_local > 0.30 * avg_torso and net_disp < 0.25 * avg_torso:
            print(f"      [DEBUG ADVANCE-RETURN] 駁回：前進後退回起點 (Max Fwd: {max_fwd_local:.2f}, Net Disp: {net_disp:.2f})")
            return False

    # ── 3. 雙腳雙向運動（Feet Dual-Movement，先退後移） ──
    if not is_global_check_for_lunge and person_id is not None:
        retreat_dir = -attack_dir
        ba_list = [get_keypoint(kps, back_ankle_idx)[0] for kps in all_frames_kps if get_keypoint(kps, back_ankle_idx) is not None]
        if len(ba_list) >= 2:
            # 引入 3 幀滾動平滑處理以過濾單幀骨架漂移尖峰
            ba_smoothed = []
            for idx_ba in range(len(ba_list)):
                window = ba_list[max(0, idx_ba-1):min(len(ba_list), idx_ba+2)]
                ba_smoothed.append(sum(window)/len(window))
            ba_retreat_peaks = [(ba - ba_smoothed[0]) * retreat_dir for ba in ba_smoothed]
            ba_advance_peaks = [(ba - ba_smoothed[0]) * attack_dir for ba in ba_smoothed]
            max_ba_ret = max(ba_retreat_peaks) if ba_retreat_peaks else 0.0
            max_ba_adv = max(ba_advance_peaks) if ba_advance_peaks else 0.0
            # 放寬後腳退後門檻至 0.18，且當後腳前進幅度極大（>= 2.0 torso）時放行避免誤殺
            if max_ba_ret > 0.18 * avg_torso and max_ba_adv > 0.10 * avg_torso and max_ba_adv < 2.0 * avg_torso:
                print(f"      [DEBUG ADVANCE-FEET-DUAL-LOCK] 駁回：雙腳雙向移動 (Ret: {max_ba_ret/avg_torso:.3f}, Adv: {max_ba_adv/avg_torso:.3f})")
                return False

    # ── 4. 前進雙向移動收尾（Advance Post-Drop Lock，先前進又後退） ──
    if not is_global_check_for_lunge and person_id is not None and len(cx_list) >= 4:
        # 計算影片結尾處雙腳中點的淨位移，若雙腳確實前移（rel_am_end >= 0.40 torso）則不視為回退收尾
        la_list = [get_keypoint(kps, KP_LEFT_ANKLE) for kps in all_frames_kps]
        ra_list = [get_keypoint(kps, KP_RIGHT_ANKLE) for kps in all_frames_kps]
        am_end = None
        am_start = None
        if la_list[0] and ra_list[0]:
            am_start = midpoint(la_list[0], ra_list[0])[0]
        if la_list[-1] and ra_list[-1]:
            am_end = midpoint(la_list[-1], ra_list[-1])[0]
        rel_am_end = 0.0
        if am_start is not None and am_end is not None:
            rel_am_end = (am_end - am_start) * attack_dir / avg_torso
            
        max_idx = cx_list.index(max(cx_list) if attack_dir > 0 else min(cx_list))
        if max_idx < len(cx_list) - 3:
            post_cx = cx_list[max_idx:]
            avg_post = sum(post_cx) / len(post_cx)
            max_val = cx_list[max_idx]
            drop_fwd = (max_val - avg_post) * attack_dir
            drop_ratio = drop_fwd / avg_torso
            if drop_ratio > 0.12 and rel_am_end < 0.40:
                print(f"      [DEBUG ADVANCE-POST-DROP] 駁回：先前進又後退收尾過多 (Drop Ratio: {drop_ratio:.3f} > 0.12, Ankle Adv: {rel_am_end:.3f} < 0.40)")
                return False

    # 計算平均軀幹長度 (avg_torso) 用於進行尺度無關化門檻補正
    torso_lengths = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        mid_s = midpoint(ls, rs)
        mid_h = midpoint(lh, rh)
        t_len = euclidean_dist(mid_s, mid_h)
        if t_len is not None:
            torso_lengths.append(t_len)
    avg_torso = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 150.0

    # ── 5. 雙向運動（Dual-Movement，擺動蓄力） ──
    if not is_global_check_for_lunge and person_id is not None:
        retreat_dir = -attack_dir
        forward_peaks = [(cx - cx_list[0]) * attack_dir for cx in cx_list]
        max_fwd = max(forward_peaks) if forward_peaks else 0.0
        
        max_cx_idx = forward_peaks.index(max_fwd) if forward_peaks else 0
        retreat_from_max = [(cx - cx_list[max_cx_idx]) * retreat_dir for cx in cx_list[max_cx_idx:]]
        max_ret = max(retreat_from_max) if retreat_from_max else 0.0
        
        initial_retreat = [(cx - cx_list[0]) * retreat_dir for cx in cx_list[:max_cx_idx+1]]
        max_initial_ret = max(initial_retreat) if initial_retreat else 0.0
        max_ret = max(max_ret, max_initial_ret)
        
        net_move_ratio = abs(net_move) / avg_torso
        if net_move_ratio < 0.30:  # dm_th = 0.30
            if max_fwd > 0.08 * avg_torso and max_ret > 0.08 * avg_torso:
                smaller = min(max_fwd, max_ret)
                larger = max(max_fwd, max_ret)
                if smaller >= 0.50 * larger:
                    print(f"      [DEBUG ADVANCE-DM] 駁回：重心雙向擺動幅度過大 (Fwd: {max_fwd/avg_torso:.3f}, Ret: {max_ret/avg_torso:.3f})")
                    return False

    # 尺度無關化位移門檻：整段影片需 >= 0.20 倍軀幹長度，短序列 >= 0.13 倍軀幹長度
    threshold = avg_torso * 0.32 if len(cx_list) >= 8 else avg_torso * 0.20
    if net_move < threshold:
        return False

    # ── 條件 B：逐幀累計趨勢——統計朝攻擊方向移動的幀比例 ───────────────
    deltas     = [cx_list[i+1] - cx_list[i] for i in range(len(cx_list) - 1)]
    consistent = sum(1 for d in deltas if d * attack_dir > 0)
    ratio      = consistent / len(deltas)

    if is_global_check_for_lunge:
        return ratio >= 0.35
    if net_move >= avg_torso * 0.40:
        return True
    return ratio >= FOOTWORK_CONSISTENCY_RATIO

def _check_retreat(all_frames_kps: list, sword_hand: str,
                   person_id=None) -> bool:
    """
    判定「後退」：重心 X 朝攻擊方向的反方向穩定移動。

    重構說明（v4）：「前後段平均重心推移趨勢法」
        net_move = (avg_back - avg_front) * attack_dir
        後退時 net_move 為負值。

        【符號互斥】若 net_move >= 0，直接回傳 False（
            整體趨勢是前進方向，不可能是後退）。

        門檻：
          總幀數 >= 8：net_move <= -0.15 * avg_torso
          總幀數 <  8：net_move <= -0.10 * avg_torso
        且逐幀朝後退方向移動的比例 >= FOOTWORK_CONSISTENCY_RATIO。
    """
    cx_list = [_get_center_of_mass_x(kps) for kps in all_frames_kps]
    cx_list = [v for v in cx_list if v is not None]
    if len(cx_list) < 3:
        return False

    attack_dir  = _get_attack_direction(all_frames_kps, sword_hand, person_id)
    retreat_dir = -attack_dir

    # ── 條件 A：前後段平均重心推移量 ────────────────────────────────────
    mid       = max(1, len(cx_list) // 2)
    avg_front = sum(cx_list[:mid]) / mid
    avg_back  = sum(cx_list[mid:]) / max(1, len(cx_list) - mid)
    net_move  = (avg_back - avg_front) * attack_dir

    # 【符號互斥】net_move >= 0 代表整體朝前進方向，不可能是後退
    if net_move >= 0:
        return False

    # 計算平均軀幹長度 (avg_torso) 用於進行尺度無關化門檻補正
    torso_lengths = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        mid_s = midpoint(ls, rs)
        mid_h = midpoint(lh, rh)
        t_len = euclidean_dist(mid_s, mid_h)
        if t_len is not None:
            torso_lengths.append(t_len)
    avg_torso = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 150.0

    # ── 3. 雙向運動（Dual-Movement，擺動蓄力） ──
    if person_id is not None:
        forward_peaks = [(cx - cx_list[0]) * attack_dir for cx in cx_list]
        retreat_peaks = [(cx - cx_list[0]) * retreat_dir for cx in cx_list]
        max_fwd = max(forward_peaks) if forward_peaks else 0.0
        max_ret = max(retreat_peaks) if retreat_peaks else 0.0
        net_move_ratio = abs(net_move) / avg_torso
        if net_move_ratio < 0.30:  # dm_th = 0.30
            if max_fwd > 0.08 * avg_torso and max_ret > 0.08 * avg_torso:
                smaller = min(max_fwd, max_ret)
                larger = max(max_fwd, max_ret)
                if smaller >= 0.50 * larger:
                    print(f"      [DEBUG RETREAT-DM] 駁回：重心雙向擺動幅度過大 (Fwd: {max_fwd/avg_torso:.3f}, Ret: {max_ret/avg_torso:.3f})")
                    return False

    # 尺度無關化後退位移門檻：整段影片需 <= -0.20 倍軀幹長度，短序列 <= -0.13 倍軀幹長度
    threshold = avg_torso * -0.20 if len(cx_list) >= 8 else avg_torso * -0.13
    if net_move > threshold:
        return False

    # ── 條件 B：逐幀累計趨勢——統計朝後退方向移動的幀比例 ───────────────
    deltas     = [cx_list[i+1] - cx_list[i] for i in range(len(cx_list) - 1)]
    consistent = sum(1 for d in deltas if d * retreat_dir > 0)
    ratio      = consistent / len(deltas)

    if net_move <= -avg_torso * 0.25:
        return True
    return ratio >= FOOTWORK_CONSISTENCY_RATIO

def _check_lunge(all_frames_kps: list, sword_hand: str,
                 front_ankle_idx: int, back_ankle_idx: int,
                 external_init_dist=None, person_id=None, is_sliced_lunge=False) -> bool:
    """
    判定「長刺（Lunge）」。

    重構說明（v3）：「區間最大表現判定法」
        v2 的 Bug：用 `cond_a and cond_b` 要求每幀同時滿足腳踝拉開
        與膝蓋深蹲，但在動態長刺中：
          - 腳踝拉到最開（重心橫移最大）通常在腳懸空時
          - 膝蓋蹲到最深（前腳承重最大）通常在落地後的撐地瞬間
        兩個極值存在跨幀時間差，永遠不在同一幀同時出現，
        導致 lunge_count 永遠為 0，符合比例永遠是 0.00。

        v3 修正：在後段幀（後 70%）中「分別」找出：
          - max_observed_ratio：整個後段的【最大腳踝拉開倍率】
          - min_observed_angle：整個後段的【最小膝蓋彎曲角度】
        再用這兩個獨立極值做「聯集雙軌」判定：

        組合一（極致大跨步型）：
            max_observed_ratio >= 1.65  且  min_observed_angle <= 125.0°

        組合二（實戰深蹲大弓箭步型）：
            max_observed_ratio >= 1.35  且  min_observed_angle <= 110.0°

        任一組合成立即 return True。
    """
    LUNGE_ANKLE_HARD_RATIO = 1.65    # 組合一：嚴格腳踝倍率下限
    LUNGE_ANKLE_SOFT_RATIO = 1.25    # 組合二：寬鬆腳踝倍率下限（包容斜向透視壓縮至 1.31 實測值）
    LUNGE_KNEE_MICRO_BEND  = 125.0   # 組合一：前膝「微蹲」上限（度）
    LUNGE_KNEE_DEEP_BEND   = 110.0   # 組合二：前膝「深蹲」上限（度）

    if len(all_frames_kps) < 5:
        return False

    # ── 計算全片平均軀幹長度（Torso Length） ──────────────────────────
    # 軀幹距離公式：左肩(5)與右肩(6)中點，到左髖(11)與右髖(12)中點的歐幾里得距離
    torso_lengths = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        
        mid_shoulder = midpoint(ls, rs)
        mid_hip      = midpoint(lh, rh)
        
        t_len = euclidean_dist(mid_shoulder, mid_hip)
        if t_len is not None:
            torso_lengths.append(t_len)
            
    avg_torso_len = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 150.0  # 預設 150 像素以防無有效點

    # ── 計算或使用傳入的初始腳踝距離基準 ────────────────────────────
    if external_init_dist is not None and external_init_dist > 0:
        init_dist_avg = external_init_dist
    else:
        init_frames    = max(1, int(len(all_frames_kps) * 0.20))
        init_distances = []
        for kps in all_frames_kps[:init_frames]:
            fa = get_keypoint(kps, front_ankle_idx)
            ba = get_keypoint(kps, back_ankle_idx)
            d  = euclidean_dist(fa, ba)
            if d is not None:
                init_distances.append(d)

        if not init_distances:
            return False

        init_dist_avg = sum(init_distances) / len(init_distances)

    if init_dist_avg < 1e-6:
        return False

    # ── 確定前腳同側的髖、膝索引 ──────────────────────────────────────
    front_knee_idx = (KP_RIGHT_KNEE if front_ankle_idx == KP_RIGHT_ANKLE
                      else KP_LEFT_KNEE)
    front_hip_idx  = (KP_RIGHT_HIP  if front_ankle_idx == KP_RIGHT_ANKLE
                      else KP_LEFT_HIP)

    # ── 後段幀（後 70%）極值掃描 ──────────────────────────────────────
    # 分別獨立追蹤腳踝倍率的最大值與膝蓋角度的最小值，
    # 允許兩者出現在不同幀（解決跨幀時間差問題）。
    # 如果已經是切分後的 sliced lunge，不跳過前 30% 幀。
    check_start  = 0 if is_sliced_lunge else int(len(all_frames_kps) * 0.30)
    check_frames = all_frames_kps[check_start:]
    if not check_frames:
        return False

    observed_ratios = []
    observed_knee_angles = []

    for kps in check_frames:
        # ── 腳踝倍率 ──────────────────────────────────────────────
        fa = get_keypoint(kps, front_ankle_idx)
        ba = get_keypoint(kps, back_ankle_idx)
        d  = euclidean_dist(fa, ba)
        if d is not None:
            observed_ratios.append(d / init_dist_avg)

        # ── 膝蓋彎曲角度（獨立計算，不依賴腳踝是否有效）────────────
        hip        = get_keypoint(kps, front_hip_idx)
        knee       = get_keypoint(kps, front_knee_idx)
        ankle      = get_keypoint(kps, front_ankle_idx)
        knee_angle = calc_angle(hip, knee, ankle)
        if knee_angle is not None:
            observed_knee_angles.append(knee_angle)

    if not observed_ratios:
        return False

    # 進行強健性分位數估計（排除單幀關節跳動噪聲）
    observed_ratios.sort()
    observed_knee_angles.sort()

    # 選擇第 90 百分位數（或倒數第 3 個最大值，若長度不足則取最大值）
    max_idx = max(0, len(observed_ratios) - 3)
    max_observed_ratio = observed_ratios[max_idx]

    if observed_knee_angles:
        # 選擇第 10 百分位數（或第 3 個最小值，若長度不足則取最小值）
        min_idx = min(len(observed_knee_angles) - 1, 2)
        min_observed_angle = observed_knee_angles[min_idx]
        angle_valid = True
    else:
        min_observed_angle = 180.0
        angle_valid = False

    # ── 聯集雙軌判定（用獨立極值，允許跨幀）──────────────────────────
    if angle_valid:
        combo_1 = (max_observed_ratio >= LUNGE_ANKLE_HARD_RATIO and
                   min_observed_angle  <= LUNGE_KNEE_MICRO_BEND)
        combo_2 = (max_observed_ratio >= LUNGE_ANKLE_SOFT_RATIO and
                   min_observed_angle  <= LUNGE_KNEE_DEEP_BEND)
        # 組合三：若膝蓋極端彎曲（<= 95.0度），即便腳踝水平拉開極小（>= 1.02倍）也視為有效長刺
        combo_3 = (max_observed_ratio >= 1.10 and min_observed_angle <= 95.0)
        result = combo_1 or combo_2 or combo_3
    else:
        result = (max_observed_ratio >= LUNGE_ANKLE_SOFT_RATIO)

    # ── 軀幹比例尺防火牆 ──────────────────────────────────────────────────
    max_observed_dist = max_observed_ratio * init_dist_avg
    full_t_lens = []
    for kps in all_frames_kps:
        ls, rs = get_keypoint(kps, KP_LEFT_SHOULDER), get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh, rh = get_keypoint(kps, KP_LEFT_HIP), get_keypoint(kps, KP_RIGHT_HIP)
        t = euclidean_dist(midpoint(ls, rs), midpoint(lh, rh))
        if t: full_t_lens.append(t)
    avg_torso_full = sum(full_t_lens) / len(full_t_lens) if full_t_lens else avg_torso_len

    # 1. 絕對超大跨距：若最大跨距達軀幹長 2.30 以上，且距增幅大於 1.15，確定是長刺(無視膝蓋彎度)
    if max_observed_dist >= avg_torso_full * 2.30 and max_observed_ratio >= 1.15:
        if not angle_valid or min_observed_angle <= 135.0:
            result = True

    # 2. 底線：預設為 2.50 (sliced lunge 為 2.35)
    firewall = 2.35 if is_sliced_lunge else 2.50
    if angle_valid:
        if min_observed_angle <= 100.0: firewall = min(firewall, 2.10)
        elif min_observed_angle <= 110.0: firewall = min(firewall, 2.30)
        
    if max_observed_dist < avg_torso_full * firewall:
        result = False

    # 3. 絕對救援機制 (Super Wide Stance)：如果跨距達到軀幹的 3.30 倍以上，必定是長刺 (避免 sliced lunge 被誤判)
    if max_observed_dist >= avg_torso_full * 3.30:
        if not angle_valid or min_observed_angle <= 135.0:
            result = True

    # ── DEBUG 輸出 ─────────────────────────────────────────────────
    angle_str = (f"{min_observed_angle:.2f}" if angle_valid else "N/A（偵測失敗）")
    print(f"      [DEBUG LUNGE] 初始平均腳踝距離: {init_dist_avg:.2f} px")
    print(f"      [DEBUG LUNGE] 影片最大觀測倍率: {max_observed_ratio:.2f}"
          f"  (要求: {LUNGE_ANKLE_HARD_RATIO} 或 {LUNGE_ANKLE_SOFT_RATIO})")
    print(f"      [DEBUG LUNGE] 影片最小膝蓋角度: {angle_str} 度"
          f"  (要求: {LUNGE_KNEE_MICRO_BEND} 或 {LUNGE_KNEE_DEEP_BEND})")
    combo_tag = ("組合一" if (angle_valid and combo_1) else
                 "組合二" if (angle_valid and combo_2) else
                 "退化防線" if (not angle_valid and result) else "[X] 不符合")
    print(f"      [DEBUG LUNGE] 判定結果: {'[Y] LUNGE' if result else '[X] 不符合'}"
          f"  ({combo_tag})")

    return result

def _check_fleche(all_frames_kps: list, sword_hand: str,
                  front_ankle_idx: int, back_ankle_idx: int,
                  person_id=None) -> bool:
    """
    判定「飛刺（Flèche）」：
    後腳踝的 X 座標超過前腳踝的 X 座標（腳踝 X 軸交叉換位），
    或者後腳極度靠近前腳（雙腳間距大幅縮小）且帶有明顯的前進淨位移。
    """
    if len(all_frames_kps) < 3:
        return False

    if _check_retreat(all_frames_kps, sword_hand, person_id):
        print("      [DEBUG FLECHE] 駁回：選手正在後退。")
        return False

    cx_list = [_get_center_of_mass_x(kps) for kps in all_frames_kps]
    cx_list = [v for v in cx_list if v is not None]
    net_move = 0.0
    if len(cx_list) >= 2:
        attack_dir = _get_attack_direction(all_frames_kps, sword_hand, person_id)
        mid = max(1, len(cx_list) // 2)
        avg_front = sum(cx_list[:mid]) / mid
        avg_back  = sum(cx_list[mid:]) / max(1, len(cx_list) - mid)
        net_move  = (avg_back - avg_front) * attack_dir
        if net_move < 5.0:
            print(f"      [DEBUG FLECHE] 駁回：向前淨位移 {net_move:.2f} px 不足 5.0 px。")
            return False

    # 計算軀幹長度
    t_lens = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        mid_s = midpoint(ls, rs)
        mid_h = midpoint(lh, rh)
        t = euclidean_dist(mid_s, mid_h)
        if t: t_lens.append(t)
    avg_torso = sum(t_lens) / len(t_lens) if t_lens else 150.0

    # 判斷腳間距變化
    start_fa = get_keypoint(all_frames_kps[0], front_ankle_idx)
    start_ba = get_keypoint(all_frames_kps[0], back_ankle_idx)
    end_fa = get_keypoint(all_frames_kps[-1], front_ankle_idx)
    end_ba = get_keypoint(all_frames_kps[-1], back_ankle_idx)
    
    start_dist = abs(start_fa[0] - start_ba[0]) if start_fa and start_ba else 0
    end_dist = abs(end_fa[0] - end_ba[0]) if end_fa and end_ba else 0
    diff_ratio = (end_dist - start_dist) / avg_torso

    crossover_count = 0
    valid_count     = 0

    init_fa = get_keypoint(all_frames_kps[0], front_ankle_idx)
    init_ba = get_keypoint(all_frames_kps[0], back_ankle_idx)
    if init_fa is None or init_ba is None:
        return False

    front_is_larger_x = (init_fa[0] > init_ba[0])

    for kps in all_frames_kps:
        fa = get_keypoint(kps, front_ankle_idx)
        ba = get_keypoint(kps, back_ankle_idx)
        if fa is None or ba is None:
            continue
        valid_count += 1
        diff = ba[0] - fa[0] if front_is_larger_x else fa[0] - ba[0]
        if diff > 0 and diff < 150:
            crossover_count += 1

    if valid_count == 0:
        return False

    crossover_ratio = crossover_count / valid_count
    is_cross = crossover_count >= 1

    if is_cross:
        print(f"      [DEBUG FLECHE] 判定飛刺: Crossover={is_cross} ({crossover_ratio:.2f})")
        return True
    return False


def _check_advance_lunge(all_frames_kps: list, sword_hand: str,
                         front_ankle_idx: int, back_ankle_idx: int,
                         person_id=None) -> bool:
    """
    判定「前進長刺（Advance Lunge）」：
    前段（前 ADVANCE_LUNGE_SPLIT_RATIO 比例）符合「前進」特徵，
    後段（後 1-ADVANCE_LUNGE_SPLIT_RATIO 比例）符合「長刺」特徵。
    """
    # ── 剛性行為互斥保險：檢查整段影片是否符合純前進 ─────────────────────
    is_global_advance = _check_advance(all_frames_kps, sword_hand, person_id, is_global_check_for_lunge=True)
    print(f"      [DEBUG ADVANCE-LUNGE-MUTEX] 全片純前進檢查結果: {is_global_advance}")

    if not is_global_advance:
        print("      [DEBUG ADVANCE-LUNGE-MUTEX] 駁回：全片不具備純前進事實，判定非前進長刺。")
        return False

    n = len(all_frames_kps)
    if n < 6:
        return False

    # ── (A) 全域分母鎖定 ─────────────────────────────────────────────────
    # 在切半之前，以整段影片最初始的前 20% 幀計算腳踝距離基準，避免切半後分母暴跌
    global_init_dist = None
    init_frames      = max(1, int(n * 0.20))
    init_distances   = []
    for kps in all_frames_kps[:init_frames]:
        fa = get_keypoint(kps, front_ankle_idx)
        ba = get_keypoint(kps, back_ankle_idx)
        d  = euclidean_dist(fa, ba)
        if d is not None:
            init_distances.append(d)
    
    if init_distances:
        global_init_dist = sum(init_distances) / len(init_distances)

    # ── (A-2) 動態尋找長刺啟動幀 (lunge_start_frame) ──
    # 尋找全片雙腳拉得最開的一幀 (代表長刺最深蹲極限處)
    max_dist = -1.0
    max_frame_idx = -1
    for idx, kps in enumerate(all_frames_kps):
        fa = get_keypoint(kps, front_ankle_idx)
        ba = get_keypoint(kps, back_ankle_idx)
        d  = euclidean_dist(fa, ba)
        if d is not None and d > max_dist:
            max_dist = d
            max_frame_idx = idx

    # 門檻基準腳踝距離：若有全域初始值則使用；否則用 1.0 (防止除以零)
    base_dist = global_init_dist if (global_init_dist and global_init_dist > 0) else 1.0

    lunge_start_frame = n
    if max_frame_idx != -1:
        # 從拉最開的幀往回尋找，第一個間距 <= base_dist * 1.15 的幀，就是長刺的起點！
        for idx in range(max_frame_idx, -1, -1):
            kps = all_frames_kps[idx]
            fa = get_keypoint(kps, front_ankle_idx)
            ba = get_keypoint(kps, back_ankle_idx)
            d  = euclidean_dist(fa, ba)
            if d is not None and d <= base_dist * 1.25:
                lunge_start_frame = idx
                break
        else:
            lunge_start_frame = max(0, max_frame_idx - 3) # 回退預設幀數

    # 確保前後段都有足夠幀數以防退化（最少前置 3 幀，後置 3 幀）
    lunge_start_frame = max(3, min(lunge_start_frame, n - 3))

    front_half = all_frames_kps[:lunge_start_frame]
    back_half  = all_frames_kps[lunge_start_frame:]

    if not front_half or not back_half:
        return False

    # 計算平均軀幹長度 (avg_torso) 用於進行尺度無關化補正
    torso_lengths = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        mid_s = midpoint(ls, rs)
        mid_h = midpoint(lh, rh)
        t_len = euclidean_dist(mid_s, mid_h)
        if t_len is not None:
            torso_lengths.append(t_len)
    avg_torso = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 150.0

    # ── (B) 前段實質位移檢驗（排除原地彈跳步 Bounce 與蓄力深蹲晃動）─────
    attack_dir = _get_attack_direction(all_frames_kps, sword_hand, person_id)
    cx_list = [_get_center_of_mass_x(kps) for kps in front_half]
    cx_list = [v for v in cx_list if v is not None]
    
    print(f"      [DEBUG ADVANCE-LUNGE-COORDINATES] front_half 總幀數: {len(front_half)}, 有效重心數: {len(cx_list)}")
    if len(cx_list) >= 2:
        front_net_move = (cx_list[-1] - cx_list[0]) * attack_dir
        front_move_ratio = front_net_move / avg_torso
        print(f"      [DEBUG ADVANCE-LUNGE-COORDINATES] front_half cx_list: {[round(x, 2) for x in cx_list]}")
        print(f"      [DEBUG ADVANCE-LUNGE-COORDINATES] 計算得到的 front_net_move: {front_net_move:.2f} px (Ratio: {front_move_ratio:.3f})")
    else:
        print(f"      [DEBUG ADVANCE-LUNGE] 有效重心數不足 2 ({len(cx_list)})，駁回判定。")
        return False

    # ── (C) 後腳跟進檢驗（用後腳踝位移區分純長刺與前警長刺） ─────────────────
    # 在真正的「前進長刺」中，前半段必須完成前進步，這意味著後腳踝必須向前移動跟進。
    # 如果是「純長刺（原地防守長刺）」，前半段後腳踝是釘死在原地不動的（位移接近 0）。
    la_xs = []
    ra_xs = []
    for kps in front_half:
        la = get_keypoint(kps, KP_LEFT_ANKLE)
        ra = get_keypoint(kps, KP_RIGHT_ANKLE)
        if la: la_xs.append(la[0])
        if ra: ra_xs.append(ra[0])
        
    print(f"      [DEBUG ANKLES-COORDINATES] Front Ankle Idx: {front_ankle_idx}, Back Ankle Idx: {back_ankle_idx}")
    print(f"      [DEBUG ANKLES-COORDINATES] Left Ankle X: {[round(x, 1) for x in la_xs]}")
    print(f"      [DEBUG ANKLES-COORDINATES] Right Ankle X: {[round(x, 1) for x in ra_xs]}")

    ba_list = []
    for kps in front_half:
        ba = get_keypoint(kps, back_ankle_idx)
        if ba is not None:
            ba_list.append(ba[0])
            
    if len(ba_list) >= 2:
        back_ankle_move_front = (ba_list[-1] - ba_list[0]) * attack_dir
        back_ankle_move_ratio = back_ankle_move_front / avg_torso
        print(f"      [DEBUG ADVANCE-LUNGE-BACK-ANKLE] 前段後腳踝位移: {back_ankle_move_front:.2f} px (Ratio: {back_ankle_move_ratio:.3f})")
        
        # 物理 Facts 去耦合動態門檻：
        # 如果後腳跟進幅度極大 (Ratio >= 0.120)，說明步法跟進事實確立，放寬前半段重心移動的要求至 -0.15
        # 如果後腳跟進幅度中等 (Ratio >= 0.030)，放寬前半段重心移動的要求至 -0.05
        # 否則，前半段重心必須前推至少 0.050 以上
        if back_ankle_move_ratio >= 0.120:
            effective_front_move_thr = -0.150
        elif back_ankle_move_ratio >= 0.030:
            effective_front_move_thr = -0.050
        else:
            effective_front_move_thr = 0.050

        if front_move_ratio < effective_front_move_thr:
            print(f"      [DEBUG ADVANCE-LUNGE] 前段重心位移比 {front_move_ratio:.3f} 未達動態門檻 {effective_front_move_thr:.3f}。駁回判定。")
            return False

        # 物理 Facts 去耦合檢驗：若前半段重心有極大前移（Ratio > 0.26），但後腳踝無充分跟進（Ratio < 0.08），
        # 說明此時選手僅為原地的長刺發力，而非步法跟進的前進長刺，應判定為純長刺 (3)。
        if front_move_ratio > 0.26 and back_ankle_move_ratio < 0.08:
            print(f"      [DEBUG ADVANCE-LUNGE-BACK-ANKLE] 駁回：大跨步前伸但後腳踝無充分跟進 ({back_ankle_move_ratio:.3f} < 0.08)，判定為純長刺。")
            return False
            
        # 門檻微調為最優的 0.030 比例，確保微小前置步法能被正確追蹤
        if back_ankle_move_ratio < 0.030:
            print(f"      [DEBUG ADVANCE-LUNGE-BACK-ANKLE] 駁回：前段後腳踝無充分跟進位移 ({back_ankle_move_ratio:.3f} < 0.030)，判定為純長刺，非前進長刺。")
            return False
    else:
        print(f"      [DEBUG ADVANCE-LUNGE-BACK-ANKLE] 前段後腳踝有效點數不足 2，駁回判定。")
        return False

    # ── (C-2) 新增：前置步法階段腳踝間距拉伸變化防火牆 ────────────────────
    # 避免原地雜訊、相機補正殘留干擾判定，要求前段必須有實質的跨步行為
    dists = []
    for kps in front_half:
        fa = get_keypoint(kps, front_ankle_idx)
        ba = get_keypoint(kps, back_ankle_idx)
        d = euclidean_dist(fa, ba)
        if d is not None:
            dists.append(d)
    if dists:
        ankle_range_ratio = (max(dists) - min(dists)) / avg_torso
        print(f"      [DEBUG ADVANCE-LUNGE-ANKLE-RANGE] 前置步法腳踝間距變化比: {ankle_range_ratio:.3f}")
        if ankle_range_ratio < 0.240:
            print(f"      [DEBUG ADVANCE-LUNGE-ANKLE-RANGE] 駁回：前半段腳踝間距變化比 {ankle_range_ratio:.3f} < 0.240，非實質前進步法。")
            return False

    # ── (D) 軀幹自適應尺度去耦合檢驗（對抗 early lunge 誤判為 advance lunge） ─────
    # 物理 Facts：在一個真正的「前進長刺」中，如果前半段發生了極大的腿部拉伸（Stretch/Torso >= 0.70），
    # 說明選手在大跨步發力；在此情境下，選手的後腳必須在全片中有實質的前推跟進完成步法（TotalBA/Torso >= 0.70）。
    # 如果 Stretch/Torso >= 0.70 且 TotalBA/Torso < 0.70，說明選手在開頭就發動了純長刺，後腳從未進行實質的步法前進，應駁回前進長刺判定，使其正確歸類為純長刺 (3: 長刺)。
    
    # 計算平均軀幹長度 (avg_torso)
    torso_lengths = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        mid_s = midpoint(ls, rs)
        mid_h = midpoint(lh, rh)
        t_len = euclidean_dist(mid_s, mid_h)
        if t_len is not None:
            torso_lengths.append(t_len)
    avg_torso = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 150.0

    # 1. 計算前半段的腿部拉伸 (front_net_stretch)
    distances = []
    for kps in front_half:
        fa = get_keypoint(kps, front_ankle_idx)
        ba = get_keypoint(kps, back_ankle_idx)
        d  = euclidean_dist(fa, ba)
        if d is not None:
            distances.append(d)
    
    if distances:
        front_net_stretch = distances[-1] - distances[0]
        stretch_torso_ratio = front_net_stretch / avg_torso
    else:
        stretch_torso_ratio = 0.0

    # 2. 計算全片後腳踝的最大前移量 (total_back_ankle_move)
    ba_list_all = [get_keypoint(kps, back_ankle_idx)[0] for kps in all_frames_kps if get_keypoint(kps, back_ankle_idx) is not None]
    if len(ba_list_all) >= 2:
        diffs = [(x - ba_list_all[0]) * attack_dir for x in ba_list_all]
        total_back_ankle_move = max(diffs)
        total_ba_torso_ratio = total_back_ankle_move / avg_torso
    else:
        total_ba_torso_ratio = 0.0

    print(f"      [DEBUG ADVANCE-LUNGE-TORSO-SCALE] Stretch/Torso: {stretch_torso_ratio:.3f}, TotalBA/Torso: {total_ba_torso_ratio:.3f}")
    dynamic_ba_threshold = 0.25 if len(front_half) >= 0.40 * len(all_frames_kps) else 0.50
    if (stretch_torso_ratio >= 0.70 and total_ba_torso_ratio < 0.70) or \
       (stretch_torso_ratio >= 0.40 and back_ankle_move_ratio < dynamic_ba_threshold):
        print(f"      [DEBUG ADVANCE-LUNGE-TORSO-SCALE] 駁回：前半段大跨步拉伸但全片後腳踝跟進不足，判定為純長刺。")
        return False

    # 後段：長刺特徵（傳入全域初始距離，鎖定分母）
    is_lunge   = _check_lunge(back_half, sword_hand, front_ankle_idx, back_ankle_idx, 
                              external_init_dist=global_init_dist, is_sliced_lunge=True)

    print(f"      [DEBUG ADVANCE-LUNGE-FINAL] 後半段長刺: {is_lunge}")

    return is_lunge



def classify_lower(all_frames_kps: list, sword_hand: str,
                   person_id=None, other_kps_seq=None) -> int:
    """
    判定下半身標籤（依優先順序判斷，最先符合者優先）。

    標籤定義：
        0 → 無
        1 → 前進
        2 → 後退
        3 → 長刺
        4 → 飛刺
        5 → 前進長刺
    """
    front_ankle_idx, back_ankle_idx = _determine_front_back_ankle(
        all_frames_kps, sword_hand, person_id
    )

    # 1. 估計平均軀幹長度
    torso_lengths = []
    for kps in all_frames_kps:
        ls = get_keypoint(kps, KP_LEFT_SHOULDER)
        rs = get_keypoint(kps, KP_RIGHT_SHOULDER)
        lh = get_keypoint(kps, KP_LEFT_HIP)
        rh = get_keypoint(kps, KP_RIGHT_HIP)
        mid_s = midpoint(ls, rs)
        mid_h = midpoint(lh, rh)
        t_len = euclidean_dist(mid_s, mid_h)
        if t_len is not None:
            torso_lengths.append(t_len)
    avg_torso = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 150.0

    # 2. 計算總重心淨位移比 (net_move_ratio)
    cx_list = [_get_center_of_mass_x(kps) for kps in all_frames_kps]
    cx_list = [v for v in cx_list if v is not None]
    net_move_ratio = 0.0
    if len(cx_list) >= 2:
        attack_dir = _get_attack_direction(all_frames_kps, sword_hand, person_id)
        mid = max(1, len(cx_list) // 2)
        avg_front = sum(cx_list[:mid]) / mid
        avg_back  = sum(cx_list[mid:]) / max(1, len(cx_list) - mid)
        net_move_ratio = ((avg_back - avg_front) * attack_dir) / avg_torso

    # 3. 計算雙人最低重心距離（用於近距離交鋒 CLOSE-QUARTERS-LOCK 護欄）
    min_dist = None
    if other_kps_seq:
        dists = []
        for kp_c, kp_o in zip(all_frames_kps, other_kps_seq):
            cx_c = _get_center_of_mass_x(kp_c)
            cx_o = _get_center_of_mass_x(kp_o)
            if cx_c is not None and cx_o is not None:
                dists.append(abs(cx_c - cx_o))
        if dists:
            min_dist = min(dists)

    # 幾何狀態判定
    is_adv_lunge = _check_advance_lunge(all_frames_kps, sword_hand, front_ankle_idx, back_ankle_idx, person_id)
    is_fleche = _check_fleche(all_frames_kps, sword_hand, front_ankle_idx, back_ankle_idx, person_id)
    is_pure_lunge = _check_lunge(all_frames_kps, sword_hand, front_ankle_idx, back_ankle_idx, person_id=person_id)

    # 終極判斷
    # (A) 優先判定飛刺 (Fleche) 因為它是極端的前進跨步
    if is_fleche:
        return 4
        
    # (B) 判斷前進弓步 (Advance-Lunge)
    if is_adv_lunge:
        return 5 if net_move_ratio >= 0.680 else 3

    # (C) 判定純長刺特徵
    if is_pure_lunge:
        return 3

    # (D) 前進
    if _check_advance(all_frames_kps, sword_hand, person_id):
        if min_dist is not None and min_dist < 1.10 * avg_torso:
            print(f"      [DEBUG CLOSE-QUARTERS-LOCK] 駁回前進：雙人距離 {min_dist/avg_torso:.3f} < 1.10")
            return 0
        return 1
    if _check_retreat(all_frames_kps, sword_hand, person_id):
        if min_dist is not None and min_dist < 1.10 * avg_torso: return 0
        return 2
    return 0

def load_frame_jsons(person_dir: str):
    """
    讀取單一 person_id 資料夾下所有 frame_*.json，按幀號排序後，
    從每個 JSON 中提取關鍵點陣列，回傳關鍵點序列。

    實際目錄結構：
        2d_detections/{video_name}/{person_id}/frame_*.json

    因為每個 JSON 已屬於特定人物，無需再篩選 person_id。
    每幀 JSON 支援以下格式（依序嘗試）：
        格式 A：直接是關鍵點陣列          [[x,y,conf], ...]
        格式 B：{"keypoints": [[x,y,conf], ...]}
        格式 C：{"instances": [{"keypoints": [...]}, ...]}
        格式 D：{"persons":   [{"keypoints": [...]}, ...]}

    Args:
        person_dir: 包含 frame_*.json 的 person_id 資料夾路徑

    Returns:
        list[list]：每個元素為一幀的關鍵點陣列（17×2 或 17×3）
    """
    pattern    = os.path.join(person_dir, "frame_*.json")
    json_files = sorted(glob.glob(pattern))

    kp_sequence = []

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [警告] 無法讀取 {os.path.basename(fpath)}：{e}")
            continue

        kps = None

        # ── 格式 A：JSON 根層直接是 list（關鍵點陣列）────────────────────
        if isinstance(data, list):
            kps = data

        # ── 格式 B：{"keypoints": [...]} ─────────────────────────────────
        elif isinstance(data, dict) and "keypoints" in data:
            kps = data["keypoints"]

        # ── 格式 C：{"instances": [{"keypoints": [...]}, ...]} ───────────
        elif isinstance(data, dict) and "instances" in data:
            instances = data["instances"]
            if instances and "keypoints" in instances[0]:
                kps = instances[0]["keypoints"]

        # ── 格式 D：{"persons": [{"keypoints": [...]}, ...]} ─────────────
        elif isinstance(data, dict) and "persons" in data:
            persons = data["persons"]
            if persons and "keypoints" in persons[0]:
                kps = persons[0]["keypoints"]

        if kps is not None:
            kp_sequence.append(kps)
        else:
            print(f"  [警告] {os.path.basename(fpath)} 無法解析關鍵點，已略過")

    return kp_sequence

# =============================================================================
# 單一人物標註函式
# =============================================================================

def label_person(video_name: str, person_id: int, person_dir: str,
                 camera_offsets: list = None):
    """
    讀取單一 person_id 資料夾的幀序列，執行幾何標註，
    並將結果寫入 temp_results/{video_name}_p{person_id}.json。

    Args:
        video_name     : 影片名稱（取自上層資料夾名稱）
        person_id      : 人物編號（取自 person_id 資料夾名稱）
        person_dir     : frame_*.json 所在的完整路徑
        camera_offsets : 由 estimate_camera_pan() 產出的鏡頭平移偏移列表，
                         若提供則在幾何判定前先修正骨架座標。
    """
    # ── 讀取並解析幀序列 ──────────────────────────────────────────────────
    kp_sequence = load_frame_jsons(person_dir)

    if not kp_sequence:
        print(f"    [跳過] 找不到任何可解析的 frame_*.json")
        return

    if len(kp_sequence) < 3:
        print(f"    [跳過] 有效幀數不足 3 幀（僅 {len(kp_sequence)} 幀）")
        return

    print(f"    讀取 {len(kp_sequence)} 幀 → 開始幾何判定")

    # ── 鏡頭平移補正（用幀號精確對應還移量）────────────────────────
    if camera_offsets:
        frame_numbers = _get_frame_numbers(person_dir)
        # 由於 YOLO 輸出 JSON 採用 0-based 幀號命名 (frame_000000000000)，此時 fn 本身即為 0-based 索引，直接對應快取偏移序列
        frame_specific_offsets = []
        for fn in frame_numbers:
            idx = fn   # 直接作為快取 offsets 序列索引
            if 0 <= idx < len(camera_offsets):
                frame_specific_offsets.append(camera_offsets[idx])
            else:
                frame_specific_offsets.append(camera_offsets[-1] if camera_offsets else (0.0, 0.0))
        kp_sequence = apply_camera_correction(kp_sequence, frame_specific_offsets)
        max_dx = max((abs(dx) for dx, dy in frame_specific_offsets), default=0.0)
        print(f"    [Camera Pan] 已套用鏡頭補正（最大 X 偏移: {max_dx:.1f}px）")

    # ── 讀取對手骨架序列（用於近距離交鋒 CLOSE-QUARTERS-LOCK） ─────────────────
    other_kps_seq = None
    other_pid = 3 - person_id
    other_dir = os.path.join(os.path.dirname(person_dir), str(other_pid))
    if os.path.exists(other_dir):
        other_kps_raw = load_frame_jsons(other_dir)
        if other_kps_raw:
            if camera_offsets:
                other_specific_offsets = []
                other_frame_numbers = _get_frame_numbers(other_dir)
                for fn in other_frame_numbers:
                    idx = fn
                    if 0 <= idx < len(camera_offsets):
                        other_specific_offsets.append(camera_offsets[idx])
                    else:
                        other_specific_offsets.append(camera_offsets[-1] if camera_offsets else (0.0, 0.0))
                other_kps_seq = apply_camera_correction(other_kps_raw, other_specific_offsets)
            else:
                other_kps_seq = other_kps_raw

    # ── 幾何判定 ──────────────────────────────────────────────────────────
    sword_hand  = determine_sword_hand(kp_sequence, person_id)
    label_lower = classify_lower(kp_sequence, sword_hand, person_id, other_kps_seq)   # 傳入對手數據
    
            
    label_upper = classify_upper(kp_sequence, sword_hand, person_id, label_lower)

    # ── 輸出小 JSON ───────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = {
        "video_name":  video_name,
        "person_id":   person_id,
        "label_upper": label_upper,
        "label_lower": label_lower
    }

    out_fname = f"{video_name}_p{person_id}.json"
    out_path  = os.path.join(OUTPUT_DIR, out_fname)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"    [OK] 持劍手={sword_hand} | Upper={label_upper} | "
          f"Lower={label_lower} -> {out_fname}")


# =============================================================================
# 主程式入口
# =============================================================================

def main():
    """
    走訪 DETECTION_DIR 下的兩層子資料夾結構：

        DETECTION_DIR/
        └── {video_name}/          ← 第一層：影片名稱資料夾
            └── {person_id}/       ← 第二層：人物編號資料夾（如 "1", "2"）
                └── frame_*.json  ← 幀資料（每幀一個 JSON）

    對每個 (video_name, person_id) 組合執行幾何標註，
    結果輸出至 temp_results/{video_name}_p{person_id}.json。
    """
    print("=" * 60)
    print("  西洋劍動作自動標註管線 — 啟動")
    print(f"  偵測資料夾：{DETECTION_DIR}")
    print(f"  結果輸出至：{OUTPUT_DIR}")
    print("=" * 60)

    if not os.path.isdir(DETECTION_DIR):
        print(f"\n[錯誤] 找不到偵測資料夾：{DETECTION_DIR}")
        print("請確認 ViTPose 輸出路徑是否正確。")
        return

    # ── 載入鏡頭偏移快取 ───────────────────────────────────────────────────
    offsets_cache = {}
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_offsets.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                offsets_cache = json.load(f)
            print(f"  [Camera Pan] 成功載入快取 offsets_cache (共 {len(offsets_cache)} 部影片)")
        except Exception as e:
            print(f"  [Camera Pan] 無法讀取快取檔案：{e}")

    # ── 第一層：影片名稱資料夾 ────────────────────────────────────────────
    video_dirs = sorted([
        d for d in os.listdir(DETECTION_DIR)
        if os.path.isdir(os.path.join(DETECTION_DIR, d))
    ])

    if not video_dirs:
        print("\n[警告] 偵測資料夾內找不到任何影片子資料夾。")
        return

    print(f"\n  找到 {len(video_dirs)} 個影片資料夾")

    total_persons = 0

    for video_name in video_dirs:
        video_path = os.path.join(DETECTION_DIR, video_name)

        # ── 第二層：person_id 資料夾（先過濾再統計）─────────────────────
        # 只保留「確實是子資料夾、且名稱為純整數」的項目（過濾 img 等雜項）
        person_dirs = sorted([
            p for p in os.listdir(video_path)
            if os.path.isdir(os.path.join(video_path, p)) and p.isdigit()
        ])

        if not person_dirs:
            print(f"\n  [跳過] {video_name}：底下找不到任何 person_id 子資料夾")
            continue

        print(f"\n{'='*60}")
        print(f"  影片：{video_name}  |  偵測到 {len(person_dirs)} 人：{person_dirs}")

        # ── 鏡頭平移補正：每部影片只計算一次（所有人共用）─────────────
        camera_offsets = []
        if CAMERA_CORRECTION_ENABLED:
            if video_name in offsets_cache:
                camera_offsets = offsets_cache[video_name]
                print(f"  [Camera Pan] 從快取中讀取 '{video_name}'，共 {len(camera_offsets)} 幀偏移")
            elif _OPENCV_AVAILABLE:
                src_video_path = _find_video_file(video_name)   # 注意：用 src_video_path，不覆蓋外層 video_path
                if src_video_path:
                    print(f"  [Camera Pan] 快取未命中，正在從原始影片估計鏡頭平移…")
                    print(f"               影片：{src_video_path}")
                    camera_offsets = estimate_camera_pan_from_video(video_name, video_path)
                    print(f"  [Camera Pan] 完成，共 {len(camera_offsets)} 幀偏移")
                else:
                    print(f"  [Camera Pan] 在 VIDEO_DIR 找不到影片 '{video_name}'.*，跳過補正。")
                    print(f"               搜尋路徑: {VIDEO_DIR}")

        for pid_str in person_dirs:
            person_id  = int(pid_str)   # 已確保為純整數，直接轉換
            person_dir = os.path.join(video_path, pid_str)   # video_path = DETECTION_DIR/{video_name}

            print(f"\n  ├─ person_id={person_id}  路徑：{person_dir}")
            label_person(video_name, person_id, person_dir, camera_offsets)
            total_persons += 1

    print("\n" + "=" * 60)
    print(f"  全部標註完成！共處理 {total_persons} 個 (影片, 人物) 組合")
    print(f"  結果已儲存至：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
