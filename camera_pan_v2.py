import cv2, os, json, numpy as np

def estimate_camera_pan_from_video(video_name: str, video_dir: str) -> list:
    video_path = os.path.join(video_dir, video_name + ".mp4")
    if not os.path.exists(video_path):
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
                        px, py = pt.ravel()
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
