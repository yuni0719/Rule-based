# Rule-based 自動標註系統

直接 clone 在跟 Vitpose、Yolo 同一層
1. `auto_labeler.py` (主程式)
2. `visualize_labels.py` (標籤完的影片)
3. `merge_labels.py` (合併標籤工具)

## 執行流程
**1. 安裝套件**
pip install numpy opencv-python Pillow

**2. 跑標註程式**
直接跑主程式就會產出各影片標籤：
python auto_labeler.py
(輸出在  temp_results/ 資料夾裡)

**3. 合併標籤**
python merge_labels.py
(輸出名為 final_labels.json)

**4. 產出標註完的影片**
python visualize_labels.py
(跑完後影片會自動存在 labeled_videos/ 資料夾裡面)

**5. 看影片檢查標籤有沒有錯，有的話記得看好檔名、ID 去 final_labels.json 更改**
