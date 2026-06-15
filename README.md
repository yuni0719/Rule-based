# Rule-based 自動標註系統

## 📥 需要下載的檔案
直接 clone 在跟 Vitpose 同一層
1. `auto_labeler.py` (主程式)
2. `visualize_labels.py` (畫骨架影片用的)
3. `merge_labels.py` (合併標籤工具)
## 🚀 怎麼執行
**1. 安裝套件**
pip install numpy opencv-python Pillow
**2. 跑標註程式**
把骨架 JSON 放好後，直接跑主程式就會產出標籤：
python auto_labeler.py
**3. 想看標註完的影片**
python visualize_labels.py
(跑完後影片會自動存在 labeled_videos/ 資料夾裡面)
