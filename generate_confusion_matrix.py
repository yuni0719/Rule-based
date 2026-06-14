import json
import os
import sys

# 強制 Windows 終端機以 UTF-8 輸出中文，防止出現亂碼 (Mojibake)
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

def get_display_width(s):
    width = 0
    for char in str(s):
        if ord(char) > 127:
            width += 2
        else:
            width += 1
    return width

def pad_string(s, width, align="left"):
    s = str(s)
    current_width = get_display_width(s)
    pad_len = max(0, width - current_width)
    if align == "left":
        return s + " " * pad_len
    elif align == "right":
        return " " * pad_len + s
    else:  # center
        left_pad = pad_len // 2
        right_pad = pad_len - left_pad
        return " " * left_pad + s + " " * right_pad

def print_confusion_matrix(gt_labels, pred_labels, title="Confusion Matrix", label_names=None):
    classes = sorted(list(set(gt_labels) | set(pred_labels)))
    class_map = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    
    matrix = [[0] * n for _ in range(n)]
    for gt, pred in zip(gt_labels, pred_labels):
        if gt in class_map and pred in class_map:
            matrix[class_map[gt]][class_map[pred]] += 1
            
    # Calculate Precision, Recall, F1
    report = {}
    for c in classes:
        idx = class_map[c]
        actual_total = sum(matrix[idx])
        pred_total = sum(matrix[r][idx] for r in range(n))
        true_positive = matrix[idx][idx]
        
        precision = true_positive / pred_total if pred_total > 0 else 0.0
        recall = true_positive / actual_total if actual_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        report[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_total
        }
        
    correct = sum(matrix[i][i] for i in range(n))
    total = sum(sum(row) for row in matrix)
    accuracy = correct / total if total > 0 else 0.0

    col1_width = 16
    col_width = 12
    matrix_width = col1_width + n * col_width + n + 1
    
    h_class = pad_string("動作類別", 12, "center")
    h_prec  = pad_string("精確率 (Precision)", 18, "center")
    h_rec   = pad_string("召回率 (Recall)", 18, "center")
    h_f1    = pad_string("F1 值 (F1-Score)", 18, "center")
    h_supp  = pad_string("樣本數 (Support)", 15, "center")
    report_header = f" {h_class} | {h_prec} | {h_rec} | {h_f1} | {h_supp}"
    
    report_width = get_display_width(report_header)
    
    separator_matrix = "-" * matrix_width
    separator_report = "-" * report_width
    
    # --- Matrix Lines ---
    matrix_lines = []
    header_parts = []
    header_parts.append(pad_string(" GT \\ PRED", col1_width, "left"))
    for c in classes:
        name = label_names.get(c, str(c)) if label_names else str(c)
        header_parts.append(pad_string(f" P={name}", col_width, "center"))
    
    header_str = "|".join(header_parts) + "|"
    matrix_lines.append(header_str)
    matrix_lines.append(separator_matrix)
    
    for c in classes:
        idx = class_map[c]
        row_parts = []
        name = label_names.get(c, str(c)) if label_names else str(c)
        row_parts.append(pad_string(f"  GT={name}", col1_width, "left"))
        for c2 in classes:
            val = matrix[idx][class_map[c2]]
            row_parts.append(pad_string(val, col_width, "center"))
        
        row_str = "|".join(row_parts) + "|"
        matrix_lines.append(row_str)
        
    matrix_lines.append(separator_matrix)
    
    # --- Report Lines ---
    report_lines = []
    report_title = " 分類指標報告 (Classification Report) "
    pad_len = max(0, report_width - get_display_width(report_title))
    left_pad = pad_len // 2
    right_pad = pad_len - left_pad
    report_lines.append("-" * left_pad + report_title + "-" * right_pad)
    
    report_lines.append(report_header)
    report_lines.append(separator_report)
    
    for c in classes:
        rep = report[c]
        name = label_names.get(c, str(c)) if label_names else str(c)
        c_str    = pad_string(name, 12, "center")
        prec_str = pad_string(f"{rep['precision']*100:.1f}%", 18, "center")
        rec_str  = pad_string(f"{rep['recall']*100:.1f}%", 18, "center")
        f1_str   = pad_string(f"{rep['f1']*100:.1f}%", 18, "center")
        supp_str = pad_string(rep['support'], 15, "center")
        report_lines.append(f" {c_str} | {prec_str} | {rec_str} | {f1_str} | {supp_str}")
    report_lines.append(separator_report)
    
    macro_p = sum(r["precision"] for r in report.values()) / len(report)
    macro_r = sum(r["recall"] for r in report.values()) / len(report)
    macro_f1 = sum(r["f1"] for r in report.values()) / len(report)
    
    avg_str  = pad_string("平均值 (Macro)", 12, "center")
    avg_p    = pad_string(f"{macro_p*100:.1f}%", 18, "center")
    avg_r    = pad_string(f"{macro_r*100:.1f}%", 18, "center")
    avg_f1   = pad_string(f"{macro_f1*100:.1f}%", 18, "center")
    total_support = sum(r["support"] for r in report.values())
    avg_supp = pad_string(total_support, 15, "center")
    report_lines.append(f" {avg_str} | {avg_p} | {avg_r} | {avg_f1} | {avg_supp}")
    report_lines.append("=" * report_width)
    
    # --- Print Vertically ---
    print(f"\n【 {title} 】")
    print(f"整體正確率 (Accuracy): {accuracy*100:.1f}%\n")
    
    for ml in matrix_lines:
        print(ml)
        
    print()
    
    for rl in report_lines:
        print(rl)

def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(workspace, "ground_truth.json")
    pred_path = os.path.join(workspace, "final_labels.json")
    
    if not os.path.exists(gt_path):
        print(f"[錯誤] 找不到 ground_truth.json，請確認檔案在：{gt_path}")
        return
    if not os.path.exists(pred_path):
        print(f"[錯誤] 找不到 final_labels.json，請確認檔案在：{pred_path}")
        return
        
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
        
    gt_dict = {(r["video_name"], r["person_id"]): r for r in gt_data}
    pred_dict = {(r["video_name"], r["person_id"]): r for r in pred_data}
    
    gt_lower = []
    pred_lower = []
    gt_upper = []
    pred_upper = []
    
    for key, gt in gt_dict.items():
        pred = pred_dict.get(key)
        if not pred: continue
        
        gt_lower.append(gt["label_lower"])
        pred_lower.append(pred["label_lower"])
        
        gt_upper.append(gt["label_upper"])
        pred_upper.append(pred["label_upper"])
        
    upper_names = {0: "無", 1: "直刺"}
    lower_names = {0: "無", 1: "前進", 2: "後退", 3: "長刺", 4: "飛刺", 5: "前進長刺"}
    
    print_confusion_matrix(gt_upper, pred_upper, "Upper Body (上半身)", label_names=upper_names)
    print()
    print_confusion_matrix(gt_lower, pred_lower, "Lower Body (下半身)", label_names=lower_names)

if __name__ == "__main__":
    main()
