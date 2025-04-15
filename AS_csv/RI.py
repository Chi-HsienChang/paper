import os
import re
import pandas as pd

# ========== 1. 讀取 CSV，建立「RI」(gene, 5ss, 3ss) -> PSI 對應 ==========

csv_path = "./RI_events_high.csv"
csv_data = pd.read_csv(csv_path)

# 建立一個字典: (gene, 5ss, 3ss) -> PSI
ri_info = {}
for _, row in csv_data.iterrows():
    gene = row["gene"]
    pos_5 = row["5ss"]
    pos_3 = row["3ss"]
    psi_val = row["PSI"]  # 假設 CSV 裡有 PSI 欄位
    ri_info[(gene, pos_5, pos_3)] = psi_val

# 也可以建立一個 set, 用於快速判斷 RI or not
ri_pairs = set(ri_info.keys())

# ========== 2. 掃描所有 .txt 檔案，記錄「只考慮 Annotated intron」+ 沒有 intron 也保留 file_index ==========

txt_folder = "./t_allParse_txt/t_allParse_intron"
records = []

# Regex helpers
gene_regex = re.compile(r"Gene\s*=\s*(\S+)")
index_regex = re.compile(r"index\s*=\s*(\d+)")
ann_5ss_regex = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
ann_3ss_regex = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
intron_line_regex = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.eE+\-]+)\s*$")

for fname in os.listdir(txt_folder):
    if not fname.endswith(".txt"):
        continue
    
    filepath = os.path.join(txt_folder, fname)
    with open(filepath, "r") as f:
        content = f.read()
    
    # 取得 Gene ID
    gene_match = gene_regex.search(content)
    gene_id = gene_match.group(1) if gene_match else None
    
    # 取得 file_index
    index_match = index_regex.search(content)
    file_index = int(index_match.group(1)) if index_match else None
    
    # 抓取 Annotated 5SS / 3SS
    ann_5_match = ann_5ss_regex.search(content)
    ann_3_match = ann_3ss_regex.search(content)
    
    # 如果這個檔案沒有 Annotated 5SS 或 3SS，或沒有 gene
    # 我們依然想保留 file_index，所以先建一個「空的」flag
    has_annotated_introns = False
    
    if gene_id and ann_5_match and ann_3_match:
        # 轉成整數列表
        ann_5_list = [int(x) for x in re.findall(r"\d+", ann_5_match.group(1))]
        ann_3_list = [int(x) for x in re.findall(r"\d+", ann_3_match.group(1))]
        
        # 用 zip 把位置配對成「annotated intron」(5ss, 3ss)
        annotated_intron_pairs = list(zip(ann_5_list, ann_3_list))
        
        # 搜尋檔案中所有 "5SS, 3SS, prob" 行
        predicted_probs = {}
        for line in content.splitlines():
            m = intron_line_regex.match(line)
            if m:
                p5 = int(m.group(1))
                p3 = int(m.group(2))
                prob = float(m.group(3))
                predicted_probs[(p5, p3)] = prob
        
        if annotated_intron_pairs:
            has_annotated_introns = True
            
            for (p5, p3) in annotated_intron_pairs:
                # 判斷是否為 RI, 同時取出 PSI
                if (gene_id, p5, p3) in ri_pairs:
                    classification = "RI"
                    psi_val = ri_info[(gene_id, p5, p3)]
                else:
                    classification = "non-RI"
                    psi_val = None
                
                # 找該 intron pair 的預測機率 (如果沒有匹配到就為 None)
                prob = predicted_probs.get((p5, p3), None)
                
                records.append({
                    "gene": gene_id,
                    "file_index": file_index,
                    "5ss": p5,
                    "3ss": p3,
                    "prob": prob,
                    "PSI": psi_val,    # <-- 新增 PSI 欄位
                    "classification": classification
                })
    
    # 如果這個檔案最終沒有任何 annotated intron (或沒有 gene/annotated info)
    if not has_annotated_introns:
        records.append({
            "gene": gene_id if gene_id else "NA",
            "file_index": file_index,
            "5ss": None,
            "3ss": None,
            "prob": None,
            "PSI": None,  # 同樣保留 PSI 欄位
            "classification": "NA"  # 或 "no-annotated-intron"
        })

# ========== 3. 整理 & 依 file_index 排序後輸出所有檔案的結果 ==========

df = pd.DataFrame(records)

if df.empty:
    print("沒有任何可用的資料。")
else:
    # 依 file_index 排序 (ascending=True)
    df = df.sort_values(by=["file_index"], ascending=True).reset_index(drop=True)

    # 建立輸出資料夾
    os.makedirs("./RI_result", exist_ok=True)
    
    # (a) 詳細資料 - 每個檔案都有 file_index
    detailed_path = "./RI_result/annotated_intron_details.csv"
    df.to_csv(detailed_path, index=False)
    print(f"所有 TXT 文件的統合詳細檔已儲存: {detailed_path}")
    
    # (b) 統計 RI / non-RI 的平均機率 (排除 classification = 'NA' 或 prob is None)
    valid_df = df[(df["classification"] != "NA") & (df["prob"].notnull())]
    summary = (
        valid_df.groupby("classification")
        .agg(count=("prob", "count"), avg_prob=("prob", "mean"))
        .reset_index()
        
    )
    
    summary_path = "./RI_result/annotated_intron_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"統計檔已儲存: {summary_path}")
    
    print("✅ Done!")
