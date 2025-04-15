import os
import re
import pandas as pd

# ===================== 1. 讀取 SE_events.csv，建立 SE 資訊與 PSI =====================
csv_path = "./SE_events_high.csv"
csv_data = pd.read_csv(csv_path)

# se_info[(gene, 5ss, 3ss)] = PSI
se_info = {}
for _, row in csv_data.iterrows():
    g = row["gene"]
    s5 = row["5ss"]
    s3 = row["3ss"]
    psi_val = row["PSI"]
    # Stash both classification info ("SE") and PSI in one place
    se_info[(g, s5, s3)] = psi_val

# ===================== 2. 掃描 .txt 檔案，解析「Annotated 5SS/3SS」並抓取 prob =====================
txt_folder = "./t_allParse_txt/t_allParse_exon"

records = []

# Regex patterns
gene_regex     = re.compile(r"Gene\s*=\s*(\S+)")
index_regex    = re.compile(r"index\s*=\s*(\d+)")
ann_5ss_regex  = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
ann_3ss_regex  = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
# Lines of the form "3SS, 5SS, prob" (allowing possible spacing)
exon_line_regex = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.eE+\-]+)\s*$")

for fname in os.listdir(txt_folder):
    if not fname.endswith(".txt"):
        continue
    
    filepath = os.path.join(txt_folder, fname)
    with open(filepath, "r") as f:
        content = f.read()
    
    # --- Extract gene and file_index ---
    gene_match = gene_regex.search(content)
    gene_id = gene_match.group(1) if gene_match else None
    
    index_match = index_regex.search(content)
    file_index = int(index_match.group(1)) if index_match else None

    # --- Extract Annotated 5SS and 3SS ---
    ann_5_match = ann_5ss_regex.search(content)
    ann_3_match = ann_3ss_regex.search(content)
    
    # Parse the lines "3SS, 5SS, prob" into a dictionary
    predicted_probs = {}
    for line in content.splitlines():
        m = exon_line_regex.match(line)
        if m:
            three_val = int(m.group(1))  # 3SS
            five_val  = int(m.group(2))  # 5SS
            prob_val  = float(m.group(3))
            predicted_probs[(three_val, five_val)] = prob_val
    
    has_exons = False
    
    if gene_id and ann_5_match and ann_3_match:
        # Convert annotated 5SS/3SS to integer lists
        annotated_5_list = [int(x) for x in re.findall(r"\d+", ann_5_match.group(1))]
        annotated_3_list = [int(x) for x in re.findall(r"\d+", ann_3_match.group(1))]
        
        # Generate exon pairs (3SS[i], 5SS[i+1])
        exon_pairs = []
        pair_count = min(len(annotated_3_list), len(annotated_5_list)) - 1
        for i in range(pair_count):
            three_ss = annotated_3_list[i]
            five_ss  = annotated_5_list[i+1]
            exon_pairs.append((three_ss, five_ss))
        
        if exon_pairs:
            has_exons = True
            for (three_val, five_val) in exon_pairs:
                # Check if (gene, 5ss, 3ss) is in se_info
                # i.e. (gene_id, five_val, three_val)
                if (gene_id, five_val, three_val) in se_info:
                    classification = "SE"
                    psi_val = se_info[(gene_id, five_val, three_val)]
                else:
                    classification = "non-SE"
                    psi_val = None
                
                # Look up probability
                prob = predicted_probs.get((three_val, five_val), None)
                
                records.append({
                    "gene": gene_id,
                    "file_index": file_index,
                    "3ss": three_val,
                    "5ss": five_val,
                    "prob": prob,
                    "PSI": psi_val,
                    "classification": classification
                })
    
    # If no valid exons found or no gene data, record a line with classification="NA"
    if not has_exons:
        records.append({
            "gene": gene_id if gene_id else "NA",
            "file_index": file_index,
            "3ss": None,
            "5ss": None,
            "prob": None,
            "PSI": None,
            "classification": "NA"
        })

# ===================== 3. 輸出 =====================
df = pd.DataFrame(records)

if df.empty:
    print("No data found.")
else:
    # Sort by file_index
    df = df.sort_values(by=["file_index"], ascending=True).reset_index(drop=True)
    
    output_dir = "./SE_result"
    os.makedirs(output_dir, exist_ok=True)
    
    # (A) 詳細資訊
    details_path = os.path.join(output_dir, "annotated_exon_details.csv")
    df.to_csv(details_path, index=False)
    print(f"Detailed exon data (with PSI) saved to: {details_path}")
    
    # (B) 統計 (含 avg prob)
    # Excluding classification="NA" or prob=None
    valid_df = df[(df["classification"] != "NA") & (df["prob"].notnull())]
    
    summary = (
        valid_df.groupby("classification")
        .agg(
            count=("prob", "count"),
            avg_prob=("prob", "mean")
            # You could add an avg_PSI if you want:
            # avg_PSI=("PSI", "mean") 
        )
        .reset_index()
    )
    
    summary_path = os.path.join(output_dir, "annotated_exon_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Exon summary saved to: {summary_path}")
    
    print("✅ Done!")
