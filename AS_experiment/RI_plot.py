import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

# -----------------------------#
# 1. 讀取 RI_events_high.csv，建立 RI 資訊字典
# 格式：(gene, 5ss, 3ss) -> PSI
# -----------------------------#
csv_path = "./data/RI_events_high.csv"  # 使用 RI 資料檔
ri_data = pd.read_csv(csv_path)

ri_info = {}
for _, row in ri_data.iterrows():
    gene = row["gene"]
    s5 = row["5ss"]
    s3 = row["3ss"]
    psi_val = row["PSI"]
    ri_info[(gene, s5, s3)] = psi_val

# -----------------------------#
# 2. 掃描 exon 解析 txt 檔案並產生 intron pair 記錄
#    產生 intron pair 以 (annotated_3[i], annotated_5[i]) 表示
# -----------------------------#
# 此處讀取 RI 版本的 txt 檔 (例如儲存在 "./t_allParse_txt/t_allParse_intron")
txt_folder = "./t_allParse_txt/t_allParse_intron"

records = []
# 定義正則表達式：抓取 gene、index、Annotated 5SS/3SS 與 "5SS, 3SS, prob" 格式行
gene_regex    = re.compile(r"Gene\s*=\s*(\S+)")
index_regex   = re.compile(r"index\s*=\s*(\d+)")
ann_5ss_regex = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
ann_3ss_regex = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
# 此處假設接下來的行格式為 "5SS, 3SS, prob"
exon_line_regex = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.eE+\-]+)\s*$")

for fname in os.listdir(txt_folder):
    if not fname.endswith(".txt"):
        continue
    filepath = os.path.join(txt_folder, fname)
    with open(filepath, "r") as f:
        content = f.read()
    
    # 取得 gene 與 file_index
    gene_match = gene_regex.search(content)
    gene_id = gene_match.group(1) if gene_match else None
    index_match = index_regex.search(content)
    file_index = int(index_match.group(1)) if index_match else None

    # 取得 Annotated 5SS 與 Annotated 3SS 的列表 (取中括號內所有數字)
    ann5_match = ann_5ss_regex.search(content)
    ann3_match = ann_3ss_regex.search(content)
    
    if gene_id and ann5_match and ann3_match:
        # 注意：此處定義 intron pair 為 (annotated_3[i], annotated_5[i])
        ann5 = [int(x) for x in re.findall(r"\d+", ann5_match.group(1))]
        ann3 = [int(x) for x in re.findall(r"\d+", ann3_match.group(1))]
        pair_count = min(len(ann3), len(ann5))  # 以相同 i 號進行配對
        for i in range(pair_count):
            intron_3 = ann3[i]   # 3' splice site（intron的 3'端）
            intron_5 = ann5[i]   # 5' splice site（intron的 5'端）
            # 依據 CSV 資料，字典 key 順序為 (gene, 5ss, 3ss)，因此查詢以 (gene, intron_5, intron_3)
            if (gene_id, intron_5, intron_3) in ri_info:
                classification = "RI"
                psi_val = ri_info[(gene_id, intron_5, intron_3)]
            else:
                classification = "non-RI"
                psi_val = None
            # 取得對應 intron 的預測 Exon Score
            prob = None
            for line in content.splitlines():
                line = line.strip()
                m = exon_line_regex.match(line)
                if m:
                    # 假設此處行格式為 "5SS, 3SS, prob"
                    five_val  = int(m.group(1))
                    three_val = int(m.group(2))
                    p         = float(m.group(3))
                    # 比對 intron pair：
                    if five_val == intron_5 and three_val == intron_3:
                        prob = p
                        break
            records.append({
                "gene": gene_id,
                "5ss": intron_5,
                "3ss": intron_3,
                "PSI": psi_val,
                "index": file_index,
                "prob": prob,
                "classification": classification
            })
    else:
        records.append({
            "gene": gene_id if gene_id else "NA",
            "5ss": None,
            "3ss": None,
            "PSI": None,
            "index": file_index,
            "prob": None,
            "classification": "NA"
        })

# -----------------------------#
# 3. 將所有記錄整理成 DataFrame
# -----------------------------#
df = pd.DataFrame(records)
df = df.sort_values(by=["index"]).reset_index(drop=True)

# -----------------------------#
# 4. 過濾：只考慮 allowed indexes 的 txt 檔
# -----------------------------#
allowed_indexes = {310, 318, 338, 435, 708, 724, 809, 816, 846, 953, 1109}
df_filtered = df[df["index"].isin(allowed_indexes)]

# -----------------------------#
# 5. 輸出結果：詳細檔與統計摘要
# -----------------------------#
output_dir = "./RI_result"
os.makedirs(output_dir, exist_ok=True)

details_path = os.path.join(output_dir, "annotated_intron_details_filtered.csv")
df_filtered.to_csv(details_path, index=False)
print(f"Filtered detailed intron data (only allowed indexes) saved to: {details_path}")

valid_df = df_filtered[(df_filtered["classification"] != "NA") & (df_filtered["prob"].notnull())]
summary = (
    valid_df.groupby("classification")
    .agg(
        count=("prob", "count"),
        avg_prob=("prob", "mean"),
        std_prob=("prob", "std")
    )
    .reset_index()
)
summary_path = os.path.join(output_dir, "annotated_intron_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"Intron summary saved to: {summary_path}")

# -----------------------------#
# 6. 繪製圖形：
#    左圖：Boxplot + Swarmplot (顯示各分類的原始 Exon Score)
#    右圖：平滑 eCDF (利用 Gaussian KDE 累積分布)
# -----------------------------#

def kde_cdf_smooth(values, x_grid=None, bw_method='scott', grid_size=300):
    """
    利用 Gaussian KDE 估計平滑 PDF，再進行累積和歸一化得到平滑 CDF。
    values: 一維樣本資料 (例如 prob)
    x_grid: 若為 None，則從資料最小值到最大值均勻取 grid_size 個點
    bw_method: 帶寬方法，預設 'scott'
    grid_size: 評估點數量，預設 300
    """
    values = np.array(values)
    if len(values) == 0:
        return np.array([]), np.array([])
    if x_grid is None:
        xmin, xmax = values.min(), values.max()
        x_grid = np.linspace(xmin, xmax, grid_size)
    kde = gaussian_kde(values, bw_method=bw_method)
    pdf = kde.evaluate(x_grid)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return x_grid, cdf

# 分組取得 RI 與 non-RI 的資料 (有效資料)
ri_df_plot = valid_df[valid_df["classification"] == "RI"]
nonri_df_plot = valid_df[valid_df["classification"] == "non-RI"]

# -------------------- 左圖：Boxplot + Swarmplot --------------------
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

sns.boxplot(x="classification", y="prob", data=valid_df,
            ax=ax1, showfliers=False, palette={"RI": "red", "non-RI": "black"},
            width=0.9, boxprops={"facecolor": "white", "edgecolor": "black"},
            whiskerprops={"color": "black"}, capprops={"color": "black"},
            medianprops={"color": "black"})
sns.swarmplot(x="classification", y="prob", data=valid_df,
              ax=ax1, palette={"RI": "red", "non-RI": "black"},
              dodge=False, size=4, linewidth=0)
ax1.set_xlabel("Classification", fontsize=14)
ax1.set_ylabel("Intron Score", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=14)

# -------------------- 右圖：平滑 eCDF --------------------
ri_probs = ri_df_plot["prob"].values
nonri_probs = nonri_df_plot["prob"].values
x_ri, cdf_ri = kde_cdf_smooth(ri_probs, bw_method='scott')
x_nonri, cdf_nonri = kde_cdf_smooth(nonri_probs, bw_method='scott')
ax2.plot(x_ri, cdf_ri, label="RI", color="red", linewidth=2)
ax2.plot(x_nonri, cdf_nonri, label="non-RI", color="black", linewidth=2)
ax2.set_xlabel("Intron Score", fontsize=14)
ax2.set_ylabel("eCDF", fontsize=14)
ax2.set_xlim(0, 1)
ax2.set_xticks([0.0, 0.5, 1.0])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=14)
ax2.tick_params(axis='both', labelsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plot_path = os.path.join(output_dir, "boxplot_swarm_ecdf_RI.png")
plt.savefig(plot_path, dpi=300)
print(f"Boxplot+Swarm and smoothed eCDF plot saved to: {plot_path}")
plt.show()

print("✅ Done!")
