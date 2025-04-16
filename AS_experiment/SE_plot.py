import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

# -----------------------------#
# 1. 讀取 SE_events_high.csv，建立 SE 資訊字典
# 格式：(gene, 5ss, 3ss) -> PSI
# -----------------------------#
csv_path = "./data/SE_events_high.csv"
se_data = pd.read_csv(csv_path)

se_info = {}
for _, row in se_data.iterrows():
    gene = row["gene"]
    s5 = row["5ss"]
    s3 = row["3ss"]
    psi_val = row["PSI"]
    se_info[(gene, s5, s3)] = psi_val

# -----------------------------#
# 2. 掃描 exon 解析 txt 檔案並產生 exon pair 記錄
# -----------------------------#
# 根據實際情況調整資料夾路徑，此處使用 "./t_allParse_txt/t_allParse_exon" 或 "./t_result_exon_1000"
txt_folder = "./t_allParse_txt/t_allParse_exon"

records = []
# 定義正則表達式 (抓取 gene, index, Annotated 5SS/3SS 與 "3SS, 5SS, prob")
gene_regex    = re.compile(r"Gene\s*=\s*(\S+)")
index_regex   = re.compile(r"index\s*=\s*(\d+)")
ann_5ss_regex = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
ann_3ss_regex = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
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

    # 取得 Annotated 5SS 與 3SS 的列表 (取中括號內所有數字)
    ann5_match = ann_5ss_regex.search(content)
    ann3_match = ann_3ss_regex.search(content)
    
    if gene_id and ann5_match and ann3_match:
        annotated_5_list = [int(x) for x in re.findall(r"\d+", ann5_match.group(1))]
        annotated_3_list = [int(x) for x in re.findall(r"\d+", ann3_match.group(1))]
        # 產生 exon pair：以 (annotated_3[i], annotated_5[i+1]) 表示
        pair_count = min(len(annotated_3_list), len(annotated_5_list)) - 1
        for i in range(pair_count):
            three_ss = annotated_3_list[i]
            five_ss  = annotated_5_list[i+1]
            # 判斷是否屬於 SE event：檢查 (gene, five_ss, three_ss) 是否存在於 se_info 中
            if (gene_id, five_ss, three_ss) in se_info:
                classification = "SE"
                psi_val = se_info[(gene_id, five_ss, three_ss)]
            else:
                classification = "non-SE"
                psi_val = None
            # 從檔案中尋找對應 exon pair 的預測 Exon Score
            prob = None
            for line in content.splitlines():
                line = line.strip()
                m = exon_line_regex.match(line)
                if m:
                    three_val = int(m.group(1))
                    five_val  = int(m.group(2))
                    p         = float(m.group(3))
                    if three_val == three_ss and five_val == five_ss:
                        prob = p
                        break
            records.append({
                "gene": gene_id,
                "5ss": five_ss,
                "3ss": three_ss,
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
# 4. 只考慮指定 allowed indexes 的 txt 檔
# -----------------------------#
allowed_indexes = {
    1, 3, 5, 6, 7, 9, 32, 44, 47, 48, 55, 58, 70, 75, 79, 129, 134, 138, 141,
    152, 154, 165, 168, 171, 173, 179, 180, 181, 194, 205, 208, 209, 211, 219,
    225, 226, 230, 251, 263, 265, 268, 269, 271, 272, 277, 278, 283, 287, 292,
    299, 308, 309, 319, 320, 328, 338, 344, 350, 355, 366, 378, 380, 384, 385,
    387, 398, 406, 414, 419, 426, 437, 438, 478, 496, 500, 508, 521, 523, 526,
    527, 531, 532, 533, 536, 545, 554, 555, 557, 559, 565, 567, 577, 583, 584,
    586, 593, 594, 595, 597, 598, 609, 620, 631, 634, 637, 645, 660, 692, 695,
    697, 699, 724, 749, 753, 773, 801, 809, 816, 820, 826, 833, 834, 846, 858,
    863, 876, 899, 906, 908, 927, 935, 936, 943, 960, 976, 979, 988, 994, 997,
    998, 1010, 1012, 1014, 1016, 1017, 1022, 1024, 1030, 1042, 1045, 1047, 1048,
    1050, 1058, 1061, 1074, 1077, 1083, 1087, 1091, 1094, 1098
}
df_filtered = df[df["index"].isin(allowed_indexes)]

# -----------------------------#
# 5. 輸出結果：詳細檔與統計摘要
# -----------------------------#
output_dir = "./SE_result"
os.makedirs(output_dir, exist_ok=True)

details_path = os.path.join(output_dir, "annotated_exon_details_filtered.csv")
df_filtered.to_csv(details_path, index=False)
print(f"Filtered detailed exon data (only allowed indexes) saved to: {details_path}")

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
summary_path = os.path.join(output_dir, "annotated_exon_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"Exon summary saved to: {summary_path}")

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

# 分組取得 SE 與 non-SE 的資料 (有效資料)
se_df_plot = valid_df[valid_df["classification"] == "SE"]
nonse_df_plot = valid_df[valid_df["classification"] == "non-SE"]

# -------------------- 左圖：Boxplot + Swarmplot --------------------
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

sns.boxplot(x="classification", y="prob", data=valid_df,
            ax=ax1, showfliers=False, palette={"SE": "red", "non-SE": "black"},
            width=0.9, boxprops={"facecolor": "white", "edgecolor": "black"},
            whiskerprops={"color": "black"}, capprops={"color": "black"},
            medianprops={"color": "black"})
sns.swarmplot(x="classification", y="prob", data=valid_df,
              ax=ax1, palette={"SE": "red", "non-SE": "black"},
              dodge=False, size=4, linewidth=0)
ax1.set_xlabel("Classification", fontsize=14)
ax1.set_ylabel("Exon Score", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=14)

# -------------------- 右圖：平滑 eCDF --------------------
se_probs = se_df_plot["prob"].values
nonse_probs = nonse_df_plot["prob"].values
x_se, cdf_se = kde_cdf_smooth(se_probs, bw_method='scott')
x_nonse, cdf_nonse = kde_cdf_smooth(nonse_probs, bw_method='scott')
ax2.plot(x_se, cdf_se, label="SE", color="red", linewidth=2)
ax2.plot(x_nonse, cdf_nonse, label="non-SE", color="black", linewidth=2)
ax2.set_xlabel("Exon Score", fontsize=14)
ax2.set_ylabel("eCDF", fontsize=14)
ax2.set_xlim(0, 1)
ax2.set_xticks([0.0, 0.5, 1.0])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=14)
ax2.tick_params(axis='both', labelsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plot_path = os.path.join(output_dir, "boxplot_swarm_ecdf.png")
plt.savefig(plot_path, dpi=300)
print(f"Boxplot+Swarm and smoothed eCDF plot saved to: {plot_path}")
plt.show()

print("✅ Done!")
