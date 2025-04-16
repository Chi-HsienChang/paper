import pandas as pd
import os
import re

# 讀取 SE_events_high.csv
se_df = pd.read_csv("./data/RI_events_high.csv")

# 建立一個新欄位 index 與 prob
se_df["index"] = -1
se_df["prob"] = -1.0

# 資料夾路徑
folder = "t_result_intron_1000"

# 建立 gene → index 的對照表
gene_to_index = {}
for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        path = os.path.join(folder, filename)
        with open(path) as f:
            content = f.read()
        match = re.search(r"Gene\s*=\s*(\S+)", content)
        if match:
            gene = match.group(1)
            index_match = re.search(r"_g_(\d+)\.txt", filename)
            if index_match:
                index = int(index_match.group(1))
                gene_to_index[gene] = (index, content)

# 對 SE_events 高 PSI 的每一列資料加入 index 與 prob
for i, row in se_df.iterrows():
    gene = row["gene"]
    ss_5 = int(row["5ss"])
    ss_3 = int(row["3ss"])

    if gene not in gene_to_index:
        continue

    index, text = gene_to_index[gene]

    # 嘗試在 txt 檔中找相符的 5SS 和 3SS 的機率
    prob = None
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^\d+,\s*\d+,\s*[0-9.]+$", line):
            fields = line.split(",")
            five, three, p = int(fields[0]), int(fields[1]), float(fields[2])
            if three == ss_3 and five == ss_5:
                prob = p
                break

    # 寫入 index 與 prob（找不到 prob 則為 -1）
    se_df.at[i, "index"] = index
    se_df.at[i, "prob"] = prob if prob is not None else -1.0

# 儲存結果
# se_df.to_csv("SE_events_with_index_prob.csv", index=False)


# 篩選出有找到機率的資料
filtered_df = se_df[se_df["prob"] != -1.0]

# 存成另一份檔案
filtered_df.to_csv("./RI_result/RI_events_with_prob_only.csv", index=False)
print("所有成功對應到的 index：", sorted(filtered_df["index"].unique()))

print("共有幾個：", len(sorted(filtered_df["index"].unique())))



########################

import os
import re

# 已成功取得 prob 的 index 集合
available_indexes = set(filtered_df["index"])

# 抓出 t_allParse_exon 中有哪些 index 存在
allparse_folder = "t_allParse_intron"
existing_indexes = set()

for filename in os.listdir(allparse_folder):
    if filename.endswith(".txt"):
        match = re.search(r"_g_(\d+)\.txt", filename)
        if match:
            idx = int(match.group(1))
            existing_indexes.add(idx)

# 計算交集與差集
present = sorted(available_indexes & existing_indexes)
missing = sorted(available_indexes - existing_indexes)

# 輸出結果
print("✅ 在 t_allParse_intron 中存在的 index：")
print(present)
print("len = ", len(present))

print("\n❌ 在 t_allParse_intron 中不存在的 index：")
print(missing)
print("len = ", len(missing))

