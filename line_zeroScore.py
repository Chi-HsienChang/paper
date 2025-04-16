import matplotlib.pyplot as plt

# 資料
k_values = [100, 500, 1000]
data = {
    "Human": [47.7, 35.3, 30.6],
    "Mouse": [40.1, 29.3, 25.1],
    "Zebrafish": [6.9, 4.9, 4.1],
    "Arabidopsis": [4.7, 2.3, 1.5],
    "Moth": [17.3, 14.3, 12.2],
    "Fly": [10.2, 6.7, 5.8]
}

# 自訂 legend 排序
custom_order = ["Human", "Mouse",  "Moth", "Fly", "Zebrafish", "Arabidopsis"]

# 自訂顏色（避免咖啡色）
color_map = {
    "Human": "#1f77b4",        # 藍
    "Mouse": "#ff7f0e",        # 橘
    "Moth": "#2ca02c",         # 綠
    "Fly": "#d62728",          # 紅
    "Zebrafish": "#9467bd",    # 紫
    "Arabidopsis": "#e377c2"   # 粉紅（取代咖啡色）
}

# 繪圖
plt.figure(figsize=(5, 5))
for species in custom_order:
    plt.plot(k_values, data[species], marker='o', label=species, color=color_map[species])

# 標籤與圖例
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Percentage (%)", fontsize=14)
plt.xticks(k_values, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

# 圖例：2列×3欄，Moth & Fly 中間
plt.legend(fontsize=12, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.22))
plt.tight_layout()

plt.savefig("line_plot.png", dpi=300)
plt.show()
