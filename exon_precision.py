#!/usr/bin/env python
import os
import re
import glob
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt  # 可以移除
import random
import seaborn as sns           # 可以移除
from ipdb import set_trace

# Set to True for verbose output
trace = True

# ====== Helper functions ======
def parse_splice_file(filename):
    """讀取並回傳所有已註解(annotated)與 SMsplice 預測(splice)的 3SS / 5SS 資料，
       並同時解析 Sorted 5'/3' Splice Sites (若有) 的位置與分數。
    """
    with open(filename, "r") as f:
        text = f.read()

    # Regex patterns for 3SS / 5SS (annotated and SMsplice)
    pattern_5ss = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pattern_3ss = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    pattern_smsplice_5ss = re.compile(r"SMsplice 5SS:\s*\[([^\]]*)\]")
    pattern_smsplice_3ss = re.compile(r"SMsplice 3SS:\s*\[([^\]]*)\]")

    def parse_list(regex):
        match = regex.search(text)
        if not match:
            return set()
        inside = match.group(1).strip()
        if not inside:
            return set()
        items = re.split(r"[\s,]+", inside.strip())
        return set(map(int, items))

    annotated_5prime = parse_list(pattern_5ss)
    annotated_3prime = parse_list(pattern_3ss)
    viterbi_5prime = parse_list(pattern_smsplice_5ss)
    viterbi_3prime = parse_list(pattern_smsplice_3ss)

    if trace:
        print("annotated_5prime =", annotated_5prime)
        print("annotated_3prime =", annotated_3prime)
        print("viterbi_5prime =", viterbi_5prime)
        print("viterbi_3prime =", viterbi_3prime)

    # 解析 Sorted 5'/3' 區塊（若程式輸出有這部分）
    pattern_5prime_block = re.compile(
        r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)", 
        re.DOTALL
    )
    pattern_3prime_block = re.compile(
        r"Sorted 3['′] Splice Sites .*?\n(.*)", 
        re.DOTALL
    )
    pattern_line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")

    def parse_predictions(pattern):
        match_block = pattern.search(text)
        if not match_block:
            return []
        block = match_block.group(1)
        return [(int(m.group(1)), float(m.group(2))) for m in pattern_line.finditer(block)]

    fiveprime_preds = parse_predictions(pattern_5prime_block)
    threeprime_preds = parse_predictions(pattern_3prime_block)

    # 整理成一個 DataFrame
    rows = []
    # 1) 已出現在排序列表裡的 5' / 3' 預測
    for (pos, prob) in fiveprime_preds:
        rows.append((pos, prob, "5prime", pos in annotated_5prime, pos in viterbi_5prime))
    for (pos, prob) in threeprime_preds:
        rows.append((pos, prob, "3prime", pos in annotated_3prime, pos in viterbi_3prime))

    # 2) 沒出現在排序列表裡，但實際上是 annotated 或 SMsplice 所預測的 5' / 3' 位置 -> prob=0
    existing_5ss = {r[0] for r in rows if r[2] == "5prime"}
    existing_3ss = {r[0] for r in rows if r[2] == "3prime"}

    for pos in annotated_5prime - existing_5ss:
        rows.append((pos, 0.0, "5prime", True, pos in viterbi_5prime))
    for pos in annotated_3prime - existing_3ss:
        rows.append((pos, 0.0, "3prime", True, pos in viterbi_3prime))
    for pos in viterbi_5prime - existing_5ss:
        rows.append((pos, 0.0, "5prime", False, True))
    for pos in viterbi_3prime - existing_3ss:
        rows.append((pos, 0.0, "3prime", False, True))

    return pd.DataFrame(rows, columns=["position", "prob", "type", "is_correct", "is_viterbi"])


def method_column(method):
    return "is_correct" if method == "annotated" else "is_viterbi"

def prob3SS(pos, df, method="annotated"):
    """抓取 3' splice site 在某個 position (pos) 的機率 (prob)。"""
    row = df[
        (df["type"]=="3prime") & 
        (df[method_column(method)]==True) & 
        (df["position"]==pos)
    ]
    return row.iloc[0]["prob"] if not row.empty else 0

def prob5SS(pos, df, method="annotated"):
    """抓取 5' splice site 在某個 position (pos) 的機率 (prob)。"""
    row = df[
        (df["type"]=="5prime") & 
        (df[method_column(method)]==True) & 
        (df["position"]==pos)
    ]
    return row.iloc[0]["prob"] if not row.empty else 0


# ====== Main Batch Processing ======
species_map = {
    'h': 'human',
    'm': 'mouse',
    'z': 'zebrafish',
    't': 'arabidopsis',
    'o': 'moth',
    'f': 'fly'
}

seeds = [0]
top_ks = [1000]

results = []
all_exon_scores = []

for seed in tqdm(seeds, desc="Seeds"):
    for top_k in tqdm(top_ks, desc="top_k", leave=False):
        for code, name in species_map.items():
            # Input file pattern
            pattern = (
                f"./0_exon_score_parse_version_txt/{seed}_{code}_exon_score/"
                f"{code}_result_{top_k}/000_{name}_g_*.txt"
            )
            file_list = glob.glob(pattern)
            if not file_list:
                continue

            correct_exon_scores = []
            incorrect_exon_scores = []

            for txt_file in file_list:
                with open(txt_file) as f:
                    content = f.read()

                # 1. 讀取 annotated / SMsplice 之 5SS, 3SS
                match = re.search(r"Annotated 5SS:\s*(\[[^\]]*\])", content)
                if not match:
                    continue
                ann5ss = list(map(int, re.findall(r'\d+', match.group(1))))

                match = re.search(r"Annotated 3SS:\s*(\[[^\]]*\])", content)
                if not match:
                    continue
                ann3ss = list(map(int, re.findall(r'\d+', match.group(1))))

                match = re.search(r"SMsplice 5SS:\s*(\[[^\]]*\])", content)
                if not match:
                    continue
                sm5ss = list(map(int, re.findall(r'\d+', match.group(1))))

                match = re.search(r"SMsplice 3SS:\s*(\[[^\]]*\])", content)
                if not match:
                    continue
                sm3ss = list(map(int, re.findall(r'\d+', match.group(1))))

                # 2. 建立 annotated / SMsplice 的 exon pairs
                #    這裡設計是 (3SS, 5SS) 表示一個 exon。
                #    例如  (0, ann5ss[0]) 表示第一個 exon 的左界是 0(起點)，右界是 ann5ss[0]。
                #    也可依需求調整。
                ann_pairs_list = (
                    [(0, ann5ss[0])] +
                    [(ann3ss[i], ann5ss[i+1]) for i in range(min(len(ann3ss), len(ann5ss)-1))] +
                    [(ann3ss[-1], -1)]
                )
                sm_pairs_list = (
                    [(0, sm5ss[0])] +
                    [(sm3ss[i], sm5ss[i+1]) for i in range(min(len(sm3ss), len(sm5ss)-1))] +
                    [(sm3ss[-1], -1)]
                )

                # 用 set 方便後續直接用 in/not in 判斷
                ann_pairs_set = set(ann_pairs_list)
                sm_pairs_set = set(sm_pairs_list)

                # 3. 抓取 splice 機率
                df_splice = parse_splice_file(txt_file)

                # 4. 如果有附加的 [3SS, 5SS, prob] 表格（exon table），也先讀取進來
                exon_table = {}
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r"3SS,\s*5SS,\s*prob", line, re.IGNORECASE):
                        for subsequent_line in lines[i+1:]:
                            subsequent_line = subsequent_line.strip()
                            if not subsequent_line:
                                break
                            parts = subsequent_line.split(',')
                            if len(parts) >= 3:
                                try:
                                    exon_3ss = int(parts[0].strip())
                                    exon_5ss = int(parts[1].strip())
                                    exon_prob = float(parts[2].strip())
                                    exon_table[(exon_3ss, exon_5ss)] = exon_prob
                                except Exception:
                                    continue
                        break

                # 5. 開始檢查 SMsplice 預測的 exon pair，並計算分數
                for (three_site, five_site) in sm_pairs_set:
                    p3 = prob3SS(three_site, df_splice, method="smsplice")
                    p5 = prob5SS(five_site, df_splice, method="smsplice")

                    # 若是邊界，就強制其 prob = 1
                    if three_site == 0:
                        p3 = 1.0
                    if five_site == -1:
                        p5 = 1.0

                    # 若表格裡有顯示明確的分數，則以表格為主
                    if (three_site, five_site) in exon_table:
                        score = exon_table[(three_site, five_site)]
                    else:
                        score = p3 * p5

                    if score == 0.0:
                        continue

                    # **利用「整個 exon pair」來判斷正確性**
                    is_correct = ((three_site, five_site) in ann_pairs_set)

                    if is_correct:
                        correct_exon_scores.append(score)
                        all_exon_scores.append({
                            'seed': seed,
                            'top_k': top_k,
                            'species': code,
                            'label': 'correct',
                            'score': score
                        })
                    else:
                        incorrect_exon_scores.append(score)
                        all_exon_scores.append({
                            'seed': seed,
                            'top_k': top_k,
                            'species': code,
                            'label': 'incorrect',
                            'score': score
                        })

            # 6. 計算 precision_09 & subset_09
            total_pred = len(correct_exon_scores) + len(incorrect_exon_scores)
            if total_pred == 0:
                precision_09 = None
                subset_09 = None
            else:
                preds_above_09 = [
                    s for s in (correct_exon_scores + incorrect_exon_scores) 
                    if s >= 0.9
                ]
                correct_above_09 = [s for s in correct_exon_scores if s >= 0.9]

                if len(preds_above_09) > 0:
                    precision_09 = len(correct_above_09) / len(preds_above_09)
                    subset_09 = len(preds_above_09) / total_pred
                else:
                    precision_09 = None
                    subset_09 = 0.0  # 或 None，依需求

            results.append({
                "seed": seed,
                "top_k": top_k,
                "species": code,
                "precision_09": precision_09,
                "subset_09": subset_09
            })

# ====== Save final summary ======
df_result = pd.DataFrame(results)
df_result.to_csv("./csv/exon_scores_top_k_class.csv", index=False)
print("Saved precision_09 and subset_09 to smsplice_exon_scores_top_k_class.csv")

# ====== Also save the raw "correct"/"incorrect" exon scores if needed ======
df_exon_scores = pd.DataFrame(all_exon_scores)
df_exon_scores.to_csv("./csv/exon_scores_raw.csv", index=False)
print("\nNumber of total exon predictions recorded:", len(df_exon_scores))
print(df_exon_scores.head())
