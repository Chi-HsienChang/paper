#!/usr/bin/env python
import os
import re
import glob
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import seaborn as sns
from ipdb import set_trace

# Set to True for verbose output
trace = True

# ====== Helper functions ======
def parse_splice_file(filename):
    """Parse splice site info from a single text file."""
    with open(filename, "r") as f:
        text = f.read()

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

    # Optionally parse the "Sorted 5' / 3' Splice Sites" blocks if needed
    pattern_5prime_block = re.compile(
        r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)",
        re.DOTALL
    )
    pattern_3prime_block = re.compile(
        r"Sorted 3['′] Splice Sites .*?\n(.*)",
        re.DOTALL
    )
    pattern_line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")

    def parse_predictions(block_regex):
        """Parse lines `Position <pos>: <prob>` if found."""
        match_block = block_regex.search(text)
        if not match_block:
            return []
        block = match_block.group(1)
        return [
            (int(m.group(1)), float(m.group(2)))
            for m in pattern_line.finditer(block)
        ]

    fiveprime_preds = parse_predictions(pattern_5prime_block)
    threeprime_preds = parse_predictions(pattern_3prime_block)

    # Convert predictions into a DataFrame
    rows = []
    for (pos, prob) in fiveprime_preds:
        rows.append((pos, prob, "5prime", pos in annotated_5prime, pos in viterbi_5prime))
    for (pos, prob) in threeprime_preds:
        rows.append((pos, prob, "3prime", pos in annotated_3prime, pos in viterbi_3prime))

    # Add missing predictions with prob=0 if they appear in annotated or viterbi sets
    existing_5ss = {r[0] for r in rows if r[2] == "5prime"}
    existing_3ss = {r[0] for r in rows if r[2] == "3prime"}

    for pos in annotated_5prime - existing_5ss:
        rows.append((pos, 0.0, "5prime", True,  pos in viterbi_5prime))
    for pos in annotated_3prime - existing_3ss:
        rows.append((pos, 0.0, "3prime", True, pos in viterbi_3prime))
    for pos in viterbi_5prime - existing_5ss:
        rows.append((pos, 0.0, "5prime", False, True))
    for pos in viterbi_3prime - existing_3ss:
        rows.append((pos, 0.0, "3prime", False, True))

    return pd.DataFrame(
        rows,
        columns=["position", "prob", "type", "is_correct", "is_viterbi"]
    )

def method_column(method):
    """Return the correct DataFrame column name for the given method."""
    return "is_correct" if method == "annotated" else "is_viterbi"

def prob5SS(pos, df, method="annotated"):
    """Return the 5' site probability at position `pos` for the chosen method."""
    col = method_column(method)
    row = df[
        (df["type"] == "5prime") & (df[col] == True) & (df["position"] == pos)
    ]
    return row.iloc[0]["prob"] if not row.empty else 0.0

def prob3SS(pos, df, method="annotated"):
    """Return the 3' site probability at position `pos` for the chosen method."""
    col = method_column(method)
    row = df[
        (df["type"] == "3prime") & (df[col] == True) & (df["position"] == pos)
    ]
    return row.iloc[0]["prob"] if not row.empty else 0.0

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
all_intron_scores = []   # 用來記錄原始 intron 預測分數

for seed in tqdm(seeds, desc="Seeds"):
    for top_k in tqdm(top_ks, desc="top_k", leave=False):
        for code, name in species_map.items():
            pattern = (
                f"./0_intron_score_parse_version_txt/{seed}_{code}_intron_score/"
                f"{code}_result_{top_k}/000_{name}_g_*.txt"
            )
            file_list = glob.glob(pattern)
            if not file_list:
                continue

            # We'll track correct vs. incorrect intron scores
            correct_intron_scores = []
            incorrect_intron_scores = []

            for txt_file in file_list:
                with open(txt_file) as f:
                    content = f.read()

                # 1. 讀取 annotated & SMsplice 的 5SS / 3SS
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

                # 2. 組出 annotated / smsplice intron pair
                ann_introns = set(
                    (ann5ss[i], ann3ss[i])
                    for i in range(min(len(ann5ss), len(ann3ss)))
                )
                sm_introns = set(
                    (sm5ss[i], sm3ss[i])
                    for i in range(min(len(sm5ss), len(sm3ss)))
                )

                # set_trace()

                # 3. 取得 splice 分數 (df_splice)
                df_splice = parse_splice_file(txt_file)

                # 若有 intron table，例如：
                #   5SS,3SS,prob
                #   100,200,0.99
                intron_table = {}
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r"5SS,\s*3SS,\s*prob", line, re.IGNORECASE):
                        for subsequent_line in lines[i+1:]:
                            subsequent_line = subsequent_line.strip()
                            if not subsequent_line:
                                break
                            parts = subsequent_line.split(',')
                            if len(parts) >= 3:
                                try:
                                    intron_5 = int(parts[0].strip())
                                    intron_3 = int(parts[1].strip())
                                    intron_prob = float(parts[2].strip())
                                    intron_table[(intron_5, intron_3)] = intron_prob
                                except Exception:
                                    continue
                        break

                # 4. 計算 SMsplice 預測 intron 之分數，並判斷是否正確
                for (five_site, three_site) in sm_introns:
                    p5 = prob5SS(five_site, df_splice, method="smsplice")
                    p3 = prob3SS(three_site, df_splice, method="smsplice")

                    # 若表格有確切分數，則取表格；否則預設 p5*p3
                    if (five_site, three_site) in intron_table:
                        score = intron_table[(five_site, three_site)]
                    else:
                        score = p5 * p3

                    # 這裡不排除 score=0, 全部納入 overall
                    is_correct = ((five_site, three_site) in ann_introns)
                    if is_correct:
                        correct_intron_scores.append(score)
                        all_intron_scores.append({
                            'seed': seed,
                            'top_k': top_k,
                            'species': code,
                            'label': 'correct',
                            'score': score
                        })
                    else:
                        incorrect_intron_scores.append(score)
                        all_intron_scores.append({
                            'seed': seed,
                            'top_k': top_k,
                            'species': code,
                            'label': 'incorrect',
                            'score': score
                        })

            # 5. 計算 overall, precision_09, subset_09
            total_preds = len(correct_intron_scores) + len(incorrect_intron_scores)
            if total_preds == 0:
                overall_precision = None
                precision_09 = None
                subset_09 = None
            else:
                # (A) overall_precision：所有預測（含 score=0）
                overall_precision = len(correct_intron_scores) / total_preds

                # (B) precision_09 & subset_09：排除 score=0 後，再看分數≥0.9 的精準度
                correct_nonzero = [s for s in correct_intron_scores if s != 0]
                incorrect_nonzero = [s for s in incorrect_intron_scores if s != 0]
                total_nonzero_preds = len(correct_nonzero) + len(incorrect_nonzero)

                if total_nonzero_preds == 0:
                    # 若全部預測分數都為 0
                    precision_09 = None
                    subset_09 = 0.0
                else:
                    # 只計算 (非零) 中 >=0.9 的預測
                    scores_above_09 = [s for s in (correct_nonzero + incorrect_nonzero) if s >= 0.9]
                    correct_above_09 = [s for s in correct_nonzero if s >= 0.9]

                    if len(scores_above_09) > 0:
                        precision_09 = len(correct_above_09) / len(scores_above_09)
                        # subset_09 = 該高分區 (score≥0.9) 占非零預測比
                        subset_09 = len(scores_above_09) / total_nonzero_preds
                    else:
                        precision_09 = None
                        subset_09 = 0.0

            results.append({
                "seed": seed,
                "top_k": top_k,
                "species": code,
                "overall_precision": overall_precision,
                "precision_09": precision_09,
                "subset_09": subset_09
            })

# ====== Save summary ======
df_result = pd.DataFrame(results)
df_result.to_csv("./csv/intron_scores_top_k_class.csv", index=False)
print("Saved final metrics (overall_precision, precision_09, subset_09) to intron_scores_top_k_class.csv")

# ====== 另外輸出所有 intron 預測分數 ======
df_intron_scores = pd.DataFrame(all_intron_scores)
df_intron_scores.to_csv("./csv/intron_scores_raw.csv", index=False)
print("\nNumber of total intron predictions recorded:", len(df_intron_scores))
print(df_intron_scores.head())
