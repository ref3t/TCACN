# Automated Generation of TC–ACN Incident Vectors Using Large Language Models

## Overview
This repository contains the artifacts, data, and evaluation scripts for the paper:

> **"Incident Reporting in accordance with the taxonomy of the Agenzia per la Cybersicurezza Nazionale"**

The paper investigates whether large language models (LLMs) can generate **TC–ACN incident classification vectors** consistent with the official taxonomy issued by the Italian **National Cybersecurity Agency (ACN)**.  
We compare multiple model configurations and evaluate their accuracy against manually annotated ground truth.

---

## Research Goals

The main objectives of this work are:

1. Assess whether LLMs can automatically generate **accurate and complete TC–ACN vectors** from real incident descriptions.  
2. Compare multiple LLM configurations (free and paid, same/different sessions, with/without taxonomy file).  
3. Quantitatively evaluate **correctness, completeness, and consistency** using Hamming-based metrics.  

---

## Approach

1. **Dataset**  
   Incidents were extracted from the **Splunk BOTSv3** dataset. (https://github.com/splunk/botsv3)

2. **Ground Truth**  
   Each incident was manually annotated according to the **TC–ACN taxonomy (v1.0, 2024)**.

3. **Model Configurations**
   - Free ChatGPT with/without taxonomy
   - Paid ChatGPT with/without taxonomy
   - Same-session vs. cross-session runs

4. **Metrics**
   - Hamming Accuracy  
   - Completeness (undetermined predicates)  
   - Mismatch Other values distribution across predicates  

---

## Reproducing the Analysis and Figures

The scripts in `Results/` operate directly on the latest workbook (`Data/FFFFFFinalResults.xlsx`, 52 logs). The commands below recompute the metrics and generate every figure referenced in the paper. Feel free to append `--show` (when supported) to open the Matplotlib window in addition to saving the PNG.

### 1. Recompute final metrics / refresh the workbook
```
python Results/compute_final_results.py \
  --dataset Data/FFFFFFinalResults.xlsx \
  --detailed-output Data/FFFFFFinalResults.xlsx \
  --output Results/FFFFFFinalResults_summary.xlsx
```
- Normalizes all TC-ACN vectors, rebuilds every `*_diff_*`, `*_hamming`, and OT-equivalent column, and writes the updated workbook plus a summary file under `Results/`.

### 2. Hamming accuracy and difference boxplots (Figure 4)
```
python Results/Boxplot.py --dataset Data/FFFFFFinalResults.xlsx --output Results
```
- Reads every `*_hamming` and `*_diff_count` column, then saves `Results/boxplot_hamming.png` and `Results/boxplot_diff.png` with the means/medians annotated per configuration.

### 3. OT-equivalent count distribution (Figure 5)
```
python Results/Mismatches.py \
  --dataset Data/FFFFFFinalResults.xlsx \
  --experiment "GPT+_TC-CAN_With_File_Different_Session" \
  --output Results
```
- Plots the manual OT counts, the chosen experiment’s OT-equivalent counts, and their mismatch counts, recreating the OT boxplot in `Results/boxplot_ot.png`.

### 4. OT-equivalent mismatch categories (Figure 6)
```
python RQ2CategoryPlot.py \
  --dataset Data/FFFFFFinalResults.xlsx \
  --experiment "GPT+_TC-CAN_With_File_Different_Session" \
  --table-output Results/ot_mismatch_categories.xlsx \
  --figure-output Results/ot_mismatch_categories.png
```
- Aggregates `<experiment>_ot_equivalent_mismatch_categories`, saves the sorted counts to Excel, and produces the bar chart showing where OT mismatches occur across the taxonomy.

### 5. Generic (non-OT) mismatch categories
```
python Results/PlotMismatchCategories.py \
  --dataset Data/FFFFFFinalResults.xlsx \
  --experiment "GPT+_TC-CAN_With_File_Different_Session" \
  --table-output Results/diff_mismatch_categories_no_ot.xlsx \
  --figure-output Results/diff_mismatch_categories_no_ot.png \
  --ignore-ot
```
- Tokenizes `<experiment>_diff_locations`, optionally excludes rows where either side equals `OT`, and charts the remaining predicate mismatches. Remove `--ignore-ot` if you want the complete (OT-inclusive) counts.

---

---





