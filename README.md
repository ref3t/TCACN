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

---

## Future Work

We plan to explore three directions:

1. **Dataset Collaboration with CSIRTs**  
   Request real-world datasets for training and fine-tuning tailored models.

2. **Migration to TC–ACN v2.0**  
   Re-run experiments with the updated taxonomy (released November 2025) using:
   - re-annotated data, or  
   - new datasets  
   to evaluate adaptability and performance under the revised standard.

3. **Local vs. Online LLMs**
   Compare local models (e.g., Ollama) with online solutions (GPT, Llama 3)

   Local models may be attractive for organizations unable or unwilling to pay for commercial cloud services, or those with strict privacy constraints.

---





