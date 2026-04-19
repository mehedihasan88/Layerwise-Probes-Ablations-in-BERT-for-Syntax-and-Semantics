# 🔍 Layer-wise Analysis of BERT (Probing vs Ablation)

This project explores how different layers of **BERT (12-layer encoder)** represent and contribute to:

- **POS Tagging (Syntax)**
- **STS-B (Semantic Similarity)**

Instead of only looking at performance, we try to understand *what each layer is actually doing* using two complementary methods:

- **Linear Probing** → How much information can we extract from a single layer?  
- **Merged Leave-One-Layer-Out Ablation** → How much does a layer contribute to the final combined representation?

---

## 📊 Results

### Combined Probing (Normalized)

![Probing Results](Results/Full%20Run%201/combined_probing_normalized.png)

---

## 🧠 Key Findings

### 1. Mid-layers are the most informative

- POS peaks at **layer 6 (F1 ≈ 0.917)**
- STS-B peaks at **layer 5 (Pearson ≈ 0.81)**  

This suggests that **intermediate layers capture the most useful features** for both syntax and semantics.

---

### 2. Top layers contribute more to the final representation

- POS ablation peak → **layer 11**
- STS-B ablation peak → **layer 10**

So even though these layers are not the best individually, they still play an important role when all layers are combined.

---

### 3. A key insight

> **Information ≠ Importance**

A layer can:
- contain useful information (high probing score)
- but still not be critical to the final prediction (low ablation impact)

---

## 📌 Interpretation

Putting everything together:

- Information is **distributed across layers**, not isolated
- Middle layers → best for **extracting features**
- Higher layers → help **refine and combine representations**
- There is strong **redundancy** — removing a single layer barely hurts performance

---

## ⚠️ Limitations

This experiment gives useful insights, but there are some important limitations:

- **Mean merging is simplistic**  
  We combine layers using a simple average, which may smooth out differences and make layers appear more redundant than they actually are.

- **Ablation is not fully causal**  
  We remove layers from a merged representation, not from the internal transformer computation. So this does not reflect true causal importance inside BERT.

- **Linear probes are limited**  
  Since we use only a linear classifier, we only measure *linearly accessible information*. Some information might exist but not be linearly separable.

- **Single run (no variance analysis)**  
  Results are based on one run. Layer rankings may shift slightly across different seeds or setups.

---

## ⚙️ Setup

- Model: `bert-base-uncased` (frozen)
- Tasks:
  - POS (UD English EWT)
  - STS-B (GLUE)
- Probes: Linear classifier / regressor
- Ablation: Mean merged leave-one-layer-out

---

## 🚀 Takeaway

> Just because a layer makes information easy to extract does not mean it is essential for the model’s final behavior.

This highlights an important distinction between:
- **what a model knows (representation)**
- and **what it actually uses (contribution)**