# 🔍 Layer-wise Analysis of BERT: From Feature Extraction to Decision Formation

This project investigates how different layers of **BERT (12-layer encoder)** contribute to solving NLP tasks by separating:

- **Feature representation** (what information is encoded)
- **Decision formation** (how that information is used)
---

## 🎯 Objective

Most transformer analyses assume:

> “Deeper layers learn more complex understanding.”

However, this mixes two different aspects:

- **Where information is most clearly represented**
- **Which layers actually influence final predictions**

This project reframes the problem by asking:

1. At which layer is task-relevant information most **linearly separable**?
2. Which layers contribute most to **final decision making**?
3. Do these two coincide?

We study this across two tasks:
- **POS Tagging**
- **STS-B Similarity**
---

## 📘 Tasks

### POS Tagging
Assigns grammatical labels to each word.

Example:  
The cat sat on the mat → DET NOUN VERB ADP DET NOUN  

---

### STS-B (Semantic Similarity)
Predicts similarity between two sentences.

Example:  
“A man is eating food”  
“A person is having a meal” → High similarity  

---

## 🔬 Methodology

We use two complementary techniques:

---

### 1. Linear Probing (Feature Extraction)

For each layer:

- Extract hidden representations
- Freeze BERT (no training)
- Train a simple **linear model**

This measures:

> “At which layer is task information most easily separable?”

If performance is high:
- features are well-structured
- decision boundary is simple

---

### 2. Merged Leave-One-Layer-Out Ablation (Decision Contribution)

- Combine all layer outputs using **mean pooling**
- Train a single model on the merged representation

Then for each layer:
- Remove that layer from the merge
- Measure performance drop

This measures:

> “Which layers matter most for final decision making?”

---

### Key Idea

- **Probing → representation quality**
- **Ablation → decision contribution**

---

## 📊 Results

### POS (Feature vs Decision)

![POS Probe vs Ablation](Results/Full%20Run%201/pos_probe_vs_ablation_normalized.png)

---

### STS-B (Feature vs Decision)

![STS-B Probe vs Ablation](Results/Full%20Run%201/stsb_probe_vs_ablation_normalized.png)

---

## 🧠 Results & Analysis

### 1. Intermediate layers are best for feature extraction

- POS best probing → **layer 6**
- STS-B best probing → **layer 5**

👉 These layers provide the most **linearly separable representations**

---

### 2. Higher layers contribute more to decisions

- POS highest ablation impact → **layer 11**
- STS-B highest ablation impact → **layer 10**

👉 These layers influence the **final prediction more strongly**

---

### 3. Representation ≠ Decision


This shows a separation between:

- **Feature extraction layers (mid layers)**
- **Decision/integration layers (top layers)**

---

### 4. Distributed information

- Removing a single layer causes only a small drop
- Information is **spread across layers**, not localized

---

## 📌 Interpretation

A consistent pattern emerges across both tasks:

- **Layers 5–6** → best for extracting useful features  
- **Layers 10–11** → more involved in combining these features for prediction  

This suggests:

> BERT first builds structured representations, then later layers transform them into task-specific decisions.

---

## ⚠️ Limitations

- **Mean merging is simplistic**  
  It may hide stronger layer-specific effects.

- **Ablation is not fully causal**  
  Layers are removed after computation, not within the transformer.

- **Linear probes are limited**  
  Only measure linearly separable information.

- **Single run**  
  Results may vary across seeds.

---

## ⚙️ Setup

- Model: `bert-base-uncased` (frozen)
- Tasks:
  - POS (UD English EWT)
  - STS-B (GLUE)
- Probes: Linear classifier / regressor
- Ablation: Mean merged leave-one-layer-out
- Framework: PyTorch + HuggingFace Transformers
