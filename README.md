# 🔍 Layer-wise Analysis of BERT (Probing vs Ablation)

This project investigates how different layers of **BERT (12-layer encoder)** represent and contribute to language understanding.

Instead of focusing only on performance, we aim to understand:
- what each layer learns  
- where information is stored  
- how different layers contribute to final predictions  

---

## 🎯 Objective

Modern transformer models like BERT are often treated as black boxes. A common assumption is:

> “Deeper layers capture more complex understanding.”

However, this assumption mixes two different ideas:

- **Representation** → what information a layer contains  
- **Contribution** → how important that layer is for prediction  

This project separates these two by asking:

1. Where is task-related information stored in BERT?  
2. Which layers actually matter for making predictions?  
3. Do the most informative layers also contribute the most?

We analyze this across two types of tasks:
- **Syntax POS tagging(Parts Of Speecch tagging)**  
- **Semantics (similarity between two sentences)**  

---

## 📘 Tasks

### POS Tagging (Syntax)
Assigns grammatical roles to each word in a sentence.

Example:
The cat sat on the mat → DET NOUN VERB ADP DET NOUN  

This task evaluates how well the model understands **sentence structure**.

---

### STS-B (Semantic Similarity)
Measures how similar two sentences are.

Example:
“A man is eating food”  
“A person is having a meal”  
→ High similarity  

This task evaluates **meaning understanding**.

---

## 🔬 Methodology

We use two complementary techniques:

### 1. Linear Probing (Information Extraction)

For each BERT layer:
- Extract hidden representations
- Freeze BERT (no training)
- Train a simple **linear classifier/regressor**

This answers:

> “How much task-relevant information is present in this layer?”

- POS → token-level classification  
- STS-B → sentence-level regression (after pooling)  

---

### 2. Merged Leave-One-Layer-Out Ablation (Contribution)

Instead of using a single layer:

- Combine all layers using **mean pooling**
- Train one linear model on the merged representation

Then for each layer:
- Remove that layer from the merge
- Evaluate performance drop (without retraining)

This answers:

> “How much does this layer contribute to the full representation?”

---

### Why both?

- Probing → **what a layer knows**  
- Ablation → **what the model uses**

Comparing them reveals:
- redundancy  
- indirect contributions  
- mismatch between knowledge and usage  

---

## 📊 Results

![Probing Results](Results/Full%20Run%201/combined_probing_normalized.png)

---

## 🧠 Findings

### 1. Mid-layers are most informative

- POS peaks at **layer 6 (F1 ≈ 0.917)**
- STS-B peaks at **layer 5 (Pearson ≈ 0.81)**  

👉 Intermediate layers encode the strongest task-relevant features.

---

### 2. Higher layers contribute more to final predictions

- POS ablation peak → **layer 11**
- STS-B ablation peak → **layer 10**

👉 Higher layers help **refine and combine information**, even if they are not the best individually.

---

### 3. Probing and ablation do not align

- Best probing layers ≠ most impactful layers  

This shows:

- Some layers are **informative but redundant**  
- Some layers contribute **indirectly**  

---

## 📌 Interpretation

- Information in BERT is **distributed across layers**
- Middle layers → strongest for **feature extraction**
- Higher layers → involved in **aggregation**
- Removing one layer causes only small drops → **high redundancy**

---

## ⚠️ Limitations

- **Mean merging is simplistic**  
  Averaging layers smooths differences and may hide true importance.

- **Ablation is not fully causal**  
  Layers are removed only from the merged representation, not from internal transformer computation.

- **Linear probes are limited**  
  Only linearly accessible information is measured.

- **Single run**  
  Results may vary slightly with different seeds.

---

## ⚙️ Setup

- Model: `bert-base-uncased` (frozen)
- Tasks:
  - POS (UD English EWT)
  - STS-B (GLUE benchmark)
- Probe: Linear classifier / regressor
- Ablation: Mean merged leave-one-layer-out