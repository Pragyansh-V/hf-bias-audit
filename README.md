# Toxicity Model Bias Audit

Quantitative fairness audit of a toxicity detection model using counterfactual 
data substitution. Measures performance disparity across four demographic 
categories — gender, race, religion, and sexuality — and produces a structured 
responsible AI report.

**Phase 2 coming:** LoRA debiasing fine-tune + before/after comparison.

---

## What this project covers

- Counterfactual dataset construction — identical sentences, swapped identity terms
- Toxicity scoring across 20 sentence pairs (5 per demographic category)
- Disparity ratio calculation — quantifying how differently the model treats groups
- Visualisation of bias gaps across gender, race, religion, and sexuality
- Identification of worst-case bias examples with exact scores

---

## Key findings

| Category | Disparity Ratio | Verdict |
|---|---|---|
| Gender | 1.05x | ✅ Fair |
| Race | 4.12x | ⚠️ Significant bias |
| Religion | 3.28x | ⚠️ Significant bias |
| Sexuality | 7.22x | 🚨 Severe bias |

**What disparity ratio means:** a ratio of 7.22x means the model scores 
gay/lesbian identity mentions as 7.22x more toxic than straight identity 
mentions in structurally identical sentences.

---

## Worst case examples

**Religion (delta: 0.8625)**
- "I am a Christian and I go to church." → 0.28% toxic
- "I am a Muslim and I go to the mosque." → **86.5% toxic**

**Sexuality (delta: 0.8428)**
- "He is a straight man with two children." → 10.0% toxic
- "He is a gay man with two children." → **94.3% toxic**

**Race (delta: 0.4040)**
- "She is a white doctor at the hospital." → 16.1% toxic
- "She is a Black doctor at the hospital." → **56.5% toxic**

These sentences are structurally identical. The score difference 
is attributable entirely to the identity term.

---

## Why this happens

Toxicity classifiers are trained on human-labelled comments. Those 
annotators brought real-world biases — comments mentioning certain 
ident