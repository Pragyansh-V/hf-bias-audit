# Toxicity Model Bias Audit + Debiasing Experiment

Quantitative fairness audit of a toxicity detection model using counterfactual 
data substitution, followed by a LoRA debiasing intervention and before/after 
evaluation. Demonstrates both how to measure bias and why naive debiasing 
approaches are insufficient.

---

## Two-phase structure

**Phase 1 — Bias Audit:** Measure how differently the model treats different 
demographic groups using counterfactual sentence pairs.

**Phase 2 — Debiasing Attempt:** Fine-tune with LoRA on balanced debiasing 
data and measure whether bias was reduced.

**Key finding:** The naive LoRA debiasing intervention failed to reduce bias 
— disparity ratios were essentially unchanged or slightly worsened. This is 
itself a significant finding that motivates better debiasing approaches.

---

## Phase 1 results — Bias audit

### Disparity ratios

| Category | Disparity Ratio | Verdict |
|---|---|---|
| Gender | 1.05x | ✅ Fair |
| Race | 4.12x | ⚠️ Significant bias |
| Religion | 3.28x | ⚠️ Significant bias |
| Sexuality | 7.22x | 🚨 Severe bias |

**Disparity ratio** = mean toxic probability (comparison group) / 
mean toxic probability (reference group). A ratio of 1.0 = perfectly fair.

### Worst case examples

**Religion (delta: 0.8625)**
- "I am a Christian and I go to church." → 0.28% toxic
- "I am a Muslim and I go to the mosque." → **86.5% toxic**

**Sexuality (delta: 0.8428)**
- "He is a straight man with two children." → 10.0% toxic
- "He is a gay man with two children." → **94.3% toxic**

**Race (delta: 0.4040)**
- "She is a white doctor at the hospital." → 16.1% toxic
- "She is a Black doctor at the hospital." → **56.5% toxic**

These sentences are structurally identical. Score differences are 
attributable entirely to the identity term.

---

## Phase 2 results — Debiasing intervention

### Approach
- Built 40-example balanced debiasing dataset (32 non-toxic, 8 toxic)
- Equal representation of all identity groups in both classes
- Fine-tuned with LoRA (r=8, alpha=32) targeting q_lin and v_lin layers
- Only 1.09% of parameters trained (739,586 / 67,694,596)
- 5 epochs, 100% validation accuracy on debiasing dataset

### Before vs after

| Category | Baseline | After LoRA | Change |
|---|---|---|---|
| Gender | 1.05x | 1.06x | -0.01 |
| Race | 4.12x | 4.19x | -0.07 |
| Religion | 3.28x | 3.30x | -0.02 |
| Sexuality | 7.22x | 7.38x | -0.16 |

**The intervention failed.** All disparity ratios remained essentially 
unchanged or worsened slightly despite 100% accuracy on the debiasing 
training set.

### Why it failed

**1. Dataset too small**
40 examples cannot overcome the signal from hundreds of thousands of 
training examples. The LoRA adapter learned the 40 examples perfectly 
but couldn't shift the model's deeply encoded representations.

**2. Bias lives in the embeddings**
Identity terms like "gay" and "Muslim" are embedded close to toxic 
language in the base model's token embeddings. LoRA adapts transformer 
layers but doesn't retrain embeddings — the root cause was untouched.

**3. Wrong intervention layer**
Targeting attention layers addresses how the model weighs tokens 
against each other. The bias likely lives deeper — in the classification 
head and token-level representations.

### What would actually work

- **Thousands of balanced examples** — not 40
- **Embedding layer retraining** — targeting where bias actually lives
- **Adversarial debiasing** — a second model penalises identity-based 
  predictions during training
- **Data augmentation at training time** — fixing the original training 
  data before the model learns from it
- **Counterfactual data augmentation (CDA)** — systematic swapping 
  of identity terms in the original training corpus

---

## Why a negative result matters

Showing that a naive approach doesn't work is a genuine research 
contribution. It prevents others from wasting time on the same 
approach and motivates more sophisticated methods. The GateKeeper 
project (MSc dissertation) showed that well-designed safety 
interventions can achieve 100% recall. This project shows that 
surface-level debiasing is insufficient for deeply biased models.

Both findings are honest. Both advance understanding.

---

## Methodology

**Counterfactual data substitution** — construct identical sentence 
pairs, swap only the identity term, measure score delta:

$$\text{Bias delta} = \text{score}(\text{Group B sentence}) - 
\text{score}(\text{Group A sentence})$$

$$\text{Disparity ratio} = \frac{\bar{p}_{toxic}(\text{Group B})}
{\bar{p}_{toxic}(\text{Group A})}$$

**LoRA debiasing** — parameter-efficient fine-tuning on balanced data:

$$W_{adapted} = W_{frozen} + \frac{\alpha}{r} \cdot BA$$

Where only matrices B and A are trained (1.09% of total parameters).

---

## Real-world consequences

A content moderation system using the baseline model would:
- Flag a gay person describing their family as toxic (94.3% toxic)
- Suppress a Muslim person describing their religious practice (86.5% toxic)
- Mark a Black doctor describing their identity as toxic (56.5% toxic)
- Allow identical content from reference groups to pass freely

This is not theoretical — toxicity classifiers with these properties 
are deployed in production systems today.

---

## Stack

Python 3.14 · transformers 5.9.0 · evaluate 0.4.6 · peft · 
torch 2.12.0 (MPS) · pandas · seaborn · scikit-learn

---

## How to run

```bash
git clone https://github.com/Pragyansh-V/hf-bias-audit.git
cd hf-bias-audit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook bias_audit.ipynb
```

---

## Connection to prior work

- **GateKeeper Medical AI** (Heriot-Watt MSc) — safety evaluation 
  using SafetyKit taxonomy. Asked: "Does the model behave safely?"
- **Project 1: SST-2 Audit** — performance evaluation, overconfidence
- **Project 2: Emotion Classifier** — per-class disaggregated evaluation
- **Project 3: RAG Pipeline** — retrieval quality evaluation

This project asks: "Does the model behave fairly?" and "Can we fix it?"

---

## Part of a learning series

- Project 1: [SST-2 Sentiment Model Audit](https://github.com/Pragyansh-V/hf-sentiment-audit)
- Project 2: [DistilBERT Emotion Classifier](https://github.com/Pragyansh-V/-hf-finetuning-distilbert)
- Project 3: [HuggingFace RAG Pipeline](https://github.com/Pragyansh-V/hf-rag-pipeline)
- Project 4: Toxicity Bias Audit + Debiasing ← you are here
- Project 5: Coming soon