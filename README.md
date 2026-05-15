# Cultural Item Classification using GCN and DistilBERT

This repository contains the implementation and experiments for a cultural item classification task using both:

- A **Graph Neural Network (GCN)** based approach (non-language model)
- A **Transformer-based Language Model (DistilBERT)** approach

The goal is to classify cultural items into:

- **Cultural Agnostic**
- **Cultural Representative**
- **Cultural Exclusive**

based on structured knowledge graph information and textual descriptions.

---

## 📌 Project Overview

This project explores two different paradigms for cultural classification.

### 1. Non-LM Approach: Graph Neural Network (GCN)

The non-language-model method formulates the problem as a **node classification task** over a **knowledge graph** built from Wikidata.

Key components:

- Knowledge graph construction using **Wikidata SPARQL**
- Graph representation with **NetworkX**
- Node classification using **Graph Convolutional Networks (GCN)** via `torch_geometric`
- Feature extraction using:
  - TF-IDF
  - Truncated SVD
  - One-hot encoded metadata

The GCN performs message passing across connected entities to leverage cultural relationships between items.

---

### 2. LM Approach: DistilBERT

The language-model solution uses:

- `distilbert-base-uncased`
- Hugging Face `transformers`
- `Trainer` API
- Optuna hyperparameter optimization

All textual fields are concatenated and tokenized with a maximum sequence length of 512 tokens.

The model is fine-tuned for sequence classification using:

- Cross-entropy loss
- Weighted F1 score
- Dynamic padding
- Automatic hyperparameter search with Optuna

---

## ⚙️ Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- Hugging Face Transformers
- Optuna
- Scikit-learn
- NetworkX
- Wikidata SPARQL API

---

## 🧠 Model Architectures

### GCN Configuration

- Hidden layers: 64 → 96
- Activation: ReLU
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.005
- Weight decay: 5e-4
- Epochs: 300

### DistilBERT Hyperparameter Search

Search space:

- Learning rate: `[1e-6, 5e-5]`
- Weight decay: `[0.0, 0.1]`
- Batch size: `{16, 32}`
- Epochs: `{2, 3, 4, 5}`

Optimization objective:

- Maximize weighted F1 score on validation set

---

## 📊 Results

| Model | Accuracy | Weighted F1 |
|---|---|---|
| GCN (non-LM) | 0.64 | 0.63 |
| DistilBERT (LM) | 0.76 | 0.75 |

### Key Findings

- The **language model significantly outperformed** the graph-based approach.
- DistilBERT handled minority cultural classes better thanks to:
  - richer textual understanding
  - pretrained world knowledge
  - access to semantic clues in descriptions

---

## 📈 Visualizations

The report includes:

- GCN training loss curve
- Validation accuracy curve
- Confusion matrices for both models

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the GCN Model

```bash
python train_gcn.py
```

### Train the DistilBERT Model

```bash
python train_lm.py
```

---

## 📂 Project Structure

```text
.
├── data/
├── models/
├── train_gcn.py
├── train_lm.py
├── utils/
├── results/
└── README.md
```

---

## 📖 Report

The full project report describing the methodologies, experiments, and results is available in `Report.pdf`.

---

## 👥 Authors

- Sahar Khanlari
- Marco Natale
