# 🏆 CuseHacks Datathon 2026 — Track 2 | Category 2

> **Winner — Track 2 (Multi-Label Text Classification)**  
> CuseHacks Datathon 2026 · Hosted by Innovate Orange · Syracuse University

---

## 📋 Task

Multi-label text classification on a collection of newswire articles. Each article may belong to **one or more** of **116 topic categories** (e.g. `earn`, `trade`, `wheat`, `crude`, `money-fx`).

**Category 2** — All representations learned from scratch using provided data only. No pretrained embeddings or language models used.

---

## 🎯 Results

| Metric | Score |
|---|---|
| **Micro F1 (Primary)** | **0.827** |
| Macro F1 | 0.612 |
| Exact-Match Accuracy | 0.789 |
| Hamming Accuracy | 0.996 |

---

## 🗂️ Repository Structure

```
├── data/
│   ├── train.csv          ← training articles with labels
│   ├── val.csv            ← validation articles with labels
│   └── label_list.txt     ← all 116 valid topic labels
├── train.py               ← trains and saves model artifacts
├── predict.py             ← inference script (submitted to organizers)
├── requirements.txt       ← pinned dependencies
└── report.pdf             ← technical report (2 pages)
```

> **Note:** `model/` folder containing trained `.pkl` artifacts is not included in the repo due to file size (~90MB). Run `train.py` to regenerate it locally.

---

## ⚙️ Pipeline Overview

### 1. Preprocessing
- Combined `title` (repeated 2×) + `body text` for stronger title signal
- Lowercased, removed punctuation and numbers via regex
- Filtered custom stopword list (generic English + finance noise: `mln`, `pct`, `dlrs`, `corp`)
- Removed tokens shorter than 3 characters

### 2. Feature Extraction (TF-IDF)
- `max_features = 75,000`
- `ngram_range = (1, 2)` — unigrams + bigrams (e.g. *crude oil*, *money supply*)
- `min_df = 1` — retains rare terms for low-frequency labels
- `sublinear_tf = True` — log-dampens high-frequency terms

### 3. Model
- `OneVsRestClassifier` — trains 116 independent binary classifiers
- `LogisticRegression` with `C=7.0`, `class_weight='balanced'`, `solver='lbfgs'`
- `class_weight='balanced'` handles severe label imbalance (e.g. `none`: 7,132 vs `castor-oil`: 1 example)

### 4. Threshold Tuning ← key innovation
- Default 0.5 threshold fails — model probabilities are spread thin across 116 labels
- Swept thresholds from 0.01 → 0.50 per label on validation set
- Selected threshold maximizing per-label F1
- This single step was the biggest driver of performance improvement

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train.py
```
Trains on `data/train.csv`, tunes thresholds on `data/val.csv`, saves 4 artifacts to `model/`:
- `model/classifier.pkl` — trained OneVsRest model
- `model/tfidf.pkl` — fitted TF-IDF vectorizer
- `model/mlb.pkl` — MultiLabelBinarizer
- `model/thresholds.pkl` — per-label tuned thresholds

### Self-evaluate on val set
Set paths at top of `predict.py`:
```python
INPUT_PATH  = "data/val.csv"
LABELS_PATH = "data/val.csv"
```
Then run:
```bash
python predict.py
```

### Run inference on test set
```python
INPUT_PATH  = "test.csv"
OUTPUT_PATH = "predictions.csv"
LABELS_PATH = None
MODEL_PATH  = "model/"
```
```bash
python predict.py
```

---

## 📦 Dependencies

```
scikit-learn==1.8.0
pandas==3.0.1
numpy==2.0.0
joblib==1.5.0
```

---

## 📄 Output Format

One prediction per article, pipe-separated:
```
article_id,topics
1001,earn
1002,money-fx|trade
1003,none
1004,earn|trade|grain
```

---

*Built with 💻 at CuseHacks Datathon 2026 · Syracuse University*
