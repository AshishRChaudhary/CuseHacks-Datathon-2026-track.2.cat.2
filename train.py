"""
train.py
========
Train the TF-IDF + Logistic Regression model for Track 2.
Saves all artifacts to model/ directory.

Usage:
    python train.py
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

# ==============================================================================
# PATHS
# ==============================================================================

TRAIN_PATH = "data/train.csv"
VAL_PATH   = "data/val.csv"
MODEL_PATH = "model/"

# ==============================================================================
# STOPWORDS + CLEAN TEXT
# ==============================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "we", "you", "he",
    "she", "they", "who", "which", "what", "how", "when", "where", "not",
    "no", "nor", "so", "yet", "both", "either", "each", "all", "any",
    "said", "also", "into", "than", "then", "there", "their", "they",
    "about", "after", "before", "between", "during", "through", "while",
    "as", "up", "out", "over", "under", "again", "further", "s", "re",
    "inc", "corp", "co", "ltd", "mln", "bln", "pct", "cts", "dlrs", "vs"
}

def clean_text(title, text):
    title    = title if isinstance(title, str) else ""
    text     = text  if isinstance(text,  str) else ""
    combined = title + " " + title + " " + text
    combined = combined.lower()
    combined = re.sub(r"[^a-z\s]", " ", combined)
    tokens   = combined.split()
    tokens   = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(MODEL_PATH, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH, dtype=str)
    val_df   = pd.read_csv(VAL_PATH,   dtype=str)

    print("Cleaning text...")
    train_df['clean']= train_df.apply(lambda r: clean_text(r['title'], r['text']), axis=1)
    val_df['clean']= val_df.apply(lambda r: clean_text(r['title'], r['text']), axis=1)
    train_df['label_list']= train_df['topics'].apply(lambda x: x.split('|'))
    val_df['label_list']= val_df['topics'].apply(lambda x: x.split('|'))

    print("Fitting TF-IDF..")
    tfidf = TfidfVectorizer(
        max_features=75000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X_train = tfidf.fit_transform(train_df['clean'])
    X_val   = tfidf.transform(val_df['clean'])

    print("Encoding labels...")
    mlb     = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_df['label_list'])
    y_val   = mlb.transform(val_df['label_list'])

    print("Training model...")
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            C=7.0,
            class_weight='balanced',
            solver='lbfgs',
            n_jobs=-1
        )
    )
    model.fit(X_train, y_train)
    print("✅ Model trained!")

    print("Tuning thresholds...")
    val_probs  = model.predict_proba(X_val)
    thresholds = np.zeros(len(mlb.classes_))

    for i in range(len(mlb.classes_)):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.01, 0.51, 0.01):
            preds = (val_probs[:, i] >= t).astype(int)
            f1    = f1_score(y_val[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        thresholds[i] = best_t
# Uncomment below to see per-label threshold tuning results and saving the model weights.
# models and heavy transformers are already trained and saved, so this is not necessary for inference 
# and can be time-consuming to run on every execution of train.py
'''    Y_pred = (val_probs >= thresholds).astype(int)
    micro  = f1_score(y_val, Y_pred, average='micro', zero_division=0)
    macro  = f1_score(y_val, Y_pred, average='macro', zero_division=0)
    print(f"   Micro F1 : {micro:.4f}")
    print(f"   Macro F1 : {macro:.4f}") 
    
    print("\nSaving model...")
    joblib.dump(model,      MODEL_PATH + "classifier.pkl")
    joblib.dump(tfidf,      MODEL_PATH + "tfidf.pkl")
    joblib.dump(mlb,        MODEL_PATH + "mlb.pkl")
    joblib.dump(thresholds, MODEL_PATH + "thresholds.pkl")
    print("✅ All artifacts saved to model/")   '''

if __name__ == "__main__":
    main()