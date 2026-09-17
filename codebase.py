Appendix A: Representative Code Snippets
The following listings present the core pipeline code referenced throughout Chapter Four, broken down by module, each with explanatory comments describing its role and key design decisions.

A.i Data Loading (data_loader module)
import pandas as pd
 
def load_dataset(file_paths):
    """
    Loads and merges the five YouTube Spam Collection CSV files
    into a single DataFrame, retaining only the CONTENT (raw
    comment text) and CLASS (0 = ham, 1 = spam) columns needed
    for this project's text-based classification task.
    """
    frames = [pd.read_csv(path) for path in file_paths]
    df = pd.concat(frames, ignore_index=True)
    return df[['CONTENT', 'CLASS']]
 
FILES = [
    "Youtube01-Psy.csv",
    "Youtube02-KatyPerry.csv",
    "Youtube03-LMFAO.csv",
    "Youtube04-Eminem.csv",
    "Youtube05-Shakira.csv",
]
df = load_dataset(FILES)
print(f"Total comments loaded: {len(df)}")
print(df['CLASS'].value_counts())

A.ii Text Preprocessing (preprocessor module)
import re
 
def clean_text(text):
    """
    Cleans a single raw YouTube comment string in preparation for
    TF-IDF feature extraction.
 
    Steps:
      1. Lowercase the entire comment, so that "FREE" and "free"
         map to the same token.
      2. Remove URLs (http/https links and www. addresses), since
         spam comments frequently embed links that would otherwise
         inflate the vocabulary with near-unique, non-generalizable
         tokens.
      3. Remove any character that is not a lowercase letter or
         whitespace (punctuation, digits, emoji), while preserving
         word content as fully as possible.
      4. Collapse repeated whitespace into a single space and trim
         leading/trailing whitespace.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
# Example usage:
# clean_text("Check out my new channel!! http://example.com/spam")
# -> "check out my new channel http example com spam"
#    (URL text remaining after stripping the scheme is retained,
#     but carries little weight once TF-IDF's inverse document
#     frequency weighting is applied, since such fragments rarely
#     recur verbatim across multiple comments.)

A.iii Feature Extraction (feature_extractor module)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
 
RANDOM_STATE = 42
 
df['clean_content'] = df['CONTENT'].apply(clean_text)
X, y = df['clean_content'], df['CLASS']
 
# Stratified 80:20 split preserves the original spam/ham ratio
# in both the training and test subsets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
 
# The vectorizer is fit ONLY on the training set to avoid data
# leakage (Chapter Three, Section 3.4; Chapter Four, Section 4.4).
vectorizer = TfidfVectorizer(
    stop_words='english',   # remove common English stopwords
    ngram_range=(1, 2),      # unigrams AND bigrams (Chapter Two, 2.2.3)
    max_features=3000        # cap vocabulary size given dataset size
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
 
print(f"Training feature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

A.iv Model Training and Evaluation (model_trainer / evaluator modules)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
 
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "SVM (linear kernel)": SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}
 
results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

A.v Feature-Level Analysis (Chapter Five, Section 5.2)
import numpy as np
 
# Random Forest feature importance (Chapter Five, Section 5.2)
rf = models["Random Forest"]
feat_names = np.array(vectorizer.get_feature_names_out())
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-15:][::-1]
print("Top 15 Random Forest features:")
for i in top_idx:
    print(f"  {feat_names[i]}: {importances[i]:.4f}")
 
# SVM linear coefficients (Chapter Five, Section 5.2)
svm = models["SVM (linear kernel)"]
coefs = svm.coef_.toarray()[0]
top_spam_idx = np.argsort(coefs)[-10:][::-1]
top_ham_idx = np.argsort(coefs)[:10]
print("\nTop spam-indicative terms (SVM):")
for i in top_spam_idx:
    print(f"  {feat_names[i]}: {coefs[i]:.4f}")
print("\nTop ham-indicative terms (SVM):")
for i in top_ham_idx:
    print(f"  {feat_names[i]}: {coefs[i]:.4f}")

A.vii Demonstration / Prediction Function (predictor module)
def predict_comment(raw_comment, model, vectorizer):
    """
    Output/Demo Layer (Chapter Three, Section 3.3): classifies a
    single new, raw comment string using an already-trained model
    and its associated fitted vectorizer.
    """
    cleaned = clean_text(raw_comment)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return "spam" if prediction == 1 else "ham"
 
# Example usage:
# predict_comment("Check out my channel and subscribe!!", 
#                  models["Random Forest"], vectorizer)
# -> "spam"
# predict_comment("This song is amazing, love it!", 
#                  models["Random Forest"], vectorizer)
# -> "ham"
