
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data_file = Path("data/news.csv")
df = pd.read_csv(data_file)

# Check dataset
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str).str.upper()

if df["label"].nunique() != 2 or df["label"].value_counts().min() < 2:
    raise ValueError("Dataset must contain both REAL and FAKE labels.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# Create Machine Learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nMODEL TRAINED SUCCESSFULLY!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/fake_news_model.joblib")

print("\nModel saved in models/fake_news_model.joblib")
print("Note: Demo dataset is too small to measure real-world accuracy.")
