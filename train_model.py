import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("dataset/fake_jobs.csv")

# Combine job details
data["text"] = data["title"] + " " + data["company"] + " " + data["description"]

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])

# Target value
y = data["label"]

# Train model
model = LogisticRegression()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "fake_job_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")