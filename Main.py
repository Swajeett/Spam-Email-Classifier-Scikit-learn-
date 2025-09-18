# Spam Email Classifier using Scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset (example: SMS Spam Collection dataset)
# Format: label (ham/spam), message
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels: ham -> 0, spam -> 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 3. Convert text to numerical features (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Predictions
y_pred = model.predict(X_test_tfidf)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Test with custom messages
test_emails = [
    "Congratulations! You won a $1000 gift card. Click here to claim now.",
    "Hey, are we still meeting tomorrow for lunch?"
]

test_tfidf = vectorizer.transform(test_emails)
predictions = model.predict(test_tfidf)

for email, label in zip(test_emails, predictions):
    print(f"\nEmail: {email}\nPrediction: {'Spam' if label == 1 else 'Not Spam'}")
