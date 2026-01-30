import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Download necessary NLTK resources
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['Category', 'Message']]  # Selecting only necessary columns
df.columns = ['label', 'message']

# Convert labels to numerical (spam = 1, ham = 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  # Apply stemming
    return ' '.join(words)

# Apply preprocessing
df['clean_message'] = df['message'].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label'], test_size=0.2, random_state=42)

# Build Pipeline (TF-IDF + Naïve Bayes)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB(alpha=0.1))  # Hyperparameter tuning with alpha
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, "spam_classifier.pkl")

# Make predictions
y_pred = pipeline.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Load & Test Model
loaded_model = joblib.load("spam_classifier.pkl")
sample_text = ["Congratulations! You've won a free iPhone. Click here to claim."]
prediction = loaded_model.predict(sample_text)[0]
print(f"Sample Prediction: {'Spam' if prediction == 1 else 'Ham'}")
