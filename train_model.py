import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset Loaded Successfully!")
print("Total Messages:", len(data))

# Step 2: Separate messages and labels
X = data["message"]
y = data["label"]

# Step 3: Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Messages:", len(X_train))
print("Testing Messages:", len(X_test))

# Step 4: Convert text into numerical features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Create and train the SVM model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

print("Model Training Completed!")

# Step 6: Evaluate the model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("SVM Accuracy:", round(accuracy * 100, 2), "%")

# Step 7: Save the trained model
joblib.dump(model, "spam_model.pkl")

# Step 8: Save the text vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model saved successfully!")
print("Vectorizer saved successfully!")