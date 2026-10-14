import pandas as pd

# Load the SMS dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display the first 5 messages
print(data.head())
print("\nDataset Shape:", data.shape)

print("\nLabel Counts:\n", data["label"].value_counts())
from sklearn.model_selection import train_test_split

# Separate messages and labels
X = data["message"]
y = data["label"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text messages into numerical features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay



# Create the Machine Learning model
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict the labels of testing messages
y_pred = model.predict(X_test_tfidf)

# Check model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Display detailed performance
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Compare three Machine Learning algorithms

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

print("\nAlgorithm Comparison Results:\n")

for name, algorithm in models.items():
    # Train the algorithm
    algorithm.fit(X_train_tfidf, y_train)

    # Predict test messages
    predictions = algorithm.predict(X_test_tfidf)

    # Calculate accuracy
    score = accuracy_score(y_test, predictions)

    print(name, "Accuracy:", round(score * 100, 2), "%")

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

print("\nConfusion Matrix:\n")
print(cm)

# Display the confusion matrix as a graph
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

display.plot()

import matplotlib.pyplot as plt
plt.show()

# Test the model with new messages
new_messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hi, how are you? Are we meeting today?",
    "URGENT! You have won cash. Call now to claim your reward!",
    "Please send me the class notes."
]

# Convert new messages into TF-IDF features
new_messages_tfidf = vectorizer.transform(new_messages)

# Predict spam or ham
predictions = model.predict(new_messages_tfidf)

# Display the results
print("\nNew Message Predictions:\n")

for message, prediction in zip(new_messages, predictions):
    print("Message:", message)
    print("Prediction:", prediction.upper())
    print("-" * 50) 

    # Predict a message entered by the user
print("\nInteractive Spam Classifier")

message = input("Enter a message: ")

message_tfidf = vectorizer.transform([message])

prediction = model.predict(message_tfidf)[0]

print("Prediction:", prediction.upper())
