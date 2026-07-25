import csv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load email dataset
texts = []
labels = []

with open("emails.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        texts.append(row["text"])
        labels.append(row["label"])


# 2. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)


# 3. Convert email text into numerical features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 4. Create and train Machine Learning model
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)


# 5. Make predictions
y_pred = model.predict(X_test_tfidf)


# 6. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("PHISHING EMAIL DETECTION MODEL")
print("===================================")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# 7. Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# 8. Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 9. Display confusion matrix
plt.figure(figsize=(6, 4))

plt.imshow(cm)

plt.title("Phishing Email Detection - Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

# Add numbers inside the matrix
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.xticks(
    range(len(set(labels))),
    sorted(set(labels))
)

plt.yticks(
    range(len(set(labels))),
    sorted(set(labels))
)

plt.tight_layout()

plt.show()


# 10. Test a new email
new_email = [
    "Congratulations! You have won a free prize. Click the link to claim it."
]

new_email_tfidf = vectorizer.transform(new_email)

prediction = model.predict(new_email_tfidf)


# 11. Display prediction
print("\nNew Email Prediction:", prediction[0])

print("\n===================================")
print("Project Completed Successfully!")
print("===================================")