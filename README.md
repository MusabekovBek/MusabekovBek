import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=1)

# Create an AdaBoost classifier
ada = AdaBoostClassifier(base_estimator=clf, n_estimators=200, learning_rate=0.1, random_state=42)

# Train the AdaBoost classifier
ada.fit(X_train, y_train)

# Predict the test set
y_pred = ada.predict(X_test)

# Calculate the accuracy of the AdaBoost classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
