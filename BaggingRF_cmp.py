from sklearn.datasets import load_digits
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# Load the dataset
Mnist = load_digits()
# Set the number of estimators for both Bagging and Random Forest methods
est = 20 # or 2
print("Number of estimators:", est)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Mnist.data, Mnist.target, test_size=0.2, random_state=42)

# Create a Bagging classifier with decision tree as the base estimator
bagging = BaggingClassifier(n_estimators=est, random_state=42)

# Train the classifier on the training set
time_1 = time.time()
bagging.fit(X_train, y_train)
time_bagging = time.time() - time_1

# Predict on the testing set
bagging_pred = bagging.predict(X_test)

# Print the accuracy score of the Bagging classifier
print("Bagging accuracy:", bagging.score(X_test, y_test))
print("Bagging time:", time_bagging)
print('-'*50)
# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=est, random_state=42)

# Train the classifier on the training set
time_2 = time.time()
rf.fit(X_train, y_train)
time_rf = time.time() - time_2

# Predict on the testing set
rf_pred = rf.predict(X_test)

# Print the accuracy score of the Random Forest classifier
print("Random Forest accuracy:", rf.score(X_test, y_test))
print("Random Forest time:", time_rf)
