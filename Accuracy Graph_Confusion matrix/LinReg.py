import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediction
y_pred = lr.predict(X_test)

# Convert regression output to class labels
y_pred = np.round(y_pred).astype(int)

# Keep values within valid class range
y_pred = np.clip(y_pred, 0, 2)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# -----------------------------
# Accuracy Graph
# -----------------------------
plt.figure()
plt.bar(["Linear Regression"], [acc])
plt.ylim(0,1)
plt.title("Linear Regression Accuracy")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, cmap="Purples", fmt='d')
plt.title("Linear Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()