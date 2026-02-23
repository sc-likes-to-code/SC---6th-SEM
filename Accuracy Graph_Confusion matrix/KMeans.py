import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Map clusters to actual labels
mapped_labels = np.zeros_like(labels)

for i in range(3):
    mask = (labels == i)
    mapped_labels[mask] = mode(y[mask], keepdims=True)[0]

# Accuracy
acc = accuracy_score(y, mapped_labels)
print("Accuracy:", acc)

# -----------------------------
# Accuracy Graph ONLY
# -----------------------------
plt.figure()
plt.bar(["K-Means"], [acc])
plt.ylim(0,1)
plt.title("K-Means Accuracy")
plt.ylabel("Accuracy")
plt.grid()
plt.show()