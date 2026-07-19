from sklearn.neural_network import MLPClassifier

# Training data
X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

# AND gate output
y = [0,0,0,1]

# Create MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='logistic',
    solver='lbfgs',
    max_iter=5000,
    random_state=1
)

# Train the model
mlp.fit(X, y)

# Test
print("Predictions:")
for i in X:
    print(i, "->", mlp.predict([i])[0])