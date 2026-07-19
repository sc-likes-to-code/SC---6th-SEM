from sklearn.neural_network import MLPClassifier

X = [[0,0],
     [0,1],
     [1,0],
     [1,1]]

y = [0,1,1,1]

mlp = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='logistic',
    solver='lbfgs',
    max_iter=5000,
    random_state=1
)

mlp.fit(X, y)

print("Predictions:")
for i in X:
    print(i, "->", mlp.predict([i])[0])