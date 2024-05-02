import numpy as np
from tmu.models.classification.multioutput_classifier import TMCoalesceMultiOuputClassifier

noise = 0.1
number_of_features = 12
n_epochs = 200

X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_train = np.zeros((X_train.shape[0], 3), dtype=np.uint32)
Y_train[:, 0] = np.logical_xor(X_train[:, 0], X_train[:, 1]).astype(dtype=np.uint32)
Y_train[:, 1] = np.logical_and(X_train[:, 0], X_train[:, 1]).astype(dtype=np.uint32)
Y_train[:, 2] = np.logical_or(X_train[:, 0], X_train[:, 1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(5000, 3) <= noise, 1 - Y_train, Y_train)  # Adds noise

X_test = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)

Y_test = np.zeros((X_test.shape[0], 3), dtype=np.uint32)
Y_test[:, 0] = np.logical_xor(X_test[:, 0], X_test[:, 1]).astype(dtype=np.uint32)
Y_test[:, 1] = np.logical_and(X_test[:, 0], X_test[:, 1]).astype(dtype=np.uint32)
Y_test[:, 2] = np.logical_or(X_test[:, 0], X_test[:, 1]).astype(dtype=np.uint32)

tm = TMCoalesceMultiOuputClassifier(number_of_clauses=10, T=15, s=3.9)

accs = []

for epoch in range(n_epochs):
    tm.fit(X_train, Y_train, progress_bar=False)

    pred = tm.predict(X_test)

    acc = np.mean(pred == Y_test)
    accs.append(acc)

    print(f"{epoch=}, {acc=}")

print("Average acc = ", np.mean(accs))
