import numpy as np
from tmu.data import MNIST
from tmu.models.classification.multioutput_classifier import TMCoalesceMultiOuputClassifier

from sklearn.metrics import classification_report, accuracy_score


def get_data():
    data = MNIST().get()
    xtrain, xtest, ytrain, ytest = (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
    )
    x_train = xtrain.reshape(xtrain.shape[0], 28, 28)
    x_test = xtest.reshape(xtest.shape[0], 28, 28)
    n_classes = np.max(ytrain) + 1
    y_train = np.zeros((ytrain.shape[0], n_classes))
    y_test = np.zeros((ytest.shape[0], n_classes))
    for e in range(ytrain.shape[0]):
        y_train[e, ytrain[e]] = 1
    for e in range(ytest.shape[0]):
        y_test[e, ytest[e]] = 1
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    xtrain, xtest, ytrain, ytest = get_data()

    tm = TMCoalesceMultiOuputClassifier(
        number_of_clauses=2000,
        T=4000,
        s=10.0,
        patch_dim=(10, 10),
    )

    n_epochs = 1

    for epoch in range(n_epochs):
        print("EPOCH ", epoch+1)
        tm.fit(xtrain, ytrain, progress_bar=True)

        pred = tm.predict(xtest, progress_bar=True)

        acc = accuracy_score(ytest, pred)
        class_report = classification_report(ytest, pred)

        print(class_report)
