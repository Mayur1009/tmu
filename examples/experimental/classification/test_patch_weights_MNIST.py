import argparse
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from tmu.data import MNIST
from tmu.experimental.models.multioutput_classifier import (
    TMCoalesceMultiOuputClassifier,
)
from tqdm import tqdm


def scale(X, x_min, x_max):
    nom = (X - X.min()) * (x_max - x_min)
    denom = X.max() - X.min()
    denom = denom + (denom == 0)
    return x_min + nom / denom


def dataset_mnist():
    data = MNIST().get()
    xtrain_orig, xtest_orig, ytrain_orig, ytest_orig = (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
    )

    xtrain = xtrain_orig.reshape(-1, 28, 28)
    xtest = xtest_orig.reshape(-1, 28, 28)

    ytrain = np.zeros((ytrain_orig.shape[0], 10), dtype=int)
    for i in range(ytrain_orig.shape[0]):
        ytrain[i, ytrain_orig[i]] = 1
    ytest = np.zeros((ytest_orig.shape[0], 10), dtype=int)
    for i in range(ytest_orig.shape[0]):
        ytest[i, ytest_orig[i]] = 1

    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    original = (xtrain_orig, ytrain_orig, xtest_orig, ytest_orig)
    return original, xtrain, ytrain, xtest, ytest, label_names


def metrics(true, pred):
    met = {
        "Subset accuracy": accuracy_score(true, pred),
        "Hamming loss": hamming_loss(true, pred),
        "F1 score": f1_score(true, pred, average="weighted"),
        "Precision": precision_score(true, pred, average="weighted"),
        "Recall": recall_score(true, pred, average="weighted"),
    }
    return met


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clauses", default=2000, type=int)
    parser.add_argument("--T", default=3125, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--q", default=-1, type=float)
    parser.add_argument("--type_ratio", default=1.0, type=float)
    parser.add_argument("--platform", default="GPU", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--patch", default=10, type=int)
    args = parser.parse_args()

    params = dict(
        number_of_clauses=args.clauses,
        T=args.T,
        s=args.s,
        q=args.q,
        type_i_ii_ratio=args.type_ratio,
        patch_dim=(args.patch, args.patch),
        platform=args.platform,
        seed=10,
    )

    original, xtrain, ytrain, xtest, ytest, label_names = dataset_mnist()

    tm = TMCoalesceMultiOuputClassifier(**params)

    print("Training with params: ")
    pprint(params)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        tm.fit_init(xtrain, ytrain)
        tm.fit_batch(xtrain, ytrain, progress_bar=True)
        # tm.fit(xtrain, ytrain, progress_bar=True)
        pred = tm.predict(xtest, progress_bar=True)

        met = metrics(ytest, pred)
        rep = classification_report(ytest, pred, target_names=label_names)

        pprint(met)
        print(rep)
        print("------------------------------")

    num_classes = 10
    num_clauses = tm.number_of_clauses
    num_patches = tm.clause_bank.number_of_patches

    literals = tm.clause_bank.get_literals()

    ppolarity = np.zeros((10, 3, 28, 28))
    npolarity = np.zeros((10, 3, 28, 28))

    for c in range(num_classes):
        weights = tm.weight_banks[c].get_weights()
        patch_weights = tm.weight_banks[c].get_patch_weights()

        for clause_ind in tqdm(range(num_clauses)):
            pos_lit = literals[clause_ind, 18 + 18 : 18 + 18 + 100].reshape((10, 10))
            neg_lit = literals[clause_ind, 18 + 18 + 100 + 18 + 18 :].reshape((10, 10))

            pimg = np.zeros((3, 28, 28))
            nimg = np.zeros((3, 28, 28))
            pws = patch_weights[clause_ind]
            pws = pws.reshape(19, 19)

            for m in range(19):
                for n in range(19):
                    if weights[clause_ind] > 0:
                        pimg[0, m : m + 10, n : n + 10] += pos_lit * pws[m, n]
                        pimg[1, m : m + 10, n : n + 10] += neg_lit * pws[m, n]
                        pimg[2, m : m + 10, n : n + 10] += (pos_lit - neg_lit) * pws[
                            m, n
                        ]
                    else:
                        # Negative polarity case
                        nimg[0, m : m + 10, n : n + 10] += pos_lit * pws[m, n]
                        nimg[1, m : m + 10, n : n + 10] += neg_lit * pws[m, n]
                        nimg[2, m : m + 10, n : n + 10] += (pos_lit - neg_lit) * pws[
                            m, n
                        ]

            ppolarity[c, 0] = ppolarity[c, 0] + (pimg[0] - ppolarity[c, 0]) / (
                num_clauses + 1
            )
            ppolarity[c, 1] = ppolarity[c, 1] + (pimg[1] - ppolarity[c, 1]) / (
                num_clauses + 1
            )
            ppolarity[c, 2] = ppolarity[c, 2] + (pimg[2] - ppolarity[c, 2]) / (
                num_clauses + 1
            )
            npolarity[c, 0] = npolarity[c, 0] + (nimg[0] - npolarity[c, 0]) / (
                num_clauses + 1
            )
            npolarity[c, 1] = npolarity[c, 1] + (nimg[1] - npolarity[c, 1]) / (
                num_clauses + 1
            )
            npolarity[c, 2] = npolarity[c, 2] + (pimg[2] - npolarity[c, 2]) / (
                num_clauses + 1
            )

    for c in range(num_classes):
        fig, axs = plt.subplots(
            2, 3, squeeze=False, figsize=(7, 7), layout="compressed"
        )
        for x in axs.ravel():
            x.axis("off")
        axs[0, 0].imshow(scale(ppolarity[c, 0], 0, 1))
        axs[0, 1].imshow(scale(ppolarity[c, 1], 0, 1))
        axs[0, 2].imshow(scale(ppolarity[c, 2], 0, 1))
        axs[1, 0].imshow(scale(npolarity[c, 0], 0, 1))
        axs[1, 1].imshow(scale(npolarity[c, 1], 0, 1))
        axs[1, 2].imshow(scale(npolarity[c, 2], 0, 1))
        # fig.savefig(f"~/class_{c}.png", bbox_inches="tight")
    plt.show()
