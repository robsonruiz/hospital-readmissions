import matplotlib.pyplot as plt

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def plot_roc(models, X_test, y_test):

    plt.figure()

    for name, model in models.items():

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name=name
        )

    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(
        "results/roc_curve.png",
        dpi=300
    )
    plt.close()


def plot_precision_recall(
    models,
    X_test,
    y_test
):

    plt.figure()

    for name, model in models.items():

        PrecisionRecallDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name=name
        )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        "results/precision_recall_curve.png",
        dpi=300
    )

    plt.close()