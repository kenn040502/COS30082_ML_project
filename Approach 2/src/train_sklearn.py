# src/train_sklearn.py
import argparse
import os
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV


def build_logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            multi_class="multinomial",
            solver="lbfgs",
        )),
    ])


def build_linsvc():
    base = LinearSVC(dual="auto", max_iter=10000)
    calibrated = CalibratedClassifierCV(
        estimator=base,
        cv=3,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", calibrated),
    ])


def build_knn():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=5,
        )),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True,
                        help="path to features/train.npz")
    parser.add_argument("--out_dir", required=True,
                        help="directory to save model files, for example weights")
    args = parser.parse_args()

    data = np.load(args.train)
    X = data["X"]
    y = data["y"].astype(int)
    classes = np.unique(y)

    print(f"[INFO] Training data: X={X.shape}, y={y.shape}")

    models = {
        "logreg": build_logreg(),
        "linsvc": build_linsvc(),
        "knn": build_knn(),
    }

    os.makedirs(args.out_dir, exist_ok=True)

    # train and save each classifier
    for name, pipe in models.items():
        print(f"[TRAIN] Fitting {name} ...")
        pipe.fit(X, y)

        pack = {
            "model": pipe,
            "classes": classes,
        }
        out_path = os.path.join(args.out_dir, f"{name}.pkl")
        joblib.dump(pack, out_path)
        print(f"[SAVE] {name} -> {out_path}")

    # keep a best model copy (logreg) for backward compatibility
    best_pack = {
        "model": models["logreg"],
        "classes": classes,
    }
    best_path = os.path.join(args.out_dir, "sklearn_model.pkl")
    joblib.dump(best_pack, best_path)
    print(f"[SAVE] best (logreg) -> {best_path}")


if __name__ == "__main__":
    main()
