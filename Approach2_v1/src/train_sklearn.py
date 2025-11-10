import argparse, numpy as np, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.base import clone

def proba_of(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    scores = clf.decision_function(X)
    if scores.ndim == 1:  # binary fallback
        scores = np.stack([-scores, scores], axis=1)
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cv_score(base_clf, X, y, splits=5, seed=42):
    t1s, t5s = [], []
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        clf = clone(base_clf)
        clf.fit(X[tr], y[tr])
        P = proba_of(clf, X[va])
        labels_order = getattr(clf, "classes_", None)
        pred_idx = P.argmax(1)
        pred_lab = labels_order[pred_idx]
        t1s.append(accuracy_score(y[va], pred_lab))
        t5s.append(top_k_accuracy_score(y[va], P, k=min(5, P.shape[1]), labels=labels_order))
    return float(np.mean(t1s)), float(np.mean(t5s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    D = np.load(args.train); X, y = D["X"], D["y"]

    models = {
        # fast, strong baseline
        "logreg": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs")
        ),
        # newer sklearn uses "estimator" instead of "base_estimator"
        "linsvc": CalibratedClassifierCV(
            estimator=LinearSVC(dual="auto", max_iter=10000),
            cv=3,
        ),
        # simple non-parametric
        "knn": KNeighborsClassifier(n_neighbors=5)
    }

    best_name, best_score, best_clf = None, (-1, -1), None
    for name, m in models.items():
        t1, t5 = cv_score(m, X, y)
        print(f"{name}: top1={t1:.4f} top5={t5:.4f}")
        if (t1, t5) > best_score:
            best_name, best_score, best_clf = name, (t1, t5), m

    print("best:", best_name, best_score)
    best_clf.fit(X, y)
    pack = {"model": best_clf, "cv": {best_name: best_score}}
    # persist class order
    classes = getattr(best_clf, "classes_", None)
    if classes is None and hasattr(best_clf, "named_steps"):
        # try to find first step that has classes_
        for step in best_clf.named_steps.values():
            if hasattr(step, "classes_"):
                classes = step.classes_
                break
    if classes is not None:
        pack["classes"] = classes
    joblib.dump(pack, args.out)
    print("saved", args.out)

if __name__ == "__main__":
    main()
