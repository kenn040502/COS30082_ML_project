import argparse
from pathlib import Path
import numpy as np
import joblib

def parse_species_list(path: Path):
    """
    Parses species_list.txt lines like:
      105951; Maripa glabra Choisy
    Returns:
      id_to_name[int_id] = "Maripa glabra Choisy"
    """
    id_to_name = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split on ';' exactly once
            if ";" in line:
                left, right = line.split(";", 1)
                left = left.strip()
                right = right.strip()
            else:
                # fallback: try first token as ID
                parts = line.split()
                if len(parts) < 2:
                    continue
                left = parts[0].strip()
                right = " ".join(parts[1:]).strip()

            if not left.isdigit():
                # skip header lines if any
                continue
            sid = int(left)
            id_to_name[sid] = right
    if not id_to_name:
        raise RuntimeError("Could not parse any species from species_list.txt")
    return id_to_name

def topk_acc(probs, true_cols, k):
    """
    probs: (M, C)
    true_cols: (M,) indices 0..C-1
    returns scalar accuracy@k
    """
    order = np.argsort(-probs, axis=1)[:, :k]  # top-k predicted class indices
    hits = (order == true_cols[:, None]).any(axis=1)
    return hits.mean() if hits.size > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", required=True, help="weights/sklearn_model.pkl")
    ap.add_argument("--train", required=True, help="features/train.npz")
    ap.add_argument("--test", required=True, help="features/test.npz")
    ap.add_argument("--species-list", required=True,
                    help="Herbarium_Field dataset/list/species_list.txt")
    ap.add_argument("--topk", nargs="+", type=int, default=[1,5])
    args = ap.parse_args()

    # --- load model ---
    pack = joblib.load(args.clf)
    clf = pack.get("model", pack)

    # classes the classifier was trained on (length C)
    if "classes" in pack:
        clf_classes = np.array(pack["classes"])
    elif hasattr(clf, "classes_"):
        clf_classes = np.array(clf.classes_)
    else:
        raise RuntimeError("No classes found in classifier")

    # --- load train / test embeddings ---
    train_data = np.load(args.train)
    X_train = train_data["X"]
    y_train = train_data["y"].astype(int)

    test_data = np.load(args.test)
    X_test = test_data["X"]
    y_test = test_data["y"].astype(int)

    # --- load species list (mainly for debugging / clarity) ---
    id_to_name = parse_species_list(Path(args.species_list))

    # Important:
    # y_train is what we used to fit the classifier.
    # clf_classes is the classifier's internal class ordering.
    # Usually, in scikit-learn, clf_classes is sorted unique(y_train) in ascending order.
    #
    # y_test might be DIFFERENT numeric ID space (e.g. original PlantCLEF IDs 105951, 106023, ...).
    # We need to know how to map y_test -> index in clf_classes.
    #
    # Strategy:
    # 1. Build mapping from actual species ID (the big ID like 105951) to the classifier column index.
    #    That works IFF y_train used those same big IDs.
    # 2. If y_train used remapped small IDs (0..N-1), we can't directly match y_test IDs,
    #    because then y_test==105951 will never equal 0..N-1.
    #
    # Let's detect which world we're in by checking:
    # - do any test IDs appear in clf_classes? if yes, great.
    # - if not, we're in "different ID worlds", which means we'd need ground-truth mapping we don't have.

    # map species_id -> column idx in classifier probs
    class_to_col = {int(cid): idx for idx, cid in enumerate(clf_classes)}

    overlapping_mask = []
    true_cols = []
    for i, lab in enumerate(y_test):
        if int(lab) in class_to_col:
            overlapping_mask.append(i)
            true_cols.append(class_to_col[int(lab)])

    overlapping_mask = np.array(overlapping_mask, dtype=int)
    true_cols = np.array(true_cols, dtype=int)

    if overlapping_mask.size == 0:
        # This means zero of the test labels (e.g. 105951, 106023...) appear in clf_classes.
        # That implies your train labels (y_train) were NOT those big IDs.
        #
        # That happens if build_features.py remapped training labels to 0..N-1,
        # but test.y is still the original big species IDs from groundtruth.txt.
        #
        # To evaluate properly, we need a mapping from each train remapped ID
        # back to the original big species ID. We don't have that stored yet.
        #
        # We'll emit debug info to help you dump a sample of y_train, y_test,
        # and clf_classes so we can build that mapping.
        print("No overlapping numeric IDs between test labels and classifier classes.")
        print("Let's dump some debug stats so we can build an ID map.")
        print("Sample y_train unique (first 20):", sorted(set(y_train.tolist()))[:20])
        print("Sample y_test  unique (first 20):", sorted(set(y_test.tolist()))[:20])
        print("clf_classes (first 20):", clf_classes[:20].tolist())

        # Also try to see if y_train looks like [0,1,2,...]
        train_unique = sorted(set(y_train.tolist()))
        print("Does y_train look like compact 0..K-1? First/last:", train_unique[:5], train_unique[-5:])

        # And try to map those compact IDs to species names using species_list.txt
        # if possible, just for inspection:
        mapped_names = [id_to_name.get(int(cid), None) for cid in train_unique[:20]]
        print("First few train IDs mapped via species_list.txt:", mapped_names)

        return

    # If we got here, we DO have overlap. We can evaluate accuracy now.

    # Get probability predictions for ALL test samples
    if hasattr(clf, "predict_proba"):
        P_all = clf.predict_proba(X_test)  # (N_test, C)
    else:
        S = clf.decision_function(X_test)
        if S.ndim == 1:
            S = np.stack([-S, S], axis=1)
        E = np.exp(S - S.max(axis=1, keepdims=True))
        P_all = E / E.sum(axis=1, keepdims=True)

    P_eval = P_all[overlapping_mask]  # select only overlapping species
    true_cols_eval = true_cols        # these are already col indices 0..C-1

    kept = len(true_cols_eval)
    skipped = len(y_test) - kept
    print(f"Evaluating on {kept}/{len(y_test)} test images; skipped {skipped} unseen-species images.")

    # compute top-k
    for k in args.topk:
        k_eff = min(k, P_eval.shape[1])
        acc_k = topk_acc(P_eval, true_cols_eval, k_eff)
        print(f"top-{k} accuracy: {acc_k:.4f}")

if __name__ == "__main__":
    main()
