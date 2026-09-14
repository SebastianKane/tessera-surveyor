"""learn.py — train the wall and the seam from verified outlines.

Input: a photograph, a stones file for it (the survey whose outlines were
judged), and a verdicts file mapping stone id -> verdict code. A stone
whose code is exactly "p" (pass; a trailing "c" for "cracked, correctly
two" is ignored) is a verified outline. Nothing else is used.

    learn PHOTO STONES.json VERDICTS.json [--out models/]

Writes wall.npz and seam.npz, plus held-out AUC by luminance band so the
number is honest before anything ships. Needs scikit-learn.

The wall is trained on the ring one step ahead of a front placed at
depth 0..3 inside each verified stone — depth 0 is the whole stone, so
the ring is the first mortar, and without it the set is 94% "take" and
the wall learns never to stop. The seam is trained on corridors between
adjacent verified stones, read from both sides; it is held out by
connected component of the adjacency graph so a shared stone never sits
on both sides of a fold.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wall import (LIGHT, outward_normal, planes, seam_features,  # noqa: E402
                  wall_features)


def export_npz(scaler, mlp, path):
    z = {"mean": scaler.mean_, "scale": scaler.scale_}
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        z[f"W{i}"] = W
        z[f"b{i}"] = b
    np.savez(path, **z)


def verified_masks(stones, verdicts, W, H):
    out = []
    for st in stones:
        v = verdicts.get(str(st.get("id", "")))
        if v is None or v.replace("c", "") != "p":
            continue
        m = Image.new("1", (W, H), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in st["poly"]], fill=1)
        m = np.asarray(m, bool)
        if m.sum() >= 80:
            out.append(m)
    return out


def wall_set(lum, grad, masks):
    X, Y, G, B = [], [], [], []
    for gi, m in enumerate(masks):
        for kk in (0, 1, 2, 3):
            front = m if kk == 0 else ndi.binary_erosion(m, iterations=kk)
            if front.sum() < 20:
                continue
            cand = ndi.binary_dilation(front, np.ones((3, 3), bool)) & ~front
            ys, xs = np.nonzero(cand)
            if len(ys) < 20:
                continue
            uy, ux = outward_normal(front)
            core = np.full(len(ys), lum[front].mean(), np.float32)
            X.append(wall_features(lum, grad, uy, ux, ys, xs, core))
            Y.append(m[ys, xs].astype(int))
            G.append(np.full(len(ys), gi))
            B.append(np.full(len(ys), lum[m].mean()))
    return [np.concatenate(a) for a in (X, Y, G, B)]


def seam_set(lum, cast, tex, grad, masks, corr=4):
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append((ys.min(), ys.max(), xs.min(), xs.max()))
    pairs = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            bi, bj = boxes[i], boxes[j]
            if (bi[0] > bj[1] + 2 * corr or bj[0] > bi[1] + 2 * corr
                    or bi[2] > bj[3] + 2 * corr or bj[2] > bi[3] + 2 * corr):
                continue
            pairs.append((i, j))
    parent = list(range(len(masks)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    X, Y, G, B = [], [], [], []
    for i, j in pairs:
        mA, mB = masks[i], masks[j]
        dA = ndi.distance_transform_edt(~mA)
        dB = ndi.distance_transform_edt(~mB)
        c = (dA <= corr) & (dB <= corr)
        ys, xs = np.nonzero(c)
        if len(ys) < 10:
            continue
        y = (mA | mB)[ys, xs].astype(int)
        if y.min() == y.max():
            continue
        parent[find(i)] = find(j)
        X.append(seam_features(lum, cast, tex, grad, mA, mB, ys, xs))
        Y.append(y)
        G.append(np.full(len(ys), i))
        B.append(np.full(len(ys), max(lum[mA].mean(), lum[mB].mean())))
    G = np.concatenate(G)
    G = np.array([find(int(g)) for g in G])
    return np.concatenate(X), np.concatenate(Y), G, np.concatenate(B)


def fit_report(X, Y, G, B, tag):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    P = np.zeros(len(Y))
    folds = min(5, len(np.unique(G)))
    for tr, te in GroupKFold(folds).split(X, Y, G):
        sc = StandardScaler().fit(X[tr])
        m = MLPClassifier((32, 16), max_iter=1000, random_state=0).fit(sc.transform(X[tr]), Y[tr])
        P[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    bands = ((0, 110, "dark"), (110, LIGHT, "mid"), (LIGHT, 256, "light"))
    parts = []
    for lo, hi, nm in bands:
        s = (B >= lo) & (B < hi)
        if s.sum() and Y[s].min() != Y[s].max():
            parts.append(f"{nm} {roc_auc_score(Y[s], P[s]):.3f}")
    print(f"{tag}: {len(Y)} decisions, held-out AUC by band: " + " | ".join(parts), flush=True)
    sc = StandardScaler().fit(X)
    mlp = MLPClassifier((32, 16), max_iter=1000, random_state=0).fit(sc.transform(X), Y)
    return sc, mlp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("stones")
    ap.add_argument("verdicts")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    a = ap.parse_args()
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("learn: scikit-learn is required to train (pip install scikit-learn)")
    rgb = np.asarray(Image.open(a.photo).convert("RGB"), np.float32)
    H, W = rgb.shape[:2]
    lum, grad, cast, tex = planes(rgb)
    S = json.load(open(a.stones))
    V = json.load(open(a.verdicts))
    V = V.get("v", V)
    masks = verified_masks(S["stones"], V, W, H)
    print(f"{len(masks)} verified outlines", flush=True)
    os.makedirs(a.out, exist_ok=True)
    sc, mlp = fit_report(*wall_set(lum, grad, masks), "wall")
    export_npz(sc, mlp, os.path.join(a.out, "wall.npz"))
    sc, mlp = fit_report(*seam_set(lum, cast, tex, grad, masks), "seam")
    export_npz(sc, mlp, os.path.join(a.out, "seam.npz"))
    print(f"wrote {a.out}/wall.npz and seam.npz", flush=True)


if __name__ == "__main__":
    main()
