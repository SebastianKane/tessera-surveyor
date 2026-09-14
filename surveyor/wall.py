"""wall.py — the learned wall: a stopping rule the floor taught.

The slime-mold in digitize.py grows every stone one pixel ring per round
and stops at a hand-tuned color delta. Here the rule that decides "take
this pixel or stop" is a small network trained on human-verified stone
outlines, and the survey is split into the three questions it actually
contains:

  IDENTIFY   one interior point per stone (a cell-segmentation model if
             one is available, else the gradient minima the mold uses)
  SPREAD     grow every stone from its point, all together, one ring per
             round; the WALL decides each candidate pixel from the stone's
             own point of view — six luminance samples along the outward
             normal relative to the stone's core, the core delta, and the
             gradient. Eight numbers, 833 parameters.
  BOUNDARY   where two light stones collide, a second network reads the
             corridor between them from BOTH sides (the ridge profile
             along the line joining the two cores) and releases the pixels
             that are mortar. Light-on-light joints are where a one-sided
             rule is weakest; this is the only place the seam model runs.

No closure step: ground the wall refused stays refused. What the wall
calls mortar is the record's mortar.

Inference needs only numpy, scipy and Pillow. The weights ship as .npz
(models/wall.npz, models/seam.npz); training them needs scikit-learn and
lives in learn.py.
"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from digitize import render_svg, trace_boundary

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")
LOOK = np.arange(0, 6)          # look-ahead taps along the outward normal
SEAM_TAPS = (1, 2, 3, 4, 5)     # ridge profile taps along the A->B line
LIGHT = 160.0                   # a stone whose core is this bright is "light"


# ----------------------------------------------------------------- the nets
class Net:
    """A scaler + MLP (ReLU hidden layers, logistic output) in numpy."""

    def __init__(self, path):
        z = np.load(path)
        self.mean, self.scale = z["mean"], z["scale"]
        self.layers = []
        i = 0
        while f"W{i}" in z:
            self.layers.append((z[f"W{i}"], z[f"b{i}"]))
            i += 1
        self.n_params = int(sum(W.size + b.size for W, b in self.layers))

    def __call__(self, X):
        h = (np.asarray(X, np.float64) - self.mean) / self.scale
        for k, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if k < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return 1.0 / (1.0 + np.exp(-h[:, 0]))


def load_nets(wall_path=None, seam_path=None):
    wall = Net(wall_path or os.path.join(MODELS, "wall.npz"))
    sp = seam_path or os.path.join(MODELS, "seam.npz")
    seam = Net(sp) if os.path.exists(sp) else None
    return wall, seam


# ------------------------------------------------------------- the features
def bilinear(img, ys, xs):
    H, W = img.shape
    yi = np.clip(np.floor(ys).astype(int), 0, H - 2)
    xi = np.clip(np.floor(xs).astype(int), 0, W - 2)
    fy = np.clip(ys - yi, 0, 1)
    fx = np.clip(xs - xi, 0, 1)
    return (img[yi, xi] * (1 - fy) * (1 - fx) + img[yi + 1, xi] * fy * (1 - fx)
            + img[yi, xi + 1] * (1 - fy) * fx + img[yi + 1, xi + 1] * fy * fx)


def planes(rgb):
    """Luminance, gradient magnitude, blue-minus-red cast, 3x3 roughness."""
    rgb = np.asarray(rgb, np.float32)
    lum = rgb.mean(2)
    grad = np.hypot(ndi.sobel(lum, 0), ndi.sobel(lum, 1))
    cast = rgb[:, :, 2] - rgb[:, :, 0]
    m = ndi.uniform_filter(lum, 3)
    tex = np.sqrt(np.maximum(ndi.uniform_filter(lum * lum, 3) - m * m, 0))
    return lum, grad, cast, tex


def outward_normal(claimed):
    """Unit field pointing away from the claimed region."""
    sd = ndi.distance_transform_edt(~claimed)
    gy, gx = np.gradient(ndi.gaussian_filter(sd, 1.0))
    nr = np.hypot(gy, gx) + 1e-6
    return gy / nr, gx / nr


def wall_features(lum, grad, uy, ux, ys, xs, core_lum):
    """What the wall sees, from the claiming stone's point of view."""
    prof = np.stack([bilinear(lum, ys + uy[ys, xs] * t, xs + ux[ys, xs] * t)
                     for t in LOOK], 1) - core_lum[:, None]
    return np.hstack([prof, np.abs(lum[ys, xs] - core_lum)[:, None],
                      grad[ys, xs][:, None]])


def seam_features(lum, cast, tex, grad, mA, mB, ys, xs):
    """What the seam model sees: the corridor between stones A and B, read
    from both sides. All arrays are the same (sub)window."""
    dA = ndi.distance_transform_edt(~mA)
    dB = ndi.distance_transform_edt(~mB)
    gy, gx = np.gradient(ndi.gaussian_filter(dA - dB, 1.0))
    nr = np.hypot(gy, gx) + 1e-6
    uy, ux = (gy / nr)[ys, xs], (gx / nr)[ys, xs]
    cA, cB = lum[mA].mean(), lum[mB].mean()
    l0 = lum[ys, xs]
    cols = []
    for k in SEAM_TAPS:
        p = bilinear(lum, ys + uy * k, xs + ux * k)
        q = bilinear(lum, ys - uy * k, xs - ux * k)
        cols += [l0 - 0.5 * (p + q), p - cA, q - cB]
    cols += [l0 - max(cA, cB),
             tex[ys, xs] - 0.5 * (tex[mA].mean() + tex[mB].mean()),
             cast[ys, xs] - 0.5 * (cast[mA].mean() + cast[mB].mean()),
             grad[ys, xs]]
    return np.stack(cols, 1)


# ---------------------------------------------------------------- the steps
def gradient_minima(rgb, stone_px=16, keep=None, grout_gap=16):
    """The mold's locator: minima of the color-gradient field. `keep` is an
    optional boolean mask restricting where seeds may sit. A minimum whose
    color sits within `grout_gap` of the grout estimate (the mold's own
    minimum wall) is mortar, not stone, and is not a seed: a seed placed in
    mortar would grow mortar."""
    rgb = np.asarray(rgb, np.float64)
    smooth = np.stack([ndi.gaussian_filter(rgb[..., c], 1.1) for c in range(3)], -1)
    mag = np.zeros(rgb.shape[:2])
    for c in range(3):
        mag += np.abs(ndi.sobel(smooth[..., c], 0)) + np.abs(ndi.sobel(smooth[..., c], 1))
    mag = ndi.gaussian_filter(mag, 1.0)
    ridge = mag > np.percentile(mag, 82)
    grout = np.median(rgb[ridge], axis=0) if ridge.any() else np.array([110, 95, 82])
    fp = max(5, stone_px // 2)
    mins = ndi.minimum_filter(mag, size=fp) == mag
    if keep is not None:
        mins &= keep
    mins &= np.abs(smooth - grout).max(axis=2) >= grout_gap
    lab, n = ndi.label(mins)
    return lab, n


def interior_points(claims, min_px=25):
    """One seed per claimed region: its deepest interior pixel."""
    H, W = claims.shape
    markers = np.zeros((H, W), np.int32)
    n = 0
    for k, sl in enumerate(ndi.find_objects(claims), start=1):
        if sl is None:
            continue
        m = claims[sl] == k
        if m.sum() < min_px:
            continue
        d = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        iy, ix = np.unravel_index(np.argmax(d), d.shape)
        n += 1
        markers[iy + sl[0].start, ix + sl[1].start] = n
    return markers, n


def identify_with_model(im, model_path=None, gpu=False, win=768,
                        margin=64, scale=0.5, stone_px=16):
    """Cell-segmentation claims, windowed and stitched as in fuse.py.
    Returns a label map of claims (0 = the model said nothing)."""
    from cellpose import models  # noqa: F401  (ImportError is the caller's)
    kw = {"gpu": gpu}
    if model_path:
        kw["pretrained_model"] = os.path.expanduser(model_path)
    model = models.CellposeModel(**kw)
    arr = np.asarray(im)
    H, W = arr.shape[:2]
    crops, boxes = [], []
    for y0 in range(0, H, win):
        for x0 in range(0, W, win):
            ty0, tx0 = max(0, y0 - margin), max(0, x0 - margin)
            ty1, tx1 = min(H, y0 + win + margin), min(W, x0 + win + margin)
            ch, cw = ty1 - ty0, tx1 - tx0
            crop = Image.fromarray(arr[ty0:ty1, tx0:tx1]).resize(
                (max(1, int(cw * scale)), max(1, int(ch * scale))), Image.LANCZOS)
            crops.append(np.asarray(crop))
            boxes.append((tx0, ty0, x0, y0, min(W, x0 + win), min(H, y0 + win), cw, ch))
    out = model.eval(crops)
    masks = out[0] if isinstance(out, (tuple, list)) else out
    kept = np.zeros((H, W), np.int64)
    n = 0
    for mk, (tx0, ty0, cx0, cy0, cx1, cy1, cw, ch) in zip(masks, boxes):
        mk = np.asarray(Image.fromarray(np.asarray(mk).astype(np.int32), mode="I")
                        .resize((cw, ch), Image.NEAREST)).astype(np.int64)
        for sl_k, sl in enumerate(ndi.find_objects(mk), start=1):
            if sl is None:
                continue
            ys_, xs_ = np.nonzero(mk[sl] == sl_k)
            if len(ys_) == 0:
                continue
            ys_ = ys_ + sl[0].start
            xs_ = xs_ + sl[1].start
            my, mx = ys_.mean() + ty0, xs_.mean() + tx0
            if not (cy0 <= my < cy1 and cx0 <= mx < cx1):
                continue
            n += 1
            kept[ys_ + ty0, xs_ + tx0] = n
    # the model's one confabulation mode is giant: drop claims above the
    # tessera ceiling so their ground is seeded from the field instead
    sizes = np.bincount(kept.ravel())
    giants = np.nonzero(sizes > (3.5 * stone_px) ** 2)[0]
    giants = giants[giants > 0]
    if len(giants):
        kept[np.isin(kept, giants)] = 0
    return kept, int(len(giants))


def spread(lum, grad, markers, n, wall, rounds=40, thresh=0.5):
    """Grow all stones together; the wall decides every candidate pixel."""
    lab = markers.astype(np.int32).copy()
    st3 = np.ones((3, 3), bool)
    lumf = lum.astype(np.float32)
    for _ in range(rounds):
        claimed = lab > 0
        frontier = (lab == 0) & ndi.binary_dilation(claimed, st3)
        if not frontier.any():
            break
        neigh = ndi.grey_dilation(lab, footprint=st3)
        ys, xs = np.nonzero(frontier)
        owner = neigh[ys, xs]
        tot = np.bincount(lab.ravel(), weights=lumf.ravel(), minlength=n + 1)
        cnt = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
        core = (tot / np.maximum(cnt, 1)).astype(np.float32)
        uy, ux = outward_normal(claimed)
        p = wall(wall_features(lumf, grad, uy, ux, ys, xs, core[owner]))
        take = p > thresh
        if not take.any():
            break
        lab[ys[take], xs[take]] = owner[take]
    return lab


def seam_pass(lum, cast, tex, grad, lab, seam, corr=4, thresh=0.3,
              light=LIGHT):
    """At every collision between two LIGHT stones, release the corridor
    pixels the seam model calls mortar. Windowed per pair."""
    H, W = lab.shape
    objs = ndi.find_objects(lab)
    released = np.zeros((H, W), bool)
    core = ndi.mean(lum, lab, np.arange(len(objs) + 1))
    checked = 0
    pad = corr + 3
    for k, sl in enumerate(objs, start=1):
        if sl is None or core[k] < light:
            continue
        w = (slice(max(0, sl[0].start - pad), min(H, sl[0].stop + pad)),
             slice(max(0, sl[1].start - pad), min(W, sl[1].stop + pad)))
        sub = lab[w]
        mA = sub == k
        if mA.sum() < 25:
            continue
        ring = ndi.binary_dilation(mA, np.ones((3, 3), bool)) & ~mA
        for j in np.unique(sub[ring]):
            if j <= k or core[j] < light:
                continue
            mB = sub == j
            if mB.sum() < 25:
                continue
            dA = ndi.distance_transform_edt(~mA)
            dB = ndi.distance_transform_edt(~mB)
            c = (mA | mB) & (dA <= corr) & (dB <= corr)
            ys, xs = np.nonzero(c)
            if len(ys) < 4:
                continue
            F = seam_features(lum[w], cast[w], tex[w], grad[w], mA, mB, ys, xs)
            p = seam(F)
            rel = p < thresh
            released[ys[rel] + w[0].start, xs[rel] + w[1].start] = True
            checked += 1
    lab = lab.copy()
    lab[released] = 0
    # a release can orphan a sliver: each stone keeps its largest piece
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == k
        cc, ncc = ndi.label(m)
        if ncc > 1:
            big = np.argmax(np.bincount(cc.ravel())[1:]) + 1
            lab[sl][m & (cc != big)] = 0
    return lab, checked, int(released.sum())


def measure(lab, rgb, sources, min_px=25):
    """Per-stone record, same schema as the fused survey."""
    stones = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == k
        a = int(m.sum())
        if a < min_px:
            continue
        ys, xs = np.nonzero(m)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        tb = trace_boundary(m)
        if tb is None:
            continue
        poly = [[round(float(x + sl[1].start), 1), round(float(y + sl[0].start), 1)]
                for x, y in tb]
        sh = sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                 - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
        if sh < 0:
            poly.reverse()
        cy, cx = ys.mean(), xs.mean()
        yy, xx = ys - cy, xs - cx
        cov = np.array([[(xx * xx).mean(), (xx * yy).mean()],
                        [(xx * yy).mean(), (yy * yy).mean()]])
        evals, evecs = np.linalg.eigh(cov)
        stones.append({
            "poly": poly, "rgb": [int(c) for c in rgb[ys, xs].mean(0)],
            "area_px": a, "source": sources.get(k, "field"), "conf": None,
            "flags": [], "near_grout": False,
            "aspect": round(float(np.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9))), 3),
            "angle_deg": round(float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])) % 180), 1)})
    return stones


# ----------------------------------------------------------------- the survey
def digitize_learned(path, out_prefix, crop_frac=None, max_side=2200,
                     stone_px=16, min_stone_px=25, rounds=40,
                     source_note="", model=None, gpu=False,
                     wall_path=None, seam_path=None, no_seam=False,
                     render=True):
    """IDENTIFY -> SPREAD -> BOUNDARY. `model` is "auto" (use cellpose if
    importable), a path to fine-tuned cellpose weights, or None (seed from
    the gradient field only)."""
    wall, seam = load_nets(wall_path, seam_path)
    if no_seam:
        seam = None
    im = Image.open(path).convert("RGB")
    if crop_frac:
        fx0, fy0, fx1, fy1 = crop_frac
        im = im.crop((int(fx0 * im.width), int(fy0 * im.height),
                      int(fx1 * im.width), int(fy1 * im.height)))
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    rgb = np.asarray(im, np.float64)
    H, W = rgb.shape[:2]
    lum, grad, cast, tex = planes(rgb)

    # IDENTIFY
    sources = {}
    giants = 0
    markers = np.zeros((H, W), np.int32)
    n = 0
    if model:
        try:
            claims, giants = identify_with_model(
                im, None if model == "auto" else model, gpu=gpu, stone_px=stone_px)
        except ImportError:
            raise SystemExit(
                "learned survey: cellpose is not importable — install it "
                "(pip install cellpose) or drop --model to seed from the "
                "gradient field alone. Refusing to guess silently.")
        markers, n = interior_points(claims, min_px=min_stone_px)
        sources.update({k: "model" for k in range(1, n + 1)})
        print(f"identify: the model placed {n} points "
              f"({(claims > 0).mean():.0%} of the floor claimed, "
              f"{giants} giants dropped)", flush=True)
        lab = spread(lum, grad, markers, n, wall, rounds=rounds)
        far = ndi.distance_transform_edt(lab == 0) >= max(4, stone_px // 3)
        mins, nm = gradient_minima(rgb, stone_px, keep=far)
        for k, sl in enumerate(ndi.find_objects(mins), start=1):
            if sl is None:
                continue
            ys, xs = np.nonzero(mins[sl] == k)
            n += 1
            markers[ys[0] + sl[0].start, xs[0] + sl[1].start] = n
            sources[n] = "field"
        lab[(markers > 0) & (lab == 0)] = markers[(markers > 0) & (lab == 0)]
        print(f"identify: {nm} more points from the gradient field in the "
              f"silences", flush=True)
    else:
        mins, n = gradient_minima(rgb, stone_px)
        markers = mins.astype(np.int32)
        sources.update({k: "field" for k in range(1, n + 1)})
        lab = markers.copy()
        print(f"identify: {n} points from the gradient field", flush=True)

    # SPREAD
    lab = spread(lum, grad, lab, n, wall, rounds=rounds)
    print(f"spread: {(lab > 0).mean():.1%} of the floor claimed; the rest is "
          f"what the wall refused", flush=True)

    # BOUNDARY
    checked = released = 0
    if seam is not None:
        lab, checked, released = seam_pass(lum, cast, tex, grad, lab, seam)
        print(f"boundary: {checked} light-on-light collisions read from both "
              f"sides, {released} px released as mortar", flush=True)

    stones = measure(lab, rgb, sources, min_px=min_stone_px)
    grout = (np.median(rgb[lab == 0], axis=0) if (lab == 0).any()
             else np.array([110, 95, 82]))
    tess = {"format": "tessera-surveyor/1",
            "source_file": os.path.basename(path),
            "source_note": source_note,
            "method": "learned wall" + ("" if seam is None else " + seam"),
            "wall_params": wall.n_params,
            "seam_params": None if seam is None else seam.n_params,
            "analyzed_w": W, "analyzed_h": H,
            "crop_frac": list(crop_frac) if crop_frac else None,
            "grout_rgb": [int(c) for c in grout],
            "coverage": round(float((lab > 0).mean()), 3),
            "n_model_stones": sum(s["source"] == "model" for s in stones),
            "n_field_stones": sum(s["source"] == "field" for s in stones),
            "giant_dropped": giants,
            "seam_collisions": checked, "seam_released_px": released,
            "merged_flagged": 0, "joined": 0,
            "n_stones": len(stones), "stones": stones}
    with open(out_prefix + ".stones.json", "w") as f:
        json.dump(tess, f)
    if render:
        with open(out_prefix + "-digital.svg", "w") as f:
            f.write(render_svg(tess))
        im.save(out_prefix + "-photo.png")
        outp = rgb.copy()
        outp[lab == 0] = grout
        Image.fromarray(outp.astype(np.uint8)).save(out_prefix + "-pixels.png")
    print(f"{os.path.basename(path)}: {len(stones)} stones "
          f"(coverage {tess['coverage']:.0%}) -> {out_prefix}.stones.json",
          flush=True)
    return tess
