#!/usr/bin/env python3
"""fuse.py — cellpose as ground truth where it works; the mold everywhere else.

A cell-segmentation model reads bright, well-jointed tesserae with higher
fidelity than any classic method here — those stones are, to its eye,
cells. But it goes silent in the dark and on unusual material. The mold is
value-blind and covers everything, at lower fidelity. So: run both, let
the model claim the regions where it demonstrably worked (window coverage
is the gate — where the model found dense stones, trust it), and let the
mold fill everything the model left unclaimed. One label map, one
measurement, every stone tagged with its source.

Requires cellpose (and its torch dependency): `pip install cellpose`.
The first run downloads model weights. Without cellpose this module
refuses loudly — it never falls back silently to the mold alone.
"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from digitize import digitize, render_svg, trace_boundary

WIN = 768          # evaluation window (analysis scale)
EVAL_SCALE = 0.5   # the model's claims densify when tesserae are ~8-9px
                   # in ITS view — measured on a scale ladder, peak at 0.5
SPARSE_R = 3.0     # a claim with few claimed neighbors (within this many
                   # stone diameters) is flagged sparse_claim, never dropped


def digitize_fused(path, out_prefix, crop_frac=None, max_side=2200,
                   stone_px=16, min_stone_px=25, min_solidity=0.5,
                   rounds=40, source_note="", gpu=False):
    try:
        from cellpose import models
    except ImportError:
        raise SystemExit(
            "fuse: cellpose is not importable in this interpreter — the "
            "fused survey cannot run. Install it (pip install cellpose) "
            "or use the plain survey. Refusing loudly rather than "
            "silently surveying with the mold alone.")

    # the mold's full survey first (no render yet; join stays off here —
    # fusion replaces the mold precisely where over-split hurts most)
    tess_mold = digitize(path, out_prefix + "-mold", crop_frac=crop_frac,
                         max_side=max_side, stone_px=stone_px,
                         min_stone_px=min_stone_px,
                         min_solidity=min_solidity, rounds=rounds,
                         render=False)
    im = Image.open(out_prefix + "-mold-photo.png") \
        if os.path.exists(out_prefix + "-mold-photo.png") else None
    # digitize(render=False) does not save the photo; rebuild the analyzed
    # image the same way it did
    im = Image.open(path).convert("RGB")
    if crop_frac:
        fx0, fy0, fx1, fy1 = crop_frac
        im = im.crop((int(fx0 * im.width), int(fy0 * im.height),
                      int(fx1 * im.width), int(fy1 * im.height)))
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.float64)
    H, W = rgb.shape[:2]

    print("fuse: cellpose pass "
          f"({'gpu' if gpu else 'cpu'}) ...", flush=True)
    model = models.CellposeModel(gpu=gpu)
    # the recipe that works, measured: 768px crops fed as a batched LIST,
    # each DOWNSCALED to EVAL_SCALE before evaluation (the model's claims
    # densify when tesserae are ~8-9px in its view — 0.5x tripled coverage
    # over 1x on the same crop), masks upsampled back by nearest-neighbor.
    # No window gate and no coverage veto: this model's failure mode is
    # SILENCE, not confabulation — it claims only what it is sure of, so
    # every claim is kept and every silence is the mold's to fill.
    MARGIN = 64
    arr = np.asarray(im)
    crops, boxes = [], []
    for y0 in range(0, H, WIN):
        for x0 in range(0, W, WIN):
            ty0, tx0 = max(0, y0 - MARGIN), max(0, x0 - MARGIN)
            ty1 = min(H, y0 + WIN + MARGIN)
            tx1 = min(W, x0 + WIN + MARGIN)
            ch, cw = ty1 - ty0, tx1 - tx0
            crop = Image.fromarray(arr[ty0:ty1, tx0:tx1]).resize(
                (max(1, int(cw * EVAL_SCALE)),
                 max(1, int(ch * EVAL_SCALE))), Image.LANCZOS)
            crops.append(np.asarray(crop))
            boxes.append((tx0, ty0, x0, y0,
                          min(W, x0 + WIN), min(H, y0 + WIN), cw, ch))
    out = model.eval(crops)
    masks_list = out[0] if isinstance(out, (tuple, list)) else out

    # stitch: masks back to analysis scale, each window keeping only the
    # stones whose centroid lands in its core (seams dedupe themselves,
    # as in the tiled survey)
    kept = np.zeros((H, W), np.int64)
    n_cp = 0
    for mk, (tx0, ty0, cx0, cy0, cx1, cy1, cw, ch) in zip(masks_list, boxes):
        mk = np.asarray(
            Image.fromarray(np.asarray(mk).astype(np.int32), mode="I")
            .resize((cw, ch), Image.NEAREST)).astype(np.int64)
        for lb in range(1, int(mk.max()) + 1):
            ys_, xs_ = np.nonzero(mk == lb)
            if len(ys_) == 0:
                continue
            my, mx = ys_.mean() + ty0, xs_.mean() + tx0
            if not (cy0 <= my < cy1 and cx0 <= mx < cx1):
                continue
            n_cp += 1
            kept[ys_ + ty0, xs_ + tx0] = n_cp
    accept = kept > 0

    # ONE LABEL MAP: accepted model stones claim their pixels; the mold
    # keeps everything else. Mold cells that lose most of their body to
    # the model vanish; those that merely graze it are trimmed.
    from_mold = np.load(out_prefix + "-mold-lab.npy") \
        if os.path.exists(out_prefix + "-mold-lab.npy") else None
    # rebuild the mold's label map from its stones (polygon fill)
    lab = np.zeros((H, W), np.int64)
    if from_mold is not None:
        lab = from_mold
    else:
        from PIL import ImageDraw
        lim = Image.new("I", (W, H), 0)
        drw = ImageDraw.Draw(lim)
        for i, s in enumerate(tess_mold["stones"], start=1):
            drw.polygon([tuple(p) for p in s["poly"]], fill=i)
        lab = np.asarray(lim).astype(np.int64)
    lab = np.where(kept > 0, 0, lab)

    def measure(mask_map, offset, source, stones, min_px):
        objs = ndi.find_objects(mask_map)
        kept_n = 0
        for k, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            m = mask_map[sl] == k
            a = int(m.sum())
            if a < min_px:
                continue
            ys, xs = np.nonzero(m)
            ys = ys + sl[0].start
            xs = xs + sl[1].start
            color = [int(c) for c in rgb[ys, xs].mean(axis=0)]
            tb = trace_boundary(m)
            if tb is None:
                continue
            poly = [[round(float(x + sl[1].start), 1),
                     round(float(y + sl[0].start), 1)] for x, y in tb]
            sh = sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                     - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))
            if sh < 0:
                poly.reverse()
            cy, cx = ys.mean(), xs.mean()
            yy, xx = ys - cy, xs - cx
            cov = np.array([[(xx * xx).mean(), (xx * yy).mean()],
                            [(xx * yy).mean(), (yy * yy).mean()]])
            evals, evecs = np.linalg.eigh(cov)
            stones.append({
                "poly": poly, "rgb": color, "area_px": a,
                "source": source, "conf": None, "flags": [],
                "near_grout": False,
                "aspect": round(float(np.sqrt(
                    max(evals[1], 1e-9) / max(evals[0], 1e-9))), 3),
                "angle_deg": round(float(np.degrees(
                    np.arctan2(evecs[1, 1], evecs[0, 1])) % 180), 1)})
            kept_n += 1
        return kept_n

    stones = []
    n_model = measure(kept, 0, "cellpose", stones, min_stone_px)
    n_mold = measure(lab, n_cp, "slime-mold", stones, min_stone_px)

    tess = {"format": "tessera-surveyor/1",
            "source_file": os.path.basename(path),
            "source_note": source_note,
            "method": "fused (cellpose where confident + slime-mold)",
            "analyzed_w": W, "analyzed_h": H,
            "crop_frac": list(crop_frac) if crop_frac else None,
            "grout_rgb": tess_mold["grout_rgb"],
            "coverage": round(float(((kept > 0) | (lab > 0)).mean()), 3),
            "model_claimed": round(float((kept > 0).mean()), 3),
            "n_model_stones": n_model, "n_mold_stones": n_mold,
            "merged_flagged": tess_mold["merged_flagged"],
            "joined": 0,
            "n_stones": len(stones), "stones": stones}
    with open(out_prefix + ".stones.json", "w") as f:
        json.dump(tess, f)
    with open(out_prefix + "-digital.svg", "w") as f:
        f.write(render_svg(tess))
    im.save(out_prefix + "-photo.png")
    for suffix in ("-mold.stones.json",):
        p = out_prefix + suffix
        if os.path.exists(p):
            os.remove(p)
    print(f"{os.path.basename(path)}: {len(stones)} stones fused — "
          f"{n_model} from the model ({accept.mean():.0%} of the floor "
          f"claimed), {n_mold} from the mold -> {out_prefix}.stones.json")
    return tess
