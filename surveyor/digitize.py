#!/usr/bin/env python3
"""digitize.py — take a photograph of a real mosaic, make its tessellation
digital.

Segment the stones, then MEASURE each one — boundary polygon, mean color,
area, aspect, orientation — and write the floor down as data. Proof of
faithfulness: the floor re-rendered from its own data file alone, beside
the photograph. If the re-render doesn't look like the floor, the data
doesn't say the floor, and no downstream use can fix that.

Method: classic computer vision, no learning and nothing to download.
Brightness thresholds are half-blind — dark stones read as grout — so
segmentation is by HOMOGENEITY instead: stones of any value are smooth
basins in the color-gradient field, and the grout lines are the ridges
between them. Minimum-filter markers in the basins, then watershed on the
gradient image.

Known honest limits, stated rather than hidden:
  - pale-on-pale regions are unreliable at modest resolution;
  - stones separated by sub-pixel grout merge into one basin. Merges are
    DETECTED (a stone whose convex hull dwarfs its pixel count is a merge),
    counted, and excluded — never silently kept.
"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

FORMAT = "tessera-surveyor/0"


def digitize(path, out_prefix, crop_frac=None, max_side=2200, stone_px=16,
             min_stone_px=30, min_solidity=0.55, source_note=""):
    """Segment and measure one floor photograph.

    crop_frac    (x0, y0, x1, y1) as fractions of the image, applied FIRST
                 at native resolution — grout must stay super-pixel.
    stone_px     expected stone diameter in analyzed pixels; sets the
                 marker footprint and the merge ceiling.
    Returns the tessellation dict (also written to <out_prefix>.stones.json,
    with the proof re-render at <out_prefix>-digital.svg and the analyzed
    photograph at <out_prefix>-photo.png).
    """
    im = Image.open(path).convert("RGB")
    if crop_frac:
        fx0, fy0, fx1, fy1 = crop_frac
        im = im.crop((int(fx0 * im.width), int(fy0 * im.height),
                      int(fx1 * im.width), int(fy1 * im.height)))
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.float64)

    # the color-gradient field: stones are its basins, grout its ridges
    mag = np.zeros(rgb.shape[:2])
    for c in range(3):
        ch = ndi.gaussian_filter(rgb[..., c], 1.2)
        mag += np.abs(ndi.sobel(ch, 0)) + np.abs(ndi.sobel(ch, 1))
    mag = ndi.gaussian_filter(mag, 1.0)
    fp = max(5, stone_px // 2)
    mins = ndi.minimum_filter(mag, size=fp) == mag
    markers, n = ndi.label(mins)
    cost = np.clip(mag * (255.0 / max(np.percentile(mag, 99), 1)), 0, 255)
    ws = ndi.watershed_ift(cost.astype(np.uint8), markers.astype(np.int32))

    # grout color: the high-gradient ridge pixels (boundary mortar zone)
    ridge = mag > np.percentile(mag, 80)
    grout_color = ([int(c) for c in np.median(rgb[ridge], axis=0)]
                   if ridge.any() else [110, 95, 82])

    stones = []
    merged = 0
    max_stone = int((3.5 * stone_px) ** 2)
    for sl, lab in zip(ndi.find_objects(ws), range(1, n + 1)):
        if sl is None:
            continue
        m = ws[sl] == lab
        a = int(m.sum())
        if a < min_stone_px or a > max_stone:
            continue
        # pull each basin back off the ridge: the raw boundary sits mid-grout
        m = ndi.binary_erosion(m, iterations=1)
        a = int(m.sum())
        if a < min_stone_px:
            continue
        ys, xs = np.nonzero(m)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        color = [int(c) for c in rgb[ys, xs].mean(axis=0)]
        pts = np.column_stack([xs, ys]).astype(float)
        try:
            hull = ConvexHull(pts)
            poly = [[round(float(x), 1), round(float(y), 1)]
                    for x, y in pts[hull.vertices]]
        except Exception:
            continue
        # merge gate: a stone whose hull dwarfs its pixels is several stones
        # joined by sub-pixel grout — count it, exclude it, never keep it
        if a / max(hull.volume, 1.0) < min_solidity:
            merged += 1
            continue
        cy, cx = ys.mean(), xs.mean()
        yy, xx = ys - cy, xs - cx
        cov = np.array([[(xx * xx).mean(), (xx * yy).mean()],
                        [(xx * yy).mean(), (yy * yy).mean()]])
        evals, evecs = np.linalg.eigh(cov)
        # near-grout flag: a region whose color sits at the measured grout
        # color is likely a joint fragment, not a stone — but "likely" is
        # not certainty (real floors lay stones in grout-colored minerals),
        # so it is FLAGGED for the consumer to filter, never dropped here.
        near_grout = max(abs(a - b)
                         for a, b in zip(color, grout_color)) < 22
        stones.append({
            "near_grout": near_grout,
            "poly": poly, "rgb": color, "area_px": a,
            "aspect": round(float(np.sqrt(max(evals[1], 1e-9)
                                          / max(evals[0], 1e-9))), 3),
            "angle_deg": round(float(np.degrees(
                np.arctan2(evecs[1, 1], evecs[0, 1])) % 180), 1),
        })

    tess = {"format": FORMAT,
            "source_file": os.path.basename(path),
            "source_note": source_note,
            "analyzed_w": im.width, "analyzed_h": im.height,
            "crop_frac": list(crop_frac) if crop_frac else None,
            "grout_rgb": grout_color,
            "method": "classic-CV homogeneity watershed "
                      "(pale-on-pale regions unreliable at modest resolution)",
            "merged_flagged": merged,
            "n_stones": len(stones), "stones": stones}
    with open(out_prefix + ".stones.json", "w") as f:
        json.dump(tess, f)

    # the proof: re-render the floor from its own data alone
    with open(out_prefix + "-digital.svg", "w") as f:
        f.write(render_svg(tess))
    im.save(out_prefix + "-photo.png")
    print(f"{os.path.basename(path)}: {len(stones)} stones digitized "
          f"({merged} merges flagged & excluded) -> {out_prefix}.stones.json")
    return tess


def render_svg(tess):
    """The floor, from its data file alone. This is the acceptance test a
    human can run with their eyes."""
    W, H = tess["analyzed_w"], tess["analyzed_h"]
    g = tess["grout_rgb"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
             f'<rect width="{W}" height="{H}" '
             f'fill="rgb({g[0]},{g[1]},{g[2]})"/>']
    for s in tess["stones"]:
        d = "M" + " L".join(f"{x},{y}" for x, y in s["poly"]) + " Z"
        r, gg, b = s["rgb"]
        parts.append(f'<path d="{d}" fill="rgb({r},{gg},{b})"/>')
    parts.append("</svg>")
    return "\n".join(parts)
