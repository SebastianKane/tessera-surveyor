#!/usr/bin/env python3
"""digitize.py — slime-mold the stones (Sebastian's algorithm, 08-10).

Locate each stone, then grow each tile outward from its center, absorbing
neighboring pixels while the color delta stays small; the moment it hits
grout, that's it — the wall. Properties by construction:
  - no overlap (every pixel claimed at most once, first-come by rounds)
  - true shapes (growth stops at the stone's actual edge in the image)
  - grout survives as the unclaimed lattice between stones
The delta wall is per-stone adaptive: a pale stone beside pale grout gets a
tighter tolerance than a black stone ever needs.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

def trace_boundary(mask):
    """Moore-neighbor trace: the stone's ACTUAL outline, not a hull.
    Returns ordered (x, y) points at direction changes; None on failure."""
    m = np.pad(mask, 1)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    sy, sx = int(ys.min()), int(xs[ys == ys.min()].min())
    # 8 neighbors clockwise from W
    dirs = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    pts, cur, entry = [], (sy, sx), 0
    cap = 8 * mask.sum() + 64
    prev_dir = None
    for _ in range(cap):
        found = False
        for k in range(8):
            d = (entry + k) % 8
            ny, nx = cur[0] + dirs[d][0], cur[1] + dirs[d][1]
            if m[ny, nx]:
                if d != prev_dir:
                    pts.append((cur[1] - 1, cur[0] - 1))
                    prev_dir = d
                cur = (ny, nx)
                entry = (d + 5) % 8  # back up and swing clockwise again
                found = True
                break
        if not found:  # isolated pixel
            return [(sx - 1, sy - 1)]
        if cur == (sy, sx) and len(pts) > 2:
            break
    return pts if len(pts) >= 3 else None

FORMAT = "tessera-surveyor/1"


def digitize(path, out_prefix, crop_frac=None, max_side=2200, stone_px=16,
             min_stone_px=25, min_solidity=0.5, rounds=40, render=True,
             source_note="", join=False):
    im = Image.open(path).convert("RGB")
    if crop_frac:
        fx0, fy0, fx1, fy1 = crop_frac
        im = im.crop((int(fx0 * im.width), int(fy0 * im.height),
                      int(fx1 * im.width), int(fy1 * im.height)))
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.float64)
    smooth = np.stack([ndi.gaussian_filter(rgb[..., c], 1.1) for c in range(3)], -1)

    # locate the stones: minima of the color-gradient field
    mag = np.zeros(rgb.shape[:2])
    for c in range(3):
        mag += np.abs(ndi.sobel(smooth[..., c], 0)) + np.abs(ndi.sobel(smooth[..., c], 1))
    mag = ndi.gaussian_filter(mag, 1.0)
    fp = max(5, stone_px // 2)
    markers, n = ndi.label(ndi.minimum_filter(mag, size=fp) == mag)

    # grout reference: the ridge mortar between stones
    ridge = mag > np.percentile(mag, 82)
    grout_color = np.median(rgb[ridge], axis=0) if ridge.any() else np.array([110, 95, 82])

    # per-stone reference color + adaptive wall
    idx = ndi.center_of_mass(markers > 0, markers, range(1, n + 1))
    ref = np.zeros((n + 2, 3))
    tau = np.zeros(n + 2)
    for k, (cy, cx) in enumerate(idx, start=1):
        yi, xi = int(cy), int(cx)
        ref[k] = smooth[yi, xi]
        dg = np.abs(ref[k] - grout_color).max()
        tau[k] = np.clip(0.70 * dg, 16, 55)  # never wider than the grout gap allows

    # THE GROWTH: all stones expand together, one ring per round
    lab = markers.copy()
    forbidden = ridge  # once it hits grout, that's it
    st3 = np.ones((3, 3), bool)
    for _ in range(rounds):
        frontier = (lab == 0) & ndi.binary_dilation(lab > 0, st3) & ~forbidden
        if not frontier.any():
            break
        neigh = ndi.grey_dilation(lab, footprint=st3)
        delta = np.abs(smooth - ref[neigh]).max(axis=2)
        take = frontier & (delta < tau[neigh])
        if not take.any():
            break
        lab[take] = neigh[take]

    # PHASE 2 — CLOSURE (Sebastian, 08-10): each stone expands uniformly out
    # until it meets its neighbor; the seam falls on the medial line where the
    # grout ran. Capped, so real lacunae stay open instead of growing lies.
    close_px = 6
    lab1 = lab.copy()  # color-true extents: colors are measured on these
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    fill = (lab == 0) & (dist <= close_px)
    lab[fill] = lab[iy[fill], ix[fill]]

    # THE CELL JOIN (Sebastian's rule, 08-26; OPT-IN): the mold
    # over-segments full stones along their internal ridges — a vein or a
    # shadowed crack reads as a wall. But a REAL wall has grout in it, and
    # the test is RECONSTRUCTION COST: every seam we keep gets painted
    # grout in the reconstruction, and grout is mostly one color — so
    # painting it over true grout is nearly free, while painting it over
    # stone is costly. A seam joins its two cells when writing grout there
    # would cost more than calling it stone, in the flat-fill render's own
    # pixel error.
    #
    # PASS 1 — grout candidates, from median sets of pixels: the thin
    # unclaimed lattice votes, gated three ways — CERTIFIED (only pixels
    # between two different cells; an internal sliver certifies nothing),
    # DISSIMILAR (only joints between cells whose own colors differ; an
    # over-split stone's fragments share its color, so their ridges never
    # vote — otherwise the split certifies the very ridge that caused it),
    # and QUORUM (a candidate from a handful of pixels is noise wearing a
    # uniform). Candidates are a coarse grid of local medians plus the
    # global median plus the segmentation's own ridge median — medians
    # only, no clustering: one phantom mode can veto joins floor-wide.
    #
    # PASS 2 — enforce the merge per seam, nearest model in CHROMATICITY
    # (a ridge is a darkened stone: same hue, less light — chroma sees
    # through the darkening; plain RGB cannot tell stone-brown from
    # grout-brown on a weathered floor and eats the figure). Ties go to
    # grout.
    # And the cells' own colors must agree, because the parts of one stone
    # share its color while genuinely distinct neighbors do not — stones
    # touching through sub-pixel grout stay in the merge-flag family.
    #
    # ⚠ WHY OPT-IN: color evidence has a ceiling here. A healed ridge-
    # split and two same-colored stones touching without visible grout
    # are the SAME observation — two same-colored cells, a non-grout
    # seam — so any color rule aggressive enough to heal a weathered
    # figure also devours a pebble floor. Telling them apart needs shape
    # priors or a learned model (the conf field's job). Until then the
    # join is a per-floor judgment: try it, look at the render, decide.
    if join:
        unc = lab1 == 0
        dtu = ndi.distance_transform_edt(unc)
        thin = unc & (dtu <= max(2.0, 0.15 * stone_px))
        BIG = 10 ** 9
        d1 = ndi.grey_dilation(lab1, size=5)
        d2 = -ndi.grey_dilation(-np.where(lab1 > 0, lab1, BIG), size=5)
        thin = thin & (d1 != d2) & (d2 < BIG) & (d1 > 0)
        d1c = np.clip(d1, 0, n + 1)
        d2c = np.clip(np.where(d2 == BIG, 0, d2), 0, n + 1)
        thin = thin & (np.abs(ref[d1c] - ref[d2c]).max(axis=-1) > 30)
        QUORUM = 400
        if thin.sum() < QUORUM:
            thin[:] = False
        gglobal = (np.median(rgb[thin], 0) if thin.any()
                   else np.asarray(grout_color, float))
        CELL = max(48, 4 * stone_px)
        gw = rgb.shape[1] // CELL + 1
        gcolor = {}
        gy_, gx_ = np.nonzero(thin)
        if len(gy_):
            cells = (gy_ // CELL) * gw + (gx_ // CELL)
            order = np.argsort(cells)
            cells, gy_, gx_ = cells[order], gy_[order], gx_[order]
            edges = np.nonzero(np.diff(cells))[0] + 1
            for ci, s0, s1 in zip(cells[np.r_[0, edges]],
                                  np.r_[0, edges],
                                  np.r_[edges, len(cells)]):
                if s1 - s0 >= 60:
                    gcolor[int(ci)] = np.median(rgb[gy_[s0:s1], gx_[s0:s1]], 0)
        cands = [gglobal, np.asarray(grout_color, float)]

        def chroma(c):
            c = np.asarray(c, float)
            return c / max(float(c.sum()), 1e-6)

        # each cell's MEASURED mean color — the seed pixel's color is one
        # sample and a fragment seeded on its own ridge line wears the
        # ridge's color, wrongly failing every agreement test it meets
        cmean = ref.copy()
        idx_ = np.arange(1, n + 1)
        cnt_ = ndi.sum(np.ones_like(lab1, float), lab1, idx_)
        for c_ in range(3):
            s_ = ndi.sum(rgb[..., c_], lab1, idx_)
            ok_ = cnt_ > 0
            cmean[1:n + 1, c_][ok_] = s_[ok_] / cnt_[ok_]
        pairs = {}
        for (sa, sb) in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
            m = (sa != sb) & (sa > 0) & (sb > 0)
            ys_, xs_ = np.nonzero(m)
            aa, bb = sa[m], sb[m]
            lo = np.minimum(aa, bb); hi = np.maximum(aa, bb)
            for y_, x_, l_, h_ in zip(ys_, xs_, lo, hi):
                pairs.setdefault((int(l_), int(h_)), []).append((y_, x_))
        parent = list(range(n + 2))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        joined = 0
        for (a, b), px_list in pairs.items():
            if len(px_list) < 4:
                continue
            ys_ = np.array([p[0] for p in px_list])
            xs_ = np.array([p[1] for p in px_list])
            seam_med = np.median(smooth[ys_, xs_], axis=0)
            cy_, cx_ = int(ys_.mean()), int(xs_.mean())
            local = [gcolor[(cy_ // CELL) * gw + (cx_ // CELL)]]                 if (cy_ // CELL) * gw + (cx_ // CELL) in gcolor else []
            stone_mean = 0.5 * (cmean[a] + cmean[b])
            # cost of writing grout over this seam vs calling it stone,
            # by two voices that each must clear an ABSOLUTE bar — the
            # seam must stand a real distance from every grout candidate
            # before a join is even considered, because dc_stone is
            # measured against the very cells the seam borders and is
            # small by construction; without the bar, "nearer stone" wins
            # everywhere on any low-chroma floor by that bias alone.
            #   CHROMA voice: a ridge is a darkened stone — same hue,
            #   less light — so chroma sees through the darkening. Blind
            #   where stone and grout share a hue (cream on mortar,
            #   grisaille).
            #   LUMINANCE-RGB voice: catches exactly those — a cream
            #   stone on dark grout is chroma-invisible but 110 levels
            #   apart in RGB. Its bar is high, because on a weathered
            #   floor stone-brown and grout-brown are RGB neighbors and a
            #   permissive RGB voice eats the figure.
            dc_grout = min(float(np.abs(chroma(seam_med) - chroma(g_)).max())
                           for g_ in cands + local)
            dc_stone = float(np.abs(chroma(seam_med) - chroma(stone_mean)).max())
            chroma_stone = (dc_grout > 0.025
                            and dc_stone + 0.008 < dc_grout)
            err_grout = min(float(np.abs(seam_med - g_).max())
                            for g_ in cands + local)
            err_stone = float(np.abs(seam_med - stone_mean).max())
            rgb_stone = err_grout > 45.0 and err_stone + 12.0 < err_grout
            colors_agree = np.abs(cmean[a] - cmean[b]).max() <= max(
                24.0, min(tau[a], tau[b]))
            if (chroma_stone or rgb_stone) and colors_agree:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
                    joined += 1
        if joined:
            remap = np.arange(n + 2)
            for k in range(1, n + 1):
                remap[k] = find(k)
            lab = remap[lab]
            lab1 = remap[lab1]
    else:
        joined = 0

    # measure every stone
    stones, merged = [], 0
    max_stone = int((3.5 * stone_px) ** 2)
    for sl, k in zip(ndi.find_objects(lab), range(1, n + 1)):
        if sl is None:
            continue
        m = lab[sl] == k
        a = int(m.sum())
        if a < min_stone_px or a > max_stone:
            continue
        ys, xs = np.nonzero(m)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        # color from the color-true phase-1 stone, shape from the closed tile
        m1 = lab1[sl] == k
        if m1.sum() >= min_stone_px // 2:
            y1, x1 = np.nonzero(m1)
            color = [int(c) for c in rgb[y1 + sl[0].start, x1 + sl[1].start].mean(axis=0)]
        else:
            color = [int(c) for c in rgb[ys, xs].mean(axis=0)]
        pts = np.column_stack([xs, ys]).astype(float)
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        flags = []
        if a / max(hull.volume, 1.0) < min_solidity:
            merged += 1
            flags.append("merged")   # the law of this file: flags, never deletions
        # THE SHAPE: the stone's real traced outline (hull was only the gate)
        tb = trace_boundary(m)
        if tb is None:
            continue
        poly = [[round(float(x + sl[1].start), 1), round(float(y + sl[0].start), 1)]
                for x, y in tb]
        # winding normalized: positive shoelace area (spec: CCW)
        sh = sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                 - poly[(i + 1) % len(poly)][0] * poly[i][1]
                 for i in range(len(poly)))
        if sh < 0:
            poly.reverse()
        elif sh == 0:
            flags.append("degenerate")   # zero area: the tracer names its own
        cy, cx = ys.mean(), xs.mean()
        yy, xx = ys - cy, xs - cx
        cov = np.array([[(xx*xx).mean(), (xx*yy).mean()],
                        [(xx*yy).mean(), (yy*yy).mean()]])
        evals, evecs = np.linalg.eigh(cov)
        if bool(tau[k] <= 18):
            # Sebastian's diagnosis 08-10: near-grout stones get a cramped
            # wall -> stall -> over-split. Training must see this flag.
            flags.append("near_grout")
        stones.append({"poly": poly, "rgb": color, "area_px": a,
                       "wall": round(float(tau[k]), 1),
                       "near_grout": bool(tau[k] <= 18),
                       "source": "slime-mold",
                       "conf": None,   # the mold has no notion of doubt; v0 lets it say so
                       "flags": flags,
                       "aspect": round(float(np.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9))), 3),
                       "angle_deg": round(float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])) % 180), 1)})

    W, H = im.width, im.height
    gc = [int(c) for c in np.median(rgb[lab == 0], axis=0)] if (lab == 0).any() else [int(c) for c in grout_color]
    tess = {"format": FORMAT,
            "source_file": os.path.basename(path),
            "source_note": source_note,
            "method": "slime-mold",
            "analyzed_w": W, "analyzed_h": H,
            "crop_frac": list(crop_frac) if crop_frac else None,
            "grout_rgb": gc,
            "coverage": round(float((lab > 0).sum() / lab.size), 3),
            "merged_flagged": merged, "joined": joined,
            "n_stones": len(stones), "stones": stones}
    with open(out_prefix + ".stones.json", "w") as f:
        json.dump(tess, f)

    if not render:
        print(f"{os.path.basename(path)}: {len(stones)} stones grown "
              f"(coverage {tess['coverage']:.0%}, {merged} merges flagged) "
              f"[no render] -> {out_prefix}.stones.json", flush=True)
        return tess

    with open(out_prefix + "-digital.svg", "w") as f:
        f.write(render_svg(tess))
    im.save(out_prefix + "-photo.png")

    # PIXEL-TRUE rendering (Sebastian, 08-10): the boundary is the measurement;
    # the fill is the photograph. Keep every stone's actual pixels, replace
    # only the grout — digitization as grout removal.
    seam = np.zeros(lab.shape, bool)
    seam[:-1, :] |= lab[:-1, :] != lab[1:, :]
    seam[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    outp = rgb.copy()
    outp[lab == 0] = gc
    outp[seam] = gc
    Image.fromarray(outp.astype(np.uint8)).save(out_prefix + "-pixels.png")
    print(f"{os.path.basename(path)}: {len(stones)} stones grown "
          f"({joined} ridge-splits joined, {merged} merges flagged, "
          f"coverage {tess['coverage']:.0%}) -> {out_prefix}.stones.json")
    return tess

if __name__ == "__main__":
    digitize(sys.argv[1], sys.argv[2],
             crop_frac=tuple(float(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else None)


def render_svg(tess):
    """The floor, from its data file alone — the acceptance test a human
    runs with their eyes. Merged-flagged stones are skipped here (their
    record stays in the data; consumers choose by flag)."""
    W, H = tess["analyzed_w"], tess["analyzed_h"]
    gc = tess["grout_rgb"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
             f'<rect width="{W}" height="{H}" '
             f'fill="rgb({gc[0]},{gc[1]},{gc[2]})"/>']
    for s in tess["stones"]:
        if "merged" in s["flags"]:
            continue
        d = "M" + " L".join(f"{x},{y}" for x, y in s["poly"]) + " Z"
        r, g, b = s["rgb"]
        parts.append(f'<path d="{d}" fill="rgb({r},{g},{b})" '
                     f'stroke="rgb({gc[0]},{gc[1]},{gc[2]})" '
                     f'stroke-width="0.8"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def digitize_tiled(path, out_prefix, core=1024, margin=128, stone_px=18,
                   min_stone_px=25, min_solidity=0.5, rounds=40,
                   source_note="", join=False):
    """Tiled survey at native resolution — accuracy through local
    calibration. One global grout threshold across a large photograph is
    a compromise: lighting drifts, and the 82nd percentile of the whole
    frame is right nowhere in particular. Processing in tiles (core plus
    margin, stones kept only when their centroid lands in the core, so
    seams dedupe themselves) re-derives the grout color, the adaptive
    walls, and the ridge threshold locally, where they are true.

    The pixel-true render is not produced in tiled mode (it needs the
    whole-frame label map); the data file and the proof render are.
    """
    im = Image.open(path).convert("RGB")
    W, H = im.size
    tdir = tempfile.mkdtemp(prefix=".tiles-",
                            dir=os.path.dirname(os.path.abspath(out_prefix))
                            or ".")
    stones, grout_votes, merged_total, joined_total, tiles = [], [], 0, 0, 0
    try:
        for y0 in range(0, H, core):
            for x0 in range(0, W, core):
                tx0, ty0 = max(0, x0 - margin), max(0, y0 - margin)
                tx1 = min(W, x0 + core + margin)
                ty1 = min(H, y0 + core + margin)
                # a remainder tile is widened BACKWARD into already-covered
                # ground rather than skipped: skipping would leave its core
                # surveyed by no tile at all, and the extra overlap is free
                # (centroid-in-core dedupes it)
                if tx1 - tx0 < 200:
                    tx0 = max(0, tx1 - 200)
                if ty1 - ty0 < 200:
                    ty0 = max(0, ty1 - 200)
                if tx1 - tx0 < 200 or ty1 - ty0 < 200:
                    continue   # the photograph itself is smaller than 200px
                tp = os.path.join(tdir, f"{tx0}_{ty0}.png")
                im.crop((tx0, ty0, tx1, ty1)).save(tp)
                tess = digitize(tp, os.path.join(tdir, f"t{tx0}_{ty0}"),
                                max_side=10 ** 9, stone_px=stone_px,
                                min_stone_px=min_stone_px,
                                min_solidity=min_solidity, rounds=rounds,
                                render=False, join=join)
                os.remove(tp)
                tiles += 1
                grout_votes.append(tess["grout_rgb"])
                merged_total += tess["merged_flagged"]
                joined_total += tess.get("joined", 0)
                cx1, cy1 = min(W, x0 + core), min(H, y0 + core)
                for s in tess["stones"]:
                    xs = [p[0] for p in s["poly"]]
                    ys = [p[1] for p in s["poly"]]
                    mx = sum(xs) / len(xs) + tx0
                    my = sum(ys) / len(ys) + ty0
                    if not (x0 <= mx < cx1 and y0 <= my < cy1):
                        continue
                    s["poly"] = [[round(px + tx0, 1), round(py + ty0, 1)]
                                 for px, py in s["poly"]]
                    stones.append(s)
    finally:
        shutil.rmtree(tdir, ignore_errors=True)
    gc = ([int(np.median([g[i] for g in grout_votes])) for i in range(3)]
          if grout_votes else [110, 95, 82])
    tess = {"format": FORMAT,
            "source_file": os.path.basename(path),
            "source_note": source_note,
            "method": "slime-mold (tiled)",
            "analyzed_w": W, "analyzed_h": H, "crop_frac": None,
            "grout_rgb": gc, "coverage": 1.0,
            "tiling": {"core": core, "margin": margin, "tiles": tiles},
            "merged_flagged": merged_total, "joined": joined_total,
            "n_stones": len(stones), "stones": stones}
    with open(out_prefix + ".stones.json", "w") as f:
        json.dump(tess, f)
    with open(out_prefix + "-digital.svg", "w") as f:
        f.write(render_svg(tess))
    im.save(out_prefix + "-photo.png")
    print(f"{os.path.basename(path)}: {len(stones)} stones grown in "
          f"{tiles} tiles ({joined_total} ridge-splits joined, "
          f"{merged_total} merges flagged) -> {out_prefix}.stones.json")
    return tess
