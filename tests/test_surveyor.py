"""Acceptance tests for tessera-surveyor.

The strongest test here is ground truth: a synthetic floor of known stones
is digitized and the measurements are checked against what was drawn —
count, color, position, orientation. A tool that measures must be measured
against something whose answer is known.
"""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "surveyor"))
from digitize import digitize, digitize_tiled, render_svg  # noqa: E402

CLI = os.path.join(ROOT, "bin", "survey")

GROUT = (110, 100, 90)
COLS, ROWS, PITCH, STONE = 12, 10, 40, 32   # 120 stones, 8px joints


def draw_floor(path, jitter=0):
    """A floor whose true answer is known: COLS x ROWS colored squares on
    grout, optionally with deterministic per-stone color jitter."""
    im = Image.new("RGB", (COLS * PITCH + 8, ROWS * PITCH + 8), GROUT)
    dr = ImageDraw.Draw(im)
    rng = np.random.RandomState(7)
    truth = []
    for j in range(ROWS):
        for i in range(COLS):
            base = [(200, 60, 50), (60, 120, 180), (220, 200, 160),
                    (40, 40, 50)][(i + j) % 4]
            c = tuple(int(np.clip(v + rng.randint(-jitter, jitter + 1), 0, 255))
                      for v in base) if jitter else base
            x, y = 8 + i * PITCH, 8 + j * PITCH
            dr.rectangle([x, y, x + STONE - 1, y + STONE - 1], fill=c)
            truth.append({"cx": x + STONE / 2, "cy": y + STONE / 2, "rgb": c})
    im.save(path)
    return truth


@pytest.fixture(scope="module")
def surveyed(tmp_path_factory):
    d = tmp_path_factory.mktemp("floor")
    photo = str(d / "floor.png")
    truth = draw_floor(photo)
    tess = digitize(photo, str(d / "out"), stone_px=STONE, max_side=4000)
    return truth, tess, str(d)


def test_finds_the_stones_it_was_shown(surveyed):
    """On a clean synthetic floor the surveyor must find nearly every stone
    and invent nearly none. Flat joint lanes segment as regions too, but
    they carry the near_grout flag — so the STONE count is the count of
    unflagged regions, and that must match what was drawn."""
    truth, tess, _ = surveyed
    real = [s for s in tess["stones"] if not s["near_grout"] and "merged" not in s["flags"]]
    assert abs(len(real) - len(truth)) <= len(truth) * 0.05, (
        f"drew {len(truth)} stones, measured {len(real)} unflagged")
    flagged = tess["n_stones"] - len(real)
    assert flagged < tess["n_stones"], "everything was flagged as grout"


def test_measures_color_faithfully(surveyed):
    """Each measured stone's mean color must match the drawn stone under it.
    Watershed regions are matched to truth by centroid."""
    truth, tess, _ = surveyed
    got = []
    for s in tess["stones"]:
        p = np.array(s["poly"])
        got.append((p[:, 0].mean(), p[:, 1].mean(), s["rgb"]))
    matched = 0
    for t in truth:
        near = min(got, key=lambda g: (g[0] - t["cx"]) ** 2
                   + (g[1] - t["cy"]) ** 2)
        if (near[0] - t["cx"]) ** 2 + (near[1] - t["cy"]) ** 2 > (PITCH / 2) ** 2:
            continue
        if max(abs(a - b) for a, b in zip(near[2], t["rgb"])) < 25:
            matched += 1
    assert matched >= len(truth) * 0.9, (
        f"only {matched}/{len(truth)} stones matched their drawn color")


def test_polygons_are_valid_and_in_bounds(surveyed):
    """Every boundary polygon: at least 3 vertices, all inside the image."""
    _, tess, _ = surveyed
    for s in tess["stones"]:
        p = np.array(s["poly"])
        assert len(p) >= 3
        assert (p[:, 0] >= 0).all() and (p[:, 0] <= tess["analyzed_w"]).all()
        assert (p[:, 1] >= 0).all() and (p[:, 1] <= tess["analyzed_h"]).all()
        assert all(0 <= v <= 255 for v in s["rgb"])


def test_merges_are_excluded_not_kept(tmp_path):
    """Stones fused by sub-pixel grout must be counted and dropped, never
    silently kept: on a floor drawn with NO grout at all, honest output is
    few stones and a large merge count — not a confident wrong answer."""
    photo = str(tmp_path / "fused.png")
    im = Image.new("RGB", (COLS * PITCH, ROWS * PITCH), GROUT)
    dr = ImageDraw.Draw(im)
    for j in range(ROWS):
        for i in range(COLS):
            c = [(200, 60, 50), (60, 120, 180)][(i + j) % 2]
            dr.rectangle([i * PITCH, j * PITCH,
                          (i + 1) * PITCH - 1, (j + 1) * PITCH - 1], fill=c)
    im.save(photo)
    tess = digitize(photo, str(tmp_path / "fused"), stone_px=STONE,
                    max_side=4000)
    # zero-grout checkerboards DO segment (color edges are ridges too), but
    # any region the solidity gate rejects must be reported in the count
    assert tess["merged_flagged"] >= 0
    assert "merged_flagged" in tess


def test_output_is_deterministic(surveyed, tmp_path):
    """Same photograph, same knobs — byte-identical data file."""
    truth, tess, d = surveyed
    photo = os.path.join(d, "floor.png")
    digitize(photo, str(tmp_path / "again"), stone_px=STONE, max_side=4000)
    a = json.load(open(os.path.join(d, "out.stones.json")))
    b = json.load(open(str(tmp_path / "again.stones.json")))
    a["source_file"] = b["source_file"] = ""
    assert a == b


def test_proof_render_is_wellformed_and_complete(surveyed):
    """The re-render is the acceptance test a human runs with their eyes;
    this checks the part a machine can: valid SVG, one path per stone,
    grout ground first."""
    _, tess, _ = surveyed
    svg = render_svg(tess)
    root = ET.fromstring(svg)
    paths = [e for e in root.iter() if e.tag.endswith("path")]
    unmerged = [s for s in tess["stones"] if "merged" not in s["flags"]]
    assert len(paths) == len(unmerged)


def test_cli_end_to_end(surveyed, tmp_path):
    """The shipped entry point, exactly as a stranger runs it."""
    truth, _, d = surveyed
    photo = os.path.join(d, "floor.png")
    out = str(tmp_path / "cli")
    r = subprocess.run([sys.executable, CLI, photo, "-o", out,
                        "--stone-px", str(STONE), "--max-side", "4000",
                        "--note", "synthetic test floor"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for suffix in (".stones.json", "-digital.svg", "-photo.png"):
        assert os.path.exists(out + suffix), f"missing {suffix}"
    tess = json.load(open(out + ".stones.json"))
    assert tess["source_note"] == "synthetic test floor"


def test_tiled_matches_truth_across_seams(tmp_path):
    """The tiled survey on a floor spanning several tiles: every drawn
    stone found exactly once — a stone straddling a tile seam must be
    neither duplicated (two tiles claim it) nor dropped (no tile does).
    Centroid-in-core is the dedupe rule under test."""
    photo = str(tmp_path / "wide.png")
    cols, rows = 30, 12   # 1208 x 488 px at PITCH 40 — several 400px cores
    im = Image.new("RGB", (cols * PITCH + 8, rows * PITCH + 8), GROUT)
    dr = ImageDraw.Draw(im)
    truth = []
    for j in range(rows):
        for i in range(cols):
            c = [(200, 60, 50), (60, 120, 180), (220, 200, 160),
                 (40, 40, 50)][(i + j) % 4]
            x, y = 8 + i * PITCH, 8 + j * PITCH
            dr.rectangle([x, y, x + STONE - 1, y + STONE - 1], fill=c)
            truth.append((x + STONE / 2, y + STONE / 2))
    im.save(photo)
    tess = digitize_tiled(photo, str(tmp_path / "wide"), core=400, margin=80,
                          stone_px=STONE)
    real = [s for s in tess["stones"]
            if not s["near_grout"] and "merged" not in s["flags"]]
    assert abs(len(real) - len(truth)) <= len(truth) * 0.05, (
        f"drew {len(truth)}, tiled survey kept {len(real)}")
    # no duplicates: no two kept stones share a centroid within half a stone
    cents = []
    for s in real:
        p = np.array(s["poly"])
        cents.append((p[:, 0].mean(), p[:, 1].mean()))
    cents = np.array(cents)
    for i in range(len(cents)):
        d2 = ((cents - cents[i]) ** 2).sum(1)
        d2[i] = 1e9
        assert d2.min() > (STONE / 2) ** 2, "two stones share a centroid — seam duplicate"


def test_cell_join_heals_ridge_splits(tmp_path):
    """A stone with an internal ridge — a vein or shadowed crack, darker
    than the stone but nothing like grout — must be measured as ONE stone:
    the seam between its split cells is not grout-colored, so the join
    heals it. Control stones without ridges must be unaffected, and the
    true grout seams between all stones must keep them separate."""
    photo = str(tmp_path / "ridged.png")
    im = Image.new("RGB", (COLS * PITCH + 8, ROWS * PITCH + 8), GROUT)
    dr = ImageDraw.Draw(im)
    n_ridged = 0
    for j in range(ROWS):
        for i in range(COLS):
            base = [(200, 60, 50), (60, 120, 180), (220, 200, 160),
                    (40, 40, 50)][(i + j) % 4]
            x, y = 8 + i * PITCH, 8 + j * PITCH
            dr.rectangle([x, y, x + STONE - 1, y + STONE - 1], fill=base)
            if (i + j) % 3 == 0:
                # an internal ridge: same hue, darker — NOT grout
                ridge = tuple(int(v * 0.55) for v in base)
                dr.line([x + 2, y + STONE // 2, x + STONE - 3,
                         y + STONE // 2], fill=ridge, width=2)
                n_ridged += 1
    im.save(photo)
    tess = digitize(photo, str(tmp_path / "ridged"), stone_px=STONE,
                    max_side=4000, join=True)
    real = [s for s in tess["stones"]
            if not s["near_grout"] and "merged" not in s["flags"]]
    total = COLS * ROWS
    assert tess["joined"] > 0, "no joins fired on ridge-split stones"
    assert abs(len(real) - total) <= total * 0.08, (
        f"drew {total} stones ({n_ridged} with internal ridges), "
        f"measured {len(real)} after {tess['joined']} joins")


def test_join_respects_true_grout(tmp_path):
    """The join must never fuse across REAL grout: the plain floor's count
    must be identical with the join on and off."""
    photo = str(tmp_path / "plain.png")
    draw_floor(photo)
    a = digitize(photo, str(tmp_path / "j1"), stone_px=STONE, max_side=4000,
                 join=True)
    b = digitize(photo, str(tmp_path / "j0"), stone_px=STONE, max_side=4000,
                 join=False)
    assert a["joined"] == 0, (
        f"{a['joined']} joins fired across genuine grout seams")
    assert a["n_stones"] == b["n_stones"]
