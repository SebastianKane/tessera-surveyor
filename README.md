# tessera-surveyor

**Digitize a photograph of a real mosaic into per-stone data.**

Point it at a photo of a tessellated floor and it writes the floor down as
data: every stone's boundary polygon, mean color, area, aspect ratio, and
orientation, plus the measured grout color. Then it proves itself the only
way that counts — by re-rendering the floor *from the data file alone*, so
you can put the reconstruction beside the photograph and judge with your
own eyes whether the data says the floor.

![Alexander Mosaic, photograph and reconstruction](examples/alexander-proof.jpg)

*The Alexander Mosaic (detail), House of the Faun, Pompeii — photograph
(public domain, via Wikimedia Commons) above, and below it the same floor
re-rendered purely from its `.stones.json`: 7,526 stones measured, 1,206
merged regions detected and excluded.*

## Why classic CV, no learning

Everything here is gradient fields, watershed, and convex hulls — numpy,
scipy, and Pillow, nothing to download, nothing to fine-tune, and every
result reproducible to the byte. Brightness thresholds fail on dark stones
(they read as grout), so segmentation is by **homogeneity**: stones of any
value are smooth basins in the color-gradient field, and grout lines are
the ridges between them.

## Install and run

```
pip install -r requirements.txt
bin/survey examples/alexander-detail.jpg -o alexander --crop 0.15,0.10,0.85,0.55
pytest tests/ -v
```

Three files come out:

    alexander.stones.json   the floor as data (schema below)
    alexander-digital.svg   the proof: re-rendered from the data alone
    alexander-photo.png     the analyzed image, as measured

### Options

```
--crop X0,Y0,X1,Y1   fractions, applied FIRST at native resolution —
                     grout must stay super-pixel
--stone-px N         expected stone diameter in analyzed pixels (16)
--min-stone-px N     area floor (30)
--min-solidity F     merge gate threshold (0.55)
--max-side N         longest analyzed side (2200)
--note TEXT          provenance note carried into the output
```

## The data

```json
{
  "format": "tessera-surveyor/0",
  "grout_rgb": [168, 148, 122],
  "merged_flagged": 1206,
  "n_stones": 7526,
  "stones": [
    {"poly": [[x, y], ...], "rgb": [r, g, b], "area_px": 214,
     "aspect": 1.41, "angle_deg": 63.5, "near_grout": false}
  ]
}
```

## Honesty as a design rule

A measuring tool must not be confidently wrong, so the failure modes are
surfaced instead of smoothed over:

- **Merges are counted, never kept.** Stones fused by sub-pixel grout show
  as one region whose convex hull dwarfs its pixel count; the solidity gate
  detects them, reports the count in `merged_flagged`, and excludes them.
- **Grout-colored regions are flagged, never dropped.** A region whose
  color sits at the measured grout color is probably a joint fragment — but
  real floors do lay stones in grout-colored minerals, so it carries a
  `near_grout` flag and the choice stays with the consumer.
- **Pale-on-pale is unreliable** at modest resolution, and the output says
  so in its own `method` field.
- The test suite includes a **ground-truth floor**: a synthetic tessellation
  with a known answer, against which count, color, geometry, and determinism
  are all checked. A tool that measures must be measured.

## Provenance

AI-authored, human-directed: written by **Paean-AI** (a Claude-based agent)
with Sebastian Kane directing, August 2026. Issues and PRs are read by both.
See AUTHORS.

## License

MIT — see LICENSE. The example photograph is public domain (Wikimedia
Commons); its details are noted in examples/PROVENANCE.md.
