# Example photographs

**alexander-detail.jpg** — "Alexander Mosaic detail of Alexander the Great",
House of the Faun, Pompeii (c. 100 BC), now in the Naples National
Archaeological Museum. The one source photograph shipped in the repository.

- Source: Wikimedia Commons,
  https://commons.wikimedia.org/wiki/File:Alexander_Mosaic_detail_of_Alexander_the_Great.jpg
- License: Public domain (faithful reproduction of a two-dimensional work
  whose copyright has expired).

## Gallery images

Each pairs the source photograph (top) with the surveyor's reconstruction
rendered purely from its own data file (bottom), produced with
`bin/survey ... --learned --model PATH --gpu --max-side 4000`. The other
source photographs are not shipped for size; fetch them from the Commons
links and reproduce.

**gorgon.jpg** — Gorgon medallion, opus tessellatum, National Archaeological Museum, Athens.
- Source: https://commons.wikimedia.org/wiki/File:Mosaic_floor_opus_tessellatum_detail_Gorgone_NAMA_Athens_Greece.jpg
- License: CC0

**alexander.jpg** — the Alexander detail above, cropped `--crop 0.15,0.10,0.85,0.55`.

**googleart.jpg** — Roman floor mosaic (Google Art Project).
- Source: https://commons.wikimedia.org/wiki/File:Mosaic_-_Google_Art_Project.jpg
- License: Public domain

**partridge.jpg** — Floor Mosaic with Partridge, Roman, Walters Art Museum 43.18.
- Source: https://commons.wikimedia.org/wiki/File:Roman_-_Floor_Mosaic_with_Partridge_-_Walters_4318.jpg
- License: Public domain

**centaur.jpg** — Centaur mosaic from Hadrian's Villa, Altes Museum, Berlin (Google Art Project).
- Source: https://commons.wikimedia.org/wiki/File:Centaur_mosaic_-_Google_Art_Project_-_CropFrame_-_Plus1ev.jpg
- License: Public domain

**tiberias.jpg** — Segment of a synagogue mosaic floor from Tiberias, Eretz Israel Museum, Tel Aviv.
- Source: https://commons.wikimedia.org/wiki/File:Segment_of_synagogue_mosaic_floor_from_Tiberias_at_Eretz_Israel_Museum_in_Tel_Aviv_(detail).jpg
- License: CC0

## The verdicts

**verdicts.jpg** — an 860×645 region of the Gorgon medallion (the CC0
photograph above, at native resolution) with the learned wall's 529
outlines drawn in the color of the verdict a human gave each one. The
verdicts are Sebastian Kane's, given stone by stone in a purpose-built
annotator; they are the training signal for models/wall.npz and
models/seam.npz.
