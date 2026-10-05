# face_search

Search a folder of photos for a specific person. Give it reference photos,
and it copies every photo in the search folder that appears to contain them
into an output folder.

Fully offline — runs on your own machine using [InsightFace](https://github.com/deepinsight/insightface)
(`buffalo_l` model, CPU). No cloud, no accounts, no data leaves your computer.
Originals are never modified; matches are copied, never moved.

## Requirements

- Python 3.10+
- `pip install insightface onnxruntime opencv-python numpy`

## Usage

```
python face_search.py --ref <reference_folder> --photos <photos_folder> --out <output_folder>
```

Example (Windows):

```
python face_search.py --ref .\refs\person1 --photos "C:\Users\You\Pictures" --out .\matches_person1
```

Suggested folder layout:

```
face_search/
├── face_search.py
├── README.md
├── LICENSE
├── refs/
│   ├── person1/        <- reference photos of person 1
│   └── person2/        <- reference photos of person 2
├── matches_person1/    <- created automatically
└── matches_person2/    <- created automatically
```

- `--ref` — folder with reference photos of the person (3–5 clear, front-facing shots work best)
- `--photos` — folder of photos to search (subfolders included)
- `--out` — folder where matching photos are copied (created automatically)
- `--threshold` — similarity cutoff, 0–1, default `0.45`

To search for another person, use a different reference folder and output folder:

```
python face_search.py --ref .\refs\person2 --photos "C:\Users\You\Pictures" --out .\matches_person2
```

## Tuning the threshold

- **Too many wrong matches?** Raise it: `--threshold 0.55`
- **Missing photos you know are there?** Lower it: `--threshold 0.35`

## Desktop app (Windows)

Prefer a point-and-click interface over the terminal? The same search engine
is available as a desktop app.

![Face Search GUI](screenshot.png)

1. Download **FaceSearchGUI.exe** from the
   [Releases page](https://github.com/Zabas-dot/face_search/releases).
2. Choose your folders:
   - **Reference folder** — a few clear photos of the person's face
   - **Photos folder** — the pictures you want to search through
   - **Output folder** — where matches get copied (pick any empty folder)
3. Press **Run Search**.

The very first launch downloads the face-recognition model once (~300 MB);
after that it works fully offline.

### App features

- Progress bar with live photo count and match count
- Embedding cache: repeat scans of unchanged photos take seconds
- Similarity threshold slider
- Multi-person reference folders (subfolders per person, or automatic
  separation with a pre-scan preview)
- HEIC/HEIF photo support
- Desktop notification when the scan finishes

### Building from source (optional)

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python face_search_gui.py
```

## Notes

- First run downloads the recognition model (~300 MB) into `~/.insightface`.
- Reference photos containing multiple faces will enroll every face found;
  solo shots of the person give the cleanest results.
- Works best on clear, front-facing faces. Profile shots, sunglasses, heavy
  shadows, and very small/distant faces may be missed.

## License

MIT
