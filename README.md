# face_search

Find every photo of a person in a folder of photos. Give it a few reference photos of their face, and it copies every photo that appears to contain them into an output folder.

Runs entirely on your own computer: no cloud, no accounts, no photos leave your computer. Originals are never modified; matches are copied, never moved. The face models are licensed for commercial use (see [Face models and licenses](#face-models-and-licenses)).

Two ways to use it: a **desktop app** for Windows (no installation), or a **command-line tool** (Python, any OS).

## Desktop app (Windows)

1. Download **FaceSearchGUI.exe** from the [Releases page](https://github.com/Zabas-dot/face_search/releases) and double-click it. Windows 10/11, 64-bit.

   > **"Windows protected your PC"?** The app isn't digitally signed, so Windows warns about it the first time. Click **More info → Run anyway**. The full source is in this repository if you'd rather check it or build it yourself.

2. Choose your folders:
   - **Photos of the person**: 5–10 clear photos of their face. To find several people at once, give each one a subfolder (e.g. `Refs\Mom`, `Refs\Dad`).
   - **Photos to search**: subfolders are included.
   - **Where to put the matches**: any folder; matches are copied here.
3. Press **Run search**. The app first shows the faces it will look for, so you can catch a wrong reference photo before the scan starts.

The first search downloads the face models once (about 262 MB). After that it works fully offline.

### App features

- **Fast repeat searches**: photos scanned before are re-checked in seconds.
- **Graphics-card acceleration** when available (most Windows graphics cards), with automatic fallback to the processor.
- **Strictness slider**, from "More matches" to "Fewer wrong matches".
- **Review matches**, most certain first, labelled Strong / Good / Borderline:
  - *Not a match* (with undo) removes a wrong match and never adds it again.
  - *Add as reference* teaches the search from good matches.
- **Several people at once**: one subfolder per person, or automatic separation with a preview.
- **Group photos work as references**: only the right person's face is used.
- **Thorough mode** for small faces in group shots and large photos.
- HEIC/iPhone photos, desktop notification when done.

## Command-line tool

### Requirements

- Python 3.10+
- `face_search.py` and `face_model.py` from this repository
- `pip install onnxruntime opencv-python numpy onnx requests`
- For HEIC/iPhone photos, also: `pip install pillow pi-heif`

### Usage

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
├── refs/
│   ├── person1/        <- reference photos of person 1
│   └── person2/        <- reference photos of person 2
├── matches_person1/    <- created automatically
└── matches_person2/    <- created automatically
```

- `--ref`: folder with reference photos of **one** person (3–5 clear, front-facing shots work best)
- `--photos`: folder of photos to search (subfolders included)
- `--out`: folder where matching photos are copied (created automatically)
- `--threshold`: similarity cutoff, 0–1, default `0.40`

To search for another person, run it again with that person's reference folder and a different output folder:

```
python face_search.py --ref .\refs\person2 --photos "C:\Users\You\Pictures" --out .\matches_person2
```

### Tuning the threshold

- Too many wrong matches? Raise it: `--threshold 0.50`
- Missing photos you know are there? Lower it: `--threshold 0.30`

## Running the app from source (Windows)

Needs Python 3.12 or newer.

```
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python face_search_gui.py
```

To build `FaceSearchGUI.exe` yourself, double-click `build_exe.bat`. It sets everything up, adds graphics-card support and puts the program in `dist\`.

Run the tests (fake face data, about 30 seconds): `python tests\run_tests.py`

### Tuning the strictness for your photos

`tools\calibrate.py` measures how well the face model tells apart the people in *your* photos and suggests thresholds. Make a folder with one subfolder per person (3+ people, 5+ photos each, ideally photos where they're the only face), then run:

```
python tools\calibrate.py C:\path\to\calibration_folder
```

It prints, for each threshold, how often the right person is found and how often someone else is wrongly matched. The report contains only numbers.

| File | What it is |
|---|---|
| `face_model.py` | Face detection + recognition (YuNet + AuraFace), model download |
| `face_search.py` | The command-line tool; also the image reader the app uses |
| `face_search_gui.py` | The desktop app |
| `search_engine.py` | The app's search backend: matching, scan cache, output-folder housekeeping |
| `build_exe.bat` | One-double-click builder for the exe |
| `tests/` | Automated tests |
| `tools/calibrate.py` | Measures accuracy on your photos and suggests thresholds |

## Notes

- The first run downloads the face models (about 262 MB) into `~/.face_search/models`, checking each file's SHA-256.
- Works best on clear, front-facing faces. Profile shots, sunglasses, heavy shadows and very small or distant faces may be missed (the app's Thorough mode helps with small faces).
- **Command line only:** a reference photo with several faces enrolls every face in it, so use solo shots. The app picks the right face automatically.

## Face models and licenses

| Model | Job | License | Source |
|---|---|---|---|
| YuNet (2023mar) | Finds faces | MIT | [OpenCV Zoo](https://huggingface.co/opencv/face_detection_yunet) |
| AuraFace v1 | Recognizes faces | Apache 2.0 | [fal.ai](https://huggingface.co/fal/AuraFace-v1) |

Both licenses allow commercial use. The models are downloaded from their publishers on first use, not included in this repository or the exe. Only AuraFace's own recognition model is used. The AuraFace download page also hosts InsightFace detector and landmark files, which keep InsightFace's non-commercial license, so this app doesn't use them.

As with most face models, the publishers don't fully document their training data. This isn't legal advice; if commercial use matters to you, review the licenses yourself.

## License

The code is MIT-licensed (see `LICENSE`). Third-party models and libraries keep their own licenses (see `THIRD_PARTY_NOTICES.md`).
