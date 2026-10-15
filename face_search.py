#!/usr/bin/env python3
"""
face_search.py -- search a folder of photos for a specific person.

You give it reference photos of the person; it copies every photo in the
search folder that appears to contain them into an output folder.
Runs on your own machine. Uses face_model.py: YuNet face detection (MIT)
and AuraFace recognition (Apache 2.0), both licensed for commercial use.

Setup (Windows):
    py -3.12 -m venv .venv
    .venv\\Scripts\\activate
    pip install onnxruntime opencv-python numpy onnx requests

Usage:
    python face_search.py --ref .\\refs\\person1 --photos .\\all_photos --out .\\matches_person1
    python face_search.py --ref .\\refs\\person1 --photos .\\all_photos --out .\\matches_person1 --threshold 0.50

First run downloads the face models (~262 MB) into %USERPROFILE%\\.face_search\\models.
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def list_images(folder):
    out = []
    for root, dirs, files in os.walk(folder):
        # Skip hidden/system folders inside the chosen folder: digiKam's
        # ".dtrash" trash, thumbnail caches, the Windows recycle bin, NAS
        # metadata. (A chosen folder itself is always searched.)
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and d not in ("$RECYCLE.BIN", "@eaDir", "System Volume Information")]
        for f in files:
            # "._name.jpg" files are macOS metadata left on USB drives, not photos.
            if f.startswith("._"):
                continue
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


_heif_registered = False


def _read_image(image_path):
    """Read an image file. Falls back to decoding from bytes (cv2.imread
    can't open non-ASCII paths on Windows) and then to PIL for HEIC/HEIF."""
    global _heif_registered
    img = cv2.imread(image_path)
    if img is not None:
        return img
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    # Try PIL (with pi-heif registered, handles .heic/.heif)
    try:
        from PIL import Image, ImageOps
        if not _heif_registered:
            try:
                from pi_heif import register_heif_opener  # HEIC decoding (LGPL)
                register_heif_opener()
            except ImportError:
                pass
            _heif_registered = True
        with Image.open(image_path) as pil:
            pil = ImageOps.exif_transpose(pil).convert("RGB")  # honor phone rotation
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def read_faces(app, image_path):
    """Return [(normed_embedding, face_area, (x1, y1, x2, y2))] for every
    face in the photo. Raises IOError if the file can't be decoded."""
    img = _read_image(image_path)
    if img is None:
        raise IOError("could not read image")
    out = []
    for f in app.get(img):
        x1, y1, x2, y2 = (float(v) for v in f.bbox[:4])
        out.append((f.normed_embedding.astype(np.float32),
                    max(0.0, x2 - x1) * max(0.0, y2 - y1),
                    (x1, y1, x2, y2)))
    return out


def face_embeddings(app, image_path):
    """Embeddings of every face in the photo ([] if unreadable)."""
    try:
        return [face[0] for face in read_faces(app, image_path)]
    except IOError:
        return []


def main():
    p = argparse.ArgumentParser(description="Find photos of a person using reference photos.")
    p.add_argument("--ref", required=True, help="Folder with reference photos of the person")
    p.add_argument("--photos", required=True, help="Folder of photos to search")
    p.add_argument("--out", required=True, help="Folder where matches are copied")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Similarity 0..1 (default 0.40). Raise to reduce false matches, lower to catch more.")
    args = p.parse_args()

    import face_model
    if not face_model.models_ready():
        print("First run: downloading the face models (~%d MB, once only)..."
              % face_model.DOWNLOAD_MB)
    face_model.ensure_models(log=print)
    app = face_model.FaceModel(det_size=640)

    ref_files = list_images(args.ref)
    if not ref_files:
        sys.exit(f"No images found in reference folder: {args.ref}")
    ref_embs = []
    for fp in ref_files:
        ref_embs.extend(face_embeddings(app, fp))
    if not ref_embs:
        sys.exit("No faces detected in the reference photos. Use clearer, front-facing shots.")
    ref_matrix = np.stack(ref_embs)
    print(f"Reference: {len(ref_files)} photo(s), {len(ref_embs)} face(s) enrolled.")

    photo_files = list_images(args.photos)
    if not photo_files:
        sys.exit(f"No images found in photos folder: {args.photos}")
    print(f"Searching {len(photo_files)} photos...")
    os.makedirs(args.out, exist_ok=True)

    matches = 0
    for i, fp in enumerate(photo_files, 1):
        embs = face_embeddings(app, fp)
        hit = False
        for e in embs:
            sims = ref_matrix @ e  # cosine similarity (embeddings are normalized)
            if float(np.max(sims)) >= args.threshold:
                hit = True
                break
        if hit:
            dest = os.path.join(args.out, os.path.basename(fp))
            base, ext = os.path.splitext(dest)
            n = 1
            while os.path.exists(dest):
                n += 1
                dest = f"{base}_{n}{ext}"
            shutil.copy2(fp, dest)
            matches += 1
        if i % 100 == 0 or i == len(photo_files):
            print(f"  ...{i}/{len(photo_files)} scanned, {matches} matches so far")

    print(f"Done. {matches} matching photo(s) copied to {args.out}")
    print("Tip: too many wrong matches? Re-run with --threshold 0.50. Missing some? Try --threshold 0.30.")


if __name__ == "__main__":
    main()
