#!/usr/bin/env python3
"""
face_search.py -- search a folder of photos for a specific person.

You give it reference photos of the person; it copies every photo in the
search folder that appears to contain them into an output folder.
Fully offline, runs on your own machine (InsightFace, CPU).

Setup (Windows):
    py -3.12 -m venv .venv
    .venv\\Scripts\\activate
    pip install insightface onnxruntime opencv-python numpy

Usage:
    python face_search.py --ref .\\refs\\person1 --photos .\\all_photos --out .\\matches_person1
    python face_search.py --ref .\\refs\\person1 --photos .\\all_photos --out .\\matches_person1 --threshold 0.55

First run downloads the recognition model (~300 MB) into %USERPROFILE%\\.insightface.
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def face_embeddings(app, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = app.get(img)
    return [f.normed_embedding.astype(np.float32) for f in faces]


def main():
    p = argparse.ArgumentParser(description="Find photos of a person using reference photos.")
    p.add_argument("--ref", required=True, help="Folder with reference photos of the person")
    p.add_argument("--photos", required=True, help="Folder of photos to search")
    p.add_argument("--out", required=True, help="Folder where matches are copied")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="Similarity 0..1 (default 0.45). Raise to reduce false matches, lower to catch more.")
    args = p.parse_args()

    print("Loading face model (first run downloads ~300 MB)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

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
    print("Tip: too many wrong matches? Re-run with --threshold 0.55. Missing some? Try --threshold 0.35.")


if __name__ == "__main__":
    main()
