"""calibrate.py -- measure how well the face model separates people on YOUR
photos, and suggest strictness thresholds for Face Search.

Make a folder with one subfolder per person (3 or more people, 5 or more
photos each; photos where that person is the only face, or clearly the
biggest one):

    calibration\\
        person_a\\  (5-20 photos)
        person_b\\
        person_c\\

Then, from the Face Search folder:

    .venv\\Scripts\\python tools\\calibrate.py C:\\path\\to\\calibration

Everything runs on this computer. The report contains only numbers: no
names, file names or photos. It's also saved to calibration_results.txt.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from face_search import list_images, read_faces  # noqa: E402
import search_engine as se  # noqa: E402

THRESHOLDS = [round(0.20 + 0.025 * i, 3) for i in range(17)]  # 0.20 .. 0.60


def pick_face(faces):
    """The biggest face in the photo (calibration photos should be solo)."""
    return max(faces, key=lambda f: f[1])[0] if faces else None


def collect(model, root, log):
    people = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        embs, multi, none = [], 0, 0
        for fp in list_images(folder):
            try:
                faces = read_faces(model, fp)
            except Exception:
                none += 1
                continue
            if not faces:
                none += 1
                continue
            if len(faces) > 1:
                multi += 1
            embs.append(pick_face(faces))
        log("  person %d: %d photo(s) used%s%s" % (
            len(people) + 1, len(embs),
            ", %d with several faces (biggest used)" % multi if multi else "",
            ", %d without a usable face" % none if none else ""))
        if len(embs) >= 2:
            people.append(np.stack(embs))
    return people


def scores(people):
    """Like the app: each photo vs. the rest of its own person's photos
    (should match) and vs. every other person's photos (shouldn't)."""
    same, other = [], []
    for i, mat in enumerate(people):
        for k in range(len(mat)):
            e = mat[k]
            rest = np.delete(mat, k, axis=0)
            same.append(float(np.max(rest @ e)))
            for j, mat2 in enumerate(people):
                if j != i:
                    other.append(float(np.max(mat2 @ e)))
    return np.array(same), np.array(other)


def lowest_threshold(other, max_false):
    for t in THRESHOLDS:
        if np.mean(other >= t) <= max_false:
            return t
    return THRESHOLDS[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="folder with one subfolder of photos per person")
    ap.add_argument("--detail", default="standard", choices=sorted(se.DET_SIZES))
    ap.add_argument("--cpu", action="store_true", help="don't use the graphics card")
    args = ap.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    if not se.model_ready():
        print("Downloading the face models (~%d MB, once only)..." % se.MODEL_DOWNLOAD_MB)
    t0 = time.time()
    model = se.load_model(detail=args.detail, use_gpu=not args.cpu)
    log("Face Search calibration -- model %s, detail %s, %s"
        % (se.face_model.MODEL_TAG, args.detail, model.fs_device))
    log("Reading photos...")
    people = collect(model, args.folder, log)
    if len(people) < 2:
        sys.exit("Need at least 2 people with 2+ usable photos each (3+ people and "
                 "5+ photos each give useful numbers).")
    same, other = scores(people)
    log("")
    log("%d people, %d same-person checks, %d different-person checks (%.0f s)"
        % (len(people), len(same), len(other), time.time() - t0))
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    log("same person:      min %.3f | 5%% %.3f | 25%% %.3f | median %.3f"
        % (same.min(), pct(same, 5), pct(same, 25), pct(same, 50)))
    log("different person: median %.3f | 95%% %.3f | 99%% %.3f | max %.3f"
        % (pct(other, 50), pct(other, 95), pct(other, 99), other.max()))
    log("")
    log("threshold | right person found | wrong person matched")
    for t in THRESHOLDS:
        log("  %.3f   |      %5.1f%%        |      %5.2f%%"
            % (t, 100 * np.mean(same >= t), 100 * np.mean(other >= t)))
    loose = lowest_threshold(other, 0.05)
    balanced = lowest_threshold(other, 0.01)
    strict = lowest_threshold(other, 0.002)
    log("")
    log("Suggested: loose %.3f | balanced %.3f | strict %.3f   (app now: %.3f)"
        % (loose, balanced, strict, se.THRESHOLD_DEFAULT))
    if np.median(same) < 0.25 or np.median(same) <= np.median(other) + 0.1:
        log("WARNING: same-person scores are unusually low. Something may be wrong "
            "with the photos or the model setup.")
    out = os.path.join(os.getcwd(), "calibration_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved to %s. Send its contents to Claude to tune the app." % out)


if __name__ == "__main__":
    main()
