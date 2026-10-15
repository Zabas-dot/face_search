"""compare_matches.py -- tune the new face model using matches you already
know are right (e.g. the ones InsightFace / Face Search 1.x found).

    .venv\\Scripts\\python tools\\compare_matches.py ^
        --ref   C:\\path\\to\\reference_photos ^
        --good  C:\\path\\to\\known_good_matches ^
        --other C:\\path\\to\\new_matches_folder

--ref    the same reference folder you search with (one person)
--good   a folder holding only photos that really are that person
--other  optional: the new (2.0) matches folder; photos in it that aren't in
         --good are treated as "probably wrong"

Prints how the new model scores the known-good photos versus the others,
and which threshold keeps the good ones while dropping the rest. Also
checks the model setup. Only numbers are printed: no names or photos.
Saved to compare_results.txt as well.
"""

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

import face_model  # noqa: E402
import search_engine as se  # noqa: E402
from face_search import list_images, read_faces  # noqa: E402

THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(23)]  # 0.30 .. 0.85


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def best_scores(model, ref_matrix, files, per_photo=None):
    out = []
    for fp in files:
        try:
            faces = read_faces(model, fp)
        except Exception as exc:
            if per_photo is not None:
                per_photo.append("error (%s)" % type(exc).__name__)
            continue
        if faces:
            embs = np.stack([f[0] for f in faces])
            sims = (ref_matrix @ embs.T).max(axis=0)   # best score per face
            out.append(float(sims.max()))
            if per_photo is not None:
                best = int(np.argmax(sims))
                side = (faces[best][1]) ** 0.5   # face width in pixels (approx.)
                per_photo.append("%d face(s) found | best score %.3f | that face ~%d px wide"
                                 % (len(faces), float(sims.max()), side))
        else:
            out.append(None)
            if per_photo is not None:
                per_photo.append("NO faces found")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--good", required=True)
    ap.add_argument("--other")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--detail", default="standard", choices=sorted(se.DET_SIZES),
                    help="scan detail to test (thorough finds smaller faces)")
    args = ap.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    model = se.load_model(detail=args.detail, use_gpu=not args.cpu)
    rec = face_model.model_path("recognizer")
    log("model %s, detail %s, on %s | input mean %.1f std %.1f | input %s"
        % (face_model.MODEL_TAG, args.detail, model.fs_device, model.mean, model.std,
           model.session.get_inputs()[0].shape))
    try:
        import onnx
        names = [n.op_type + ":" + n.name for n in
                 onnx.load(rec, load_external_data=False).graph.node[:6]]
        log("first graph nodes: " + ", ".join(names))
    except Exception as exc:
        log("couldn't inspect the model graph (%s)" % exc)

    people = se.enroll_references(model, args.ref)
    ref = people[""] if "" in people else np.vstack(
        [v for k, v in people.items() if not k.startswith("_")])
    info = people.get("_ref_info", {})
    log("reference faces: %d (from %d photos)" % (len(ref), info.get("photos", 0)))
    if len(ref) > 1:
        sims = ref @ ref.T
        off = sims[~np.eye(len(ref), dtype=bool)]
        log("reference faces vs each other: min %.3f | median %.3f"
            % (off.min(), np.median(off)))

    good_files = list_images(args.good)
    good_hashes = {file_hash(p) for p in good_files}
    per_photo = []
    good = best_scores(model, ref, good_files, per_photo)
    no_face = sum(1 for s in good if s is None)
    good = np.array([s for s in good if s is not None])
    log("")
    log("known-good photos: %d (%d where no face was found)" % (len(good) + no_face, no_face))
    if len(good):
        log("  scores: min %.3f | 10%% %.3f | median %.3f | max %.3f"
            % (good.min(), np.percentile(good, 10), np.median(good), good.max()))
    log("  each known-good photo (in name order, names not shown):")
    for i, line in enumerate(per_photo, 1):
        log("    photo %2d: %s" % (i, line))

    other = np.array([])
    if args.other:
        other_files = [p for p in list_images(args.other) if file_hash(p) not in good_hashes]
        other = np.array([s for s in best_scores(model, ref, other_files) if s is not None])
        log("other matches (not in the known-good set): %d" % len(other))
        if len(other):
            log("  scores: min %.3f | median %.3f | 90%% %.3f | max %.3f"
                % (other.min(), np.median(other), np.percentile(other, 90), other.max()))

    log("")
    log("threshold | known-good kept | others kept")
    for t in THRESHOLDS:
        log("  %.3f   |   %3d of %3d    | %4s" % (
            t, int(np.sum(good >= t)), len(good),
            "%d" % int(np.sum(other >= t)) if len(other) else "-"))
    if len(good):
        keep90 = float(np.percentile(good, 10))
        log("")
        log("A threshold of about %.3f keeps 90%% of the known-good photos." % keep90)
        if len(other) and np.median(good) < np.median(other) + 0.05:
            log("WARNING: known-good photos don't score clearly higher than the others. "
                "That points to a setup problem, not just the threshold.")
    out = os.path.join(os.getcwd(), "compare_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved to %s. Send its contents to Claude." % out)


if __name__ == "__main__":
    main()
