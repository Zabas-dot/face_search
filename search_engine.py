"""search_engine.py -- reusable backend for the face_search GUI.

It uses the exact same search algorithm as face_search.py: InsightFace
buffalo_l embeddings, cosine similarity against the reference embeddings,
match when similarity >= threshold.

Additions the CLI doesn't have:

1. EMBEDDING CACHE -- photo embeddings are saved to
   ~/.face_search_cache/ keyed by file path + modification time + size.
   Photos that haven't changed since the last scan are NOT re-analyzed, so
   a repeat search over the same folder takes seconds instead of tens of
   minutes. Each scan detail (quick, standard, thorough) keeps its own cache, because
   their embeddings differ. The file's content hash is cached too, so
   repeat runs don't re-read every matching photo from disk. Entries for
   photos that were deleted are pruned after each complete run.

2. PROGRESS CALLBACK -- the GUI gets live (done, total, matches) updates
   for its progress bar, plus a cancel event.

3. OUTPUT HOUSEKEEPING -- duplicate copies the app made are removed, a
   manifest tracks which files the app placed (so raising the threshold
   removes matches that no longer qualify), and a per-person blocklist
   remembers photos the user marked "Not a match". Files the user put in
   the output folder themselves are never deleted. The manifest stores
   paths relative to the output folder, so moving or copying the folder is
   safe.

4. SAFER REFERENCES -- in single-person and per-subfolder modes, a group
   photo in the reference folder contributes only the face of the person
   being searched for, not every stranger in the shot.

Results are identical to the CLI for the same threshold and detection size
(except that the CLI enrolls every face of a group reference photo).
"""

import hashlib
import json
import os
import pickle
import shutil
import time
import traceback

import numpy as np
from insightface.app import FaceAnalysis

# Same detection code as the CLI.
from face_search import list_images, read_faces

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".face_search_cache")
MANIFEST_NAME = ".face_search_manifest.json"
BLOCKLIST_NAME = ".face_search_blocklist.json"
INDEX_NAME = ".face_search_index.json"
UNREADABLE_NAME = "unreadable_photos.txt"
_META_FILES = {MANIFEST_NAME, BLOCKLIST_NAME, INDEX_NAME, UNREADABLE_NAME}

DET_NORMAL = 640
DET_FAST = 320
# Scan detail -> detector input size. Bigger finds smaller faces but is slower.
DET_SIZES = {"quick": DET_FAST, "standard": DET_NORMAL, "thorough": 1280}
_CACHE_SUFFIX = {"quick": "_fast", "standard": "", "thorough": "_thorough"}
# Graphics-card back ends, best first (DirectML works with most Windows GPUs).
GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
NO_FACES_MSG = "No faces detected in the reference photos. Use clearer, front-facing shots."

CACHE_CHECKPOINT_SECS = 60   # full scan cache (can be large)
STATE_CHECKPOINT_SECS = 10   # manifest + output index (small)
REF_PICK_MIN_SIM = 0.30      # group reference photo: min similarity to keep a face


# --------------------------------------------------------------------------
# Scan cache
# --------------------------------------------------------------------------

def _detail(fast=False, detail=None):
    return detail if detail in DET_SIZES else ("quick" if fast else "standard")


def _cache_path(photos_folder, fast=False, detail=None):
    digest = hashlib.sha1(os.path.abspath(photos_folder).encode("utf-8")).hexdigest()[:16]
    # Standard detail keeps the original file name so existing caches still work.
    return os.path.join(CACHE_DIR, digest + _CACHE_SUFFIX[_detail(fast, detail)] + ".pkl")


def load_cache(photos_folder, fast=False, detail=None):
    """Return {abspath: {"mtime", "size", "embs", "sha"?, "bad"?}}."""
    path = _cache_path(photos_folder, fast, detail)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache(photos_folder, cache, fast=False, detail=None):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        final = _cache_path(photos_folder, fast, detail)
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(cache, f, protocol=4)
        os.replace(tmp, final)
    except Exception:
        pass  # cache is a pure speedup; never fail the search over it


# --------------------------------------------------------------------------
# Model and reference enrollment
# --------------------------------------------------------------------------

def _available_providers():
    try:
        import onnxruntime
        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def gpu_available():
    """True if this onnxruntime build can use a graphics card."""
    return any(p in _available_providers() for p in GPU_PROVIDERS)


def _make_app(det, providers):
    # Only face detection + recognition are used. InsightFace otherwise also
    # runs 3 more models (landmarks, age/gender) on every face found; the
    # embeddings are identical either way because recognition aligns faces
    # with the detector's 5 key points.
    try:
        app = FaceAnalysis(name=MODEL_NAME, providers=providers,
                           allowed_modules=["detection", "recognition"])
    except TypeError:  # very old InsightFace without allowed_modules
        app = FaceAnalysis(name=MODEL_NAME, providers=providers)
    app.prepare(ctx_id=0, det_size=(det, det))
    return app


def _warm_up(app, det):
    """Run both models once, so a graphics card that can't run them fails
    here (and we fall back to the processor) instead of mid-search."""
    app.get(np.zeros((det, det, 3), dtype=np.uint8))
    rec = getattr(app, "models", {}).get("recognition")
    if rec is not None and hasattr(rec, "get_feat"):
        rec.get_feat(np.zeros((112, 112, 3), dtype=np.uint8))


def _session_on_gpu(app):
    try:
        first = app.det_model.session.get_providers()[0]
        return first in GPU_PROVIDERS
    except Exception:
        return False


def load_model(fast=False, detail=None, use_gpu=False):
    """Prepare the InsightFace model.

    detail: "quick" (320 px detector, ~3-4x faster first scan, misses small
    faces), "standard" (640 px) or "thorough" (1280 px, finds small faces in
    group shots and large photos, ~3x slower). fast=True means "quick".
    use_gpu: try the graphics card first; falls back to the processor if it
    isn't available or fails. The returned app has .fs_device set to
    "graphics card" or "processor" and .fs_detail to the detail used.
    """
    detail = _detail(fast, detail)
    det = DET_SIZES[detail]
    repair_model_folder()
    _install_download_hook()
    app = None
    if use_gpu:
        gpu = [p for p in GPU_PROVIDERS if p in _available_providers()]
        if gpu:
            try:
                candidate = _make_app(det, [gpu[0], "CPUExecutionProvider"])
                _warm_up(candidate, det)
                if _session_on_gpu(candidate):
                    app = candidate
                    app.fs_device = "graphics card"
            except Exception:
                app = None
    if app is None:
        try:
            app = _make_app(det, ["CPUExecutionProvider"])
        except Exception:
            repair_model_folder()  # don't leave a broken half-download behind
            raise
        app.fs_device = "processor"
    app.fs_detail = detail
    return app


# ---- First-run model download --------------------------------------------

MODEL_NAME = "buffalo_l"
MODEL_ROOT = os.path.join(os.path.expanduser("~"), ".insightface", "models")
_MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")  # detection + recognition
MODEL_DOWNLOAD_MB = 280  # approximate, shown before the size is known

# Filled in while the model downloads: {"done": bytes, "total": bytes or 0}
download_progress = {"done": 0, "total": 0}


def _model_dir():
    return os.path.join(MODEL_ROOT, MODEL_NAME)


def model_ready():
    """True if the face model is already downloaded (the folder holds model
    files -- either the classic layout or InsightFace's package layout)."""
    d = _model_dir()
    if not os.path.isdir(d):
        return False
    if all(os.path.isfile(os.path.join(d, f)) for f in _MODEL_FILES):
        return True
    try:  # InsightFace 2.x package layout
        from insightface.model_zoo.package_manifest import has_model_package_manifest
        return bool(has_model_package_manifest(d))
    except Exception:
        # Can't tell: treat any model file as "present" so a working
        # model is never deleted.
        for _root, _dirs, files in os.walk(d):
            if any(f.lower().endswith(".onnx") for f in files):
                return True
        return False


def repair_model_folder():
    """A download interrupted while unpacking can leave an empty model
    folder, and InsightFace then fails on every launch because it only
    checks that the folder exists. Remove it (only if it holds no model
    files at all) so the download runs again."""
    d = _model_dir()
    if os.path.isdir(d) and not model_ready():
        shutil.rmtree(d, ignore_errors=True)


def _install_download_hook():
    """Route InsightFace's console progress bar into download_progress."""
    try:
        import insightface.utils.download as dl
    except Exception:
        return

    def _progress(iterable, total=None, **_kw):
        download_progress["done"] = 0
        download_progress["total"] = int(total or 0) * 1024
        for chunk in iterable:
            download_progress["done"] += len(chunk)
            yield chunk

    dl.tqdm = _progress


def friendly_model_error(exc):
    """Turn a model load/download failure into advice for the user."""
    text = "%s: %s" % (type(exc).__name__, exc)
    lowered = text.lower()
    if not model_ready() and any(w in lowered for w in (
            "connection", "resolve", "timeout", "timed out", "network",
            "failed downloading", "ssl", "proxy", "unreachable", "getaddrinfo")):
        return ("Couldn't download the face model. Check that you're connected "
                "to the internet and press Run search again.\n(The ~%d MB "
                "download is needed only once.)" % MODEL_DOWNLOAD_MB)
    if not model_ready():
        return ("The face model download didn't finish. Press Run search to try "
                "again.\n(Details: %s)" % text)
    return "Couldn't load the face model: %s" % text


def _write_diag_and_raise(app, errors, tb_text):
    """All reference photos failed to load: write a diagnostic log and raise."""
    try:
        import insightface, onnxruntime, cv2
        diag = ("insightface %s, onnxruntime %s, opencv %s\n"
                % (getattr(insightface, "__version__", "?"),
                   getattr(onnxruntime, "__version__", "?"),
                   getattr(cv2, "__version__", "?")))
    except Exception:
        diag = "version check failed\n"
    diag += "det_model: %s\n" % type(getattr(app, "det_model", None)).__name__
    diag += "\n".join(errors) + "\n\n" + tb_text
    log_path = os.path.join(os.path.expanduser("~"), "face_search_diag.log")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(diag)
    except Exception:
        log_path = "(could not write log)"
    raise ValueError("Couldn't read the reference photos: %s\nFull details in %s"
                     % ("; ".join(errors[:2]), log_path))


def _read_reference_faces(app, files):
    """Return ([(file, [(emb, area, bbox), ...]), ...], info). Raises ValueError
    with diagnostics if no face was found and at least one file errored."""
    per_file, errors, unreadable, no_face = [], [], [], []
    tb_text = ""
    for fp in files:
        try:
            faces = read_faces(app, fp)
        except Exception as exc:
            errors.append("%s: %s" % (os.path.basename(fp), exc))
            unreadable.append(fp)
            if not tb_text and not isinstance(exc, IOError):
                tb_text = traceback.format_exc()
            continue
        if faces:
            per_file.append((fp, [(np.asarray(f[0], dtype=np.float32), f[1], f[2])
                                  for f in faces]))
        else:
            no_face.append(fp)
    if not per_file and errors:
        _write_diag_and_raise(app, errors, tb_text)
    return per_file, {"unreadable": unreadable, "no_face": no_face}


def _pick_target_faces(per_file):
    """One person's reference photos -> that person's faces only.

    Photos with a single face are trusted as-is. For a group photo, keep the
    face that best resembles the single-face photos (if any is close enough);
    without single-face photos, keep the face that recurs most across the
    group photos; with just one group photo, keep the largest face.
    Returns (embeddings, [(source file, bbox)] per embedding, faces_ignored).
    """
    embs, emb_files = [], []
    ignored = 0
    anchors = [faces[0][0] for _fp, faces in per_file if len(faces) == 1]
    anchor_matrix = np.stack(anchors) if anchors else None
    groups = [(fp, faces) for fp, faces in per_file if len(faces) > 1]
    for fp, faces in per_file:
        if len(faces) == 1:
            embs.append(faces[0][0])
            emb_files.append((fp, faces[0][2]))
            continue
        if anchor_matrix is not None:
            scores = [float(np.max(anchor_matrix @ f[0])) for f in faces]
            best = int(np.argmax(scores))
            if scores[best] < REF_PICK_MIN_SIM:
                ignored += len(faces)
                continue
        elif len(groups) > 1:
            others = [np.stack([f[0] for f in f2]) for fp2, f2 in groups if fp2 != fp]
            scores = [sum(float(np.max(m @ f[0])) for m in others) for f in faces]
            best = int(np.argmax(scores))
        else:
            best = int(np.argmax([f[1] for f in faces]))
        embs.append(faces[best][0])
        emb_files.append((fp, faces[best][2]))
        ignored += len(faces) - 1
    return embs, emb_files, ignored


def _cluster_embeddings(embs, threshold=0.35):
    """Greedy clustering of normalized face embeddings by cosine similarity.

    Returns a list of clusters (biggest first); each cluster is a list of
    indices into embs. Uses a lenient threshold since reference folders are
    user-curated -- better to merge than to split one person.
    """
    clusters = []  # list of [centroid, [indices]]
    for i, e in enumerate(embs):
        best_idx, best_sim = -1, -1.0
        for ci, (centroid, _members) in enumerate(clusters):
            sim = float(np.dot(centroid, e))
            if sim > best_sim:
                best_sim, best_idx = sim, ci
        if best_idx >= 0 and best_sim >= threshold:
            centroid, members = clusters[best_idx]
            members.append(i)
            n = len(members)
            new_centroid = centroid * ((n - 1) / n) + e / n
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            clusters[best_idx] = [new_centroid, members]
        else:
            clusters.append([e.copy(), [i]])

    # Merge pass: combine clusters whose centroids are similar enough.
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        best_pair, best_sim = None, -1.0
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                sim = float(np.dot(clusters[a][0], clusters[b][0]))
                if sim > best_sim:
                    best_sim, best_pair = sim, (a, b)
        if best_pair and best_sim >= 0.30:
            a, b = best_pair
            ca, ma = clusters[a]
            cb, mb = clusters[b]
            na, nb = len(ma), len(mb)
            ma.extend(mb)
            new_centroid = (ca * na + cb * nb) / (na + nb)
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            clusters[a] = [new_centroid, ma]
            del clusters[b]
            merged = True

    # Biggest group first, so "person1" is the person with the most photos
    # and the numbering is stable between runs.
    clusters.sort(key=lambda c: -len(c[1]))
    return [members for _, members in clusters]


def enroll_references(app, ref_dir, auto_cluster=False):
    """Enroll reference faces. Returns {person_name: embedding_matrix}.

    If ref_dir contains subfolders with images, each subfolder is a separate
    person (named after the folder). Otherwise, if auto_cluster is True, all
    faces are pooled and automatically clustered into distinct people
    (person1_matches, ...) and the key "_cluster_previews" holds the source
    files of each cluster. If False (default), all photos are one person
    (key "" meaning output goes directly to out_dir).

    Metadata keys (start with "_", not people):
      "_ref_info": {"faces_ignored": n, "unreadable": [paths], "no_face": [paths],
                    "photos": number of reference photos}
      "_faces": {person_name: [(photo path, face box), ...]} -- the faces used,
                for showing the user who will be searched for.
    """
    person_dirs = []
    try:
        for entry in sorted(os.listdir(ref_dir)):
            full = os.path.join(ref_dir, entry)
            if os.path.isdir(full) and list_images(full):
                person_dirs.append((entry, full))
    except OSError:
        pass

    people = {}
    faces_used = {}
    info = {"faces_ignored": 0, "unreadable": [], "no_face": [], "photos": 0}
    if person_dirs:
        # Manual mode: subfolders name the people.
        for name, path in person_dirs:
            files = list_images(path)
            info["photos"] += len(files)
            per_file, finfo = _read_reference_faces(app, files)
            info["unreadable"] += finfo["unreadable"]
            info["no_face"] += finfo["no_face"]
            embs, used, ignored = _pick_target_faces(per_file)
            info["faces_ignored"] += ignored
            if embs:
                people[name] = np.stack(embs)
                faces_used[name] = used
        if not people:
            raise ValueError(NO_FACES_MSG)
        people["_ref_info"] = info
        people["_faces"] = faces_used
        return people

    files = list_images(ref_dir)
    info["photos"] = len(files)
    per_file, finfo = _read_reference_faces(app, files)
    info["unreadable"] = finfo["unreadable"]
    info["no_face"] = finfo["no_face"]
    if not per_file:
        raise ValueError(NO_FACES_MSG)

    if not auto_cluster:
        embs, used, info["faces_ignored"] = _pick_target_faces(per_file)
        people[""] = np.stack(embs)
        people["_ref_info"] = info
        people["_faces"] = {"": used}
        return people

    # Auto-separate: every face counts, group photos included.
    embs, emb_src = [], []
    for fp, faces in per_file:
        for e, _a, box in faces:
            embs.append(e)
            emb_src.append((fp, box))
    clusters = _cluster_embeddings(embs)
    cluster_files = []
    for members in clusters:
        seen = []
        for i in members:
            if emb_src[i][0] not in seen:
                seen.append(emb_src[i][0])
        cluster_files.append(seen)
    if len(clusters) == 1:
        people[""] = np.stack([embs[i] for i in clusters[0]])
        faces_used[""] = [emb_src[i] for i in clusters[0]]
    else:
        for idx, members in enumerate(clusters, 1):
            name = "person%d_matches" % idx
            people[name] = np.stack([embs[i] for i in members])
            faces_used[name] = [emb_src[i] for i in members]
    people["_cluster_previews"] = cluster_files
    people["_ref_info"] = info
    people["_faces"] = faces_used
    return people


# --------------------------------------------------------------------------
# Output-folder bookkeeping
# --------------------------------------------------------------------------

def _unique_dest(out_dir, filename):
    dest = os.path.join(out_dir, filename)
    base, ext = os.path.splitext(dest)
    n = 1
    while os.path.exists(dest):
        n += 1
        dest = "%s_%d%s" % (base, n, ext)
    return dest


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(path):
    """Comparison key for a path (Windows paths are case-insensitive)."""
    return os.path.normcase(os.path.abspath(path))


def _inside(path, folder):
    """True if path is inside folder (both may be relative or absolute)."""
    p, f = _norm(path), _norm(folder)
    return p == f or p.startswith(f.rstrip(os.sep) + os.sep)


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, type(default)) else default
    except (OSError, ValueError):
        return default


def _save_json(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass


def _load_manifest(out_dir):
    """Files the app placed on the previous run, as {normalized: abspath}.

    Entries are stored relative to out_dir. Absolute entries (written by
    1.59/1.60) are accepted only when they point inside this out_dir, so a
    manifest that was copied along with the folder can never point the
    cleanup at the original folder.
    """
    out = {}
    for p in _load_json(os.path.join(out_dir, MANIFEST_NAME), []):
        if not isinstance(p, str):
            continue
        full = p if os.path.isabs(p) else os.path.join(out_dir, p)
        if _inside(full, out_dir):
            full = os.path.abspath(full)
            out[_norm(full)] = full
    return out


def _save_manifest(out_dir, manifest):
    rel = sorted(os.path.relpath(p, out_dir) for p in manifest.values())
    _save_json(os.path.join(out_dir, MANIFEST_NAME), rel)


class _OutputIndex:
    """Content hashes of files in the output folder, keyed by path relative
    to out_dir and validated by (mtime, size), so repeat runs don't re-read
    every file already in the output folder."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, INDEX_NAME)
        self.data = _load_json(self.path, {})
        self.used = set()

    def _key(self, fp):
        return os.path.relpath(fp, self.out_dir)

    def hash_of(self, fp):
        st = os.stat(fp)
        key = self._key(fp)
        self.used.add(key)
        rec = self.data.get(key)
        if isinstance(rec, list) and len(rec) == 3 and \
                rec[0] == st.st_mtime and rec[1] == st.st_size:
            return rec[1], rec[2]
        sha = _file_hash(fp)
        self.data[key] = [st.st_mtime, st.st_size, sha]
        return st.st_size, sha

    def record(self, fp, sha):
        try:
            st = os.stat(fp)
        except OSError:
            return
        key = self._key(fp)
        self.used.add(key)
        self.data[key] = [st.st_mtime, st.st_size, sha]

    def forget(self, fp):
        self.data.pop(self._key(fp), None)

    def save(self, prune=False):
        if prune:
            self.data = {k: v for k, v in self.data.items() if k in self.used}
        _save_json(self.path, self.data)


def _dedupe_and_hash(folder, app_placed, index):
    """Index the files directly in `folder` by (size, content hash) and remove
    duplicate copies that the app itself placed (keys of app_placed).

    Only the top level is scanned: matches are always copied to the top level
    of a person's folder, and each person's folder is indexed separately.
    Files the user added are never removed; if a duplicate pair includes a
    user file, the app's copy is the one removed.

    Returns ({(size, hash): filepath}, number_removed).
    """
    seen = {}
    removed = 0
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return seen, removed
    files = [os.path.join(folder, n) for n in names
             if n not in _META_FILES and not n.endswith(".tmp")
             and os.path.isfile(os.path.join(folder, n))]
    # User files first, so they become the copy that is kept.
    files.sort(key=lambda p: _norm(p) in app_placed)
    for fp in files:
        try:
            key = index.hash_of(fp)
        except OSError:
            continue
        if key not in seen:
            seen[key] = fp
        elif _norm(fp) in app_placed:
            try:
                os.remove(fp)
                index.forget(fp)
                removed += 1
            except OSError:
                pass
    return seen, removed


def _load_blocklist(folder):
    """Content hashes the user rejected for the person whose folder this is."""
    return set(_load_json(os.path.join(folder, BLOCKLIST_NAME), []))


def blocklist_add(folder, file_hash):
    """Remember that the user rejected this photo for the person whose
    matches live in `folder` (the folder the match was copied into)."""
    bl = _load_blocklist(folder)
    bl.add(file_hash)
    _save_json(os.path.join(folder, BLOCKLIST_NAME), sorted(bl))


def forget_match(out_dir, path):
    """Drop a file from the manifest after the user deleted it."""
    manifest = _load_manifest(out_dir)
    if manifest.pop(_norm(path), None) is not None:
        _save_manifest(out_dir, manifest)


def add_reference(ref_dir, person, photo, subfolders=False):
    """Copy a confirmed match into the reference photos, so later searches
    also find photos that look like it. In subfolder mode the copy goes into
    the person's subfolder. Returns the new file's path, or None if the same
    photo is already among the reference photos."""
    folder = os.path.join(ref_dir, person) if (subfolders and person) else ref_dir
    os.makedirs(folder, exist_ok=True)
    sha = _file_hash(photo)
    for existing in list_images(folder):
        try:
            if os.path.getsize(existing) == os.path.getsize(photo) and \
                    _file_hash(existing) == sha:
                return None
        except OSError:
            continue
    dest = _unique_dest(folder, "added_" + os.path.basename(photo))
    shutil.copy2(photo, dest)
    return dest


def remove_reference(path):
    """Undo add_reference."""
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _write_unreadable(out_dir, paths):
    """List photos that couldn't be read in out_dir/unreadable_photos.txt
    (removed again when a later run has none)."""
    target = os.path.join(out_dir, UNREADABLE_NAME)
    try:
        if paths:
            with open(target, "w", encoding="utf-8") as f:
                f.write("These photos could not be opened and were not searched.\n"
                        "They may be damaged, incomplete, or in an unsupported format.\n\n")
                f.write("\n".join(os.path.normpath(p) for p in paths) + "\n")
        elif os.path.isfile(target):
            os.remove(target)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def run_search(app, ref_matrices, photo_files, out_dir, photos_folder,
               threshold=0.45, use_cache=True,
               progress_cb=None, cancel_event=None, fast=False, detail=None):
    """Scan photo_files, copy matches into out_dir.

    ref_matrices: {person_name: embedding_matrix}. A photo matching a person
    is copied to out_dir/person_name/ (or out_dir/ if name is "").
    detail (or fast=True for "quick"): must match the scan detail the model
    was loaded with; it picks which scan cache to use.
    progress_cb(done, total, matches, from_cache) is called after each photo.
    Returns {"cancelled": bool, "matches": {person: [(path, score)]}, ...}
    with each person's matches sorted best first.
    """
    ref_matrices = {k: v for k, v in ref_matrices.items() if not k.startswith("_")}
    detail = _detail(fast, detail)
    os.makedirs(out_dir, exist_ok=True)
    person_dirs = {}
    for name in ref_matrices:
        person_dirs[name] = os.path.join(out_dir, name) if name else out_dir
        os.makedirs(person_dirs[name], exist_ok=True)

    old_manifest = _load_manifest(out_dir)
    new_manifest = {}
    index = _OutputIndex(out_dir)

    # Per-person indexes: content already in each folder, and rejections.
    legacy_blocklist = _load_blocklist(out_dir)  # pre-1.60 kept one shared list
    contents, blocklists = {}, {}
    cleaned = 0
    for name, folder in person_dirs.items():
        seen, removed = _dedupe_and_hash(folder, old_manifest, index)
        contents[name] = seen
        blocklists[name] = _load_blocklist(folder) | legacy_blocklist
        cleaned += removed

    cache = load_cache(photos_folder, detail=detail) if use_cache else {}
    cache_dirty = False
    counts = {"cache_hits": 0, "skipped": 0, "duplicates": 0, "blocklisted": 0}
    unreadable = []
    matches = {name: [] for name in ref_matrices}
    match_scores = {}
    total_matches = 0
    total = len(photo_files)
    last_cache_save = last_state_save = time.monotonic()

    def _manifest_with_old():
        """New entries plus earlier files not (yet) revisited, so an
        interrupted run never loses track of what the app placed."""
        merged = dict(new_manifest)
        for k, p in old_manifest.items():
            if k not in merged and os.path.isfile(p):
                merged[k] = p
        return merged

    def _result(cancelled, stale_removed):
        for name in matches:
            matches[name].sort(key=lambda m: -m[1])  # best matches first
        res = {"cancelled": cancelled, "matches": matches, "cleaned": cleaned,
               "stale_removed": stale_removed, "match_scores": match_scores,
               "unreadable": unreadable}
        res.update(counts)
        return res

    for i, fp in enumerate(photo_files, 1):
        if cancel_event is not None and cancel_event.is_set():
            if use_cache and cache_dirty:
                save_cache(photos_folder, cache, detail=detail)
            _save_manifest(out_dir, _manifest_with_old())
            index.save()
            return _result(True, [])

        key = os.path.abspath(fp)
        try:
            st = os.stat(fp)
            sig = (st.st_mtime, st.st_size)
        except OSError:
            sig = None

        entry = cache.get(key) if use_cache else None
        if entry is not None and sig is not None and \
                entry.get("mtime") == sig[0] and entry.get("size") == sig[1]:
            embs = entry["embs"]
            counts["cache_hits"] += 1
            from_cache = True
            if entry.get("bad"):
                counts["skipped"] += 1
                unreadable.append(fp)
        else:
            bad = False
            try:
                embs = [np.asarray(f[0], dtype=np.float32) for f in read_faces(app, fp)]
            except Exception:
                embs = []  # corrupt/unreadable photo -- skip it
                bad = True
                counts["skipped"] += 1
                unreadable.append(fp)
            from_cache = False
            entry = None
            if use_cache and sig is not None:
                entry = {"mtime": sig[0], "size": sig[1], "embs": embs}
                if bad:
                    entry["bad"] = True
                cache[key] = entry
                cache_dirty = True

        src_key = None  # (size, sha256), computed only once something matches
        if embs:
            emb_matrix = np.stack(embs)
            for name, matrix in ref_matrices.items():
                # Best cosine similarity of any face in the photo to any
                # reference face (embeddings are normalized).
                best = float(np.max(matrix @ emb_matrix.T))
                if best < threshold:
                    continue
                if src_key is None:
                    sha = entry.get("sha") if entry else None
                    try:
                        if sha is None:
                            sha = _file_hash(fp)
                            if entry is not None:
                                entry["sha"] = sha
                                cache_dirty = True
                        src_key = (os.path.getsize(fp), sha)
                    except OSError:
                        src_key = (-1, None)
                if src_key[1] is not None and src_key[1] in blocklists[name]:
                    counts["blocklisted"] += 1
                    continue
                existing = contents[name].get(src_key) if src_key[1] else None
                if existing:
                    # Already in this person's folder: keep it tracked and
                    # show it in results, but don't copy it again.
                    counts["duplicates"] += 1
                    dest = existing
                else:
                    dest = _unique_dest(person_dirs[name], os.path.basename(fp))
                    try:
                        shutil.copy2(fp, dest)
                    except OSError:
                        # e.g. disk full: don't leave a half-written copy.
                        try:
                            if os.path.isfile(dest):
                                os.remove(dest)
                        except OSError:
                            pass
                        counts["skipped"] += 1
                        continue
                    if src_key[1] is not None:
                        contents[name][src_key] = dest
                        index.record(dest, src_key[1])
                dest = os.path.abspath(dest)
                if dest in match_scores:
                    # Identical source photo already listed for this person
                    # this run -- don't show the same file twice.
                    match_scores[dest] = max(match_scores[dest], best)
                    continue
                # Track only files the app placed; a matching file the user
                # put there themselves must never become removable.
                if not existing or _norm(dest) in old_manifest:
                    new_manifest[_norm(dest)] = dest
                match_scores[dest] = best
                matches[name].append((dest, best))
                total_matches += 1

        if progress_cb is not None:
            progress_cb(i, total, total_matches, from_cache)

        # Crash safety (e.g. the window is closed mid-scan): checkpoint by
        # time, not photo count, so big caches aren't rewritten constantly.
        now = time.monotonic()
        if now - last_state_save >= STATE_CHECKPOINT_SECS:
            _save_manifest(out_dir, _manifest_with_old())
            index.save()
            last_state_save = now
        if use_cache and cache_dirty and now - last_cache_save >= CACHE_CHECKPOINT_SECS:
            save_cache(photos_folder, cache, detail=detail)
            cache_dirty = False
            last_cache_save = now

    if use_cache:
        # Forget photos that no longer exist in the searched folder.
        present = {os.path.abspath(p) for p in photo_files}
        stale_keys = [k for k in cache if k not in present]
        for k in stale_keys:
            del cache[k]
        if cache_dirty or stale_keys:
            save_cache(photos_folder, cache, detail=detail)

    # Remove previous-run matches that no longer match (e.g. threshold was
    # raised). Only touches files the app itself placed in this folder.
    stale_removed = []
    for k in sorted(set(old_manifest) - set(new_manifest)):
        stale = old_manifest[k]
        try:
            if os.path.isfile(stale):
                os.remove(stale)
                index.forget(stale)
                stale_removed.append(stale)
        except OSError:
            pass
    _save_manifest(out_dir, new_manifest)
    index.save(prune=True)
    _write_unreadable(out_dir, unreadable)
    return _result(False, stale_removed)


__all__ = ["load_model", "enroll_references", "run_search",
           "load_cache", "save_cache", "list_images", "CACHE_DIR",
           "blocklist_add", "forget_match", "_file_hash", "UNREADABLE_NAME",
           "model_ready", "download_progress", "friendly_model_error",
           "MODEL_DOWNLOAD_MB", "gpu_available", "add_reference",
           "remove_reference", "DET_SIZES"]
