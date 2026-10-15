"""Backend tests: matching, output-folder safety, caches, reference photos, model loading.

Run all tests with:  .venv\\Scripts\\python tests\\run_tests.py
Uses fake face data (no model download, no real photos needed) and a
temporary folder; your real settings and saved scan data are never touched.
"""
import os, sys, tempfile
tmp = tempfile.mkdtemp(prefix="face_search_test_")
# Point every "home folder" lookup at the temp folder BEFORE importing the app
# (Windows uses USERPROFILE, other systems HOME).
os.environ["HOME"] = os.environ["USERPROFILE"] = tmp
import threading, json, shutil, types, numpy as np
import search_engine as se, fake
se.CACHE_DIR = os.path.join(tmp, "cache"); se.read_faces = fake.fake_read; vec = fake.vec
se.face_model.MODEL_DIR = os.path.join(tmp, ".face_search", "models")
def w(p, t): os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(t)
def names(r, p=""): return sorted(os.path.basename(x) for x, _ in r["matches"][p])
def jpgs(d): return sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
ref, photos, out = tmp+"/ref", tmp+"/photos", tmp+"/out"
w(ref+"/a1.jpg", "A"); w(ref+"/a2.jpg", "A")
w(photos+"/p1.jpg", "A"); w(photos+"/p2.jpg", "X B"); w(photos+"/p3.jpg", "X A:0.5")
w(photos+"/sub/p4.jpg", "A:0.6"); w(photos+"/dup_of_p1.jpg", "A"); w(photos+"/bad.jpg", "bad")
w(photos+"/nothing.jpg", ""); w(photos+"/._junk.jpg", "bad")
files = se.list_images(photos)
assert not any("._junk" in f for f in files), "macOS junk should be skipped"
people = se.enroll_references(None, ref)
def run(thr=0.45, o=out, **k): return se.run_search(None, people, files, o, photos, threshold=thr, **k)

# 1. basics + unreadable reporting
r = run()
assert names(r) == ["dup_of_p1.jpg", "p3.jpg", "p4.jpg"] and r["duplicates"] == 1
assert [os.path.basename(p) for p in r["unreadable"]] == ["bad.jpg"]
txt = open(out+"/unreadable_photos.txt").read(); assert "bad.jpg" in txt
# rerun from cache still reports the unreadable photo
fake.CALLS.clear()
r = run(); assert r["cache_hits"] == len(files) - 1 and len(r["unreadable"]) == 1 and r["skipped"] == 1
assert [os.path.basename(c) for c in fake.CALLS] == ["bad.jpg"], "unreadable photos are retried, others cached"
print("1 ok: unreadable reported, and retried on the next search")

# 2. output index: second run must not re-hash output files
calls = []; orig = se._file_hash
se._file_hash = lambda p: (calls.append(p), orig(p))[1]
r = run(); se._file_hash = orig
assert calls == [], calls
print("2 ok: no output re-hashing on repeat run")

# 3. manifest is relative; copied output folder can't delete originals
man = json.load(open(out+"/.face_search_manifest.json"))
assert all(not os.path.isabs(p) for p in man), man
out_copy = tmp+"/out_copy"; shutil.copytree(out, out_copy)
# simulate a 1.60-style absolute manifest pointing at the ORIGINAL folder
json.dump([os.path.join(out, p) for p in man], open(out_copy+"/.face_search_manifest.json", "w"))
before = jpgs(out)
r = run(thr=0.99, o=out_copy)
assert jpgs(out) == before, "original folder must be untouched"
print("3 ok: original kept", before, "| copy now", jpgs(out_copy))

# 4. user files never deleted; stale cleanup still works in place
w(out+"/my_own.jpg", "A")
r = run(thr=0.55)
assert "my_own.jpg" in jpgs(out) and "p3.jpg" not in jpgs(out) and "dup_of_p1.jpg" not in jpgs(out)
r = run(thr=0.99); assert "my_own.jpg" in jpgs(out)
print("4 ok:", jpgs(out))

# 5. cache pruning
os.remove(photos+"/p2.jpg"); files = se.list_images(photos); run()
cache = se.load_cache(photos); assert not any(k.endswith("p2.jpg") for k in cache)
print("5 ok: deleted photo pruned from cache")

# 6. crash mid-run keeps manifest of copies made so far
se.STATE_CHECKPOINT_SECS = 0
out3 = tmp+"/out3"
def boom(done, total, m, fc):
    if done == 4: raise KeyboardInterrupt
try: run(o=out3, progress_cb=boom)
except KeyboardInterrupt: pass
man = json.load(open(out3+"/.face_search_manifest.json"))
assert sorted(man) == jpgs(out3) and man, (man, jpgs(out3))
se.STATE_CHECKPOINT_SECS = 10
print("6 ok: after crash, manifest tracks", man)

# 7. reference group photos
ref2 = tmp+"/ref2"; w(ref2+"/a.jpg", "A"); w(ref2+"/group.jpg", "X@900 A:0.9@100")
p = se.enroll_references(None, ref2)
assert p["_ref_info"]["faces_ignored"] == 1 and len(p[""]) == 2
assert float(np.max(p[""] @ vec(2))) < 0.3, "stranger X must not be enrolled"
ref3 = tmp+"/ref3"; w(ref3+"/g1.jpg", "X@900 A@100"); w(ref3+"/g2.jpg", "B@900 A@100")
p = se.enroll_references(None, ref3)
assert len(p[""]) == 2 and all(float(v @ vec(0)) > 0.99 for v in p[""]), "should pick recurring A"
ref4 = tmp+"/ref4"; w(ref4+"/g.jpg", "X@100 A@900")
p = se.enroll_references(None, ref4); assert float(p[""][0] @ vec(0)) > 0.99, "largest face"
# subfolder (manual multi) mode also filters strangers
ref5 = tmp+"/ref5"; w(ref5+"/alice/a.jpg", "A"); w(ref5+"/alice/g.jpg", "B A")
p = se.enroll_references(None, ref5); assert len(p["alice"]) == 2 and float(np.max(p["alice"] @ vec(1))) < 0.3
# auto-separate still uses every face
p = se.enroll_references(None, ref2, auto_cluster=True)
assert "person1_matches" in p and "person2_matches" in p
print("7 ok: group reference photos handled")

# 8. multi-person blocklist still per person
people2 = {"alice": np.stack([vec(0)]), "bob": np.stack([vec(1)])}
w(photos+"/both.jpg", "A B"); files = se.list_images(photos); out2 = tmp+"/out2"
r = se.run_search(None, people2, files, out2, photos)
path = [q for q, _ in r["matches"]["alice"] if q.endswith("both.jpg")][0]
se.blocklist_add(os.path.dirname(path), se._file_hash(path)); os.remove(path); se.forget_match(out2, path)
r = se.run_search(None, people2, files, out2, photos)
assert "both.jpg" not in names(r, "alice") and "both.jpg" in names(r, "bob")
print("8 ok: per-person rejection")

# 9. scan detail keeps separate caches
fake.CALLS.clear(); se.run_search(None, people2, files, out2, photos, detail="thorough")
assert len(fake.CALLS) == len(files), "thorough must not reuse standard cache"
fake.CALLS.clear(); se.run_search(None, people2, files, out2, photos, detail="thorough")
assert [os.path.basename(c) for c in fake.CALLS] == ["bad.jpg"]
fake.CALLS.clear(); se.run_search(None, people2, files, out2, photos)
assert [os.path.basename(c) for c in fake.CALLS] == ["bad.jpg"], "standard cache must be untouched"
print("9 ok: scan detail caches separate")

# 10. add / remove reference photos
refA = tmp+"/refA"; w(refA+"/a.jpg", "A"); w(tmp+"/m/match.jpg", "A B")
d = se.add_reference(refA, "", tmp+"/m/match.jpg")
assert os.path.basename(d) == "added_match.jpg" and os.path.isfile(d)
assert se.add_reference(refA, "", tmp+"/m/match.jpg") is None, "same photo twice"
se.remove_reference(d); assert not os.path.exists(d)
refS = tmp+"/refS"; w(refS+"/Mom/a.jpg", "A")
d = se.add_reference(refS, "Mom", tmp+"/m/match.jpg", subfolders=True)
assert os.path.dirname(d) == os.path.join(refS, "Mom")
print("10 ok: add/remove reference")

# 11. model loading: graphics card with fallback
import face_model as fm
class FakeModel:
    broken_gpu = False
    made = []
    def __init__(self, det_size, providers):
        FakeModel.made.append((det_size, tuple(providers)))
        self.det_size = det_size; self._providers = list(providers)
    @property
    def providers(self): return self._providers
    def warm_up(self):
        if FakeModel.broken_gpu and self._providers[0] != "CPUExecutionProvider":
            raise RuntimeError("DirectML device removed")
real_model, real_ensure = fm.FaceModel, fm.ensure_models
fm.FaceModel = FakeModel; fm.ensure_models = lambda log=None: None
se._available_providers = lambda: ["DmlExecutionProvider", "CPUExecutionProvider"]
app = se.load_model(detail="thorough", use_gpu=True)
assert app.fs_device == "graphics card" and app.det_size == 1280 and app.fs_detail == "thorough"
assert FakeModel.made[-1] == (1280, ("DmlExecutionProvider", "CPUExecutionProvider"))
FakeModel.broken_gpu = True
app = se.load_model(use_gpu=True)
assert app.fs_device == "processor" and app.det_size == 640, "must fall back to processor"
app = se.load_model(fast=True)
assert app.fs_device == "processor" and app.det_size == 320 and FakeModel.made[-1][1] == ("CPUExecutionProvider",)
se._available_providers = lambda: ["CPUExecutionProvider"]
FakeModel.broken_gpu = False
assert se.load_model(use_gpu=True).fs_device == "processor"
fm.FaceModel, fm.ensure_models = real_model, real_ensure
print("11 ok: graphics card used when it works, processor fallback otherwise")

# 12. face pipeline pieces (no real model needed)
#  a) alignment: exact least-squares similarity transform
rng = np.random.default_rng(1)
dst = fm._ARCFACE_DST
for _ in range(200):
    ang, sc = rng.uniform(-1, 1), rng.uniform(0.2, 8)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = (dst - 56) @ R.T * sc + rng.uniform(-500, 3000, 2)
    m = fm.similarity_transform(src, dst)
    assert np.abs(src @ m[:, :2].T + m[:, 2] - dst).max() < 1e-6
#  b) detection: coordinates mapped back to the full-size photo, tiny faces skipped
class FakeDet:
    def setInputSize(self, size): self.size = size
    def detect(self, img):
        w, h = self.size
        big = [10, 20, 100, 120] + [30, 50, 80, 50, 55, 80, 35, 100, 75, 100] + [0.95]
        tiny = [200, 200, 3, 3] + [201] * 10 + [0.9]   # 12 px in the photo: too small
        return 1, np.array([big, tiny], dtype=np.float32)
m = object.__new__(fm.FaceModel); m.det_size = 640; m.detector = FakeDet()
img = np.zeros((2560, 1920, 3), np.uint8)          # scale 0.25
faces = m.detect(img)
assert m.detector.size == (480, 640), m.detector.size
assert len(faces) == 1, "tiny face must be skipped"
f = faces[0]
assert np.allclose(f.bbox, [40, 80, 440, 560]) and np.allclose(f.kps[0], [120, 200]) and abs(f.det_score - 0.95) < 1e-6
#  c) embeddings: unit length, batched, BGR->RGB and normalization applied
class FakeSession:
    def __init__(self): self.calls = []
    def run(self, _names, feeds):
        blob = next(iter(feeds.values())); self.calls.append(blob.shape)
        assert blob.dtype == np.float32 and blob.shape[1:] == (3, 112, 112)
        # vector = per-channel means of the (normalized) input, scaled
        return [np.stack([[b[0].mean() * 3, b[1].mean() * 4, b[2].mean() * 0 + 1] for b in blob])]
m.session = FakeSession(); m.input_name = "x"; m.input_size = 112; m.batch = 16; m.mean = 127.5; m.std = 127.5
faces = [fm.Face(np.zeros(4), dst.copy(), 1.0) for _ in range(20)]
real_align = fm.align_face
fm.align_face = lambda im, kps, size=112: np.full((size, size, 3), (0, 0, 255), np.uint8)  # pure red (BGR)
m.embed(img, faces)
fm.align_face = real_align
assert m.session.calls == [(16, 3, 112, 112), (4, 3, 112, 112)], m.session.calls
e = faces[0].normed_embedding
assert abs(np.linalg.norm(e) - 1) < 1e-6
# red after BGR->RGB is channel 0 = (255-127.5)/127.5 = 1, green = -1
assert np.allclose(e, np.array([3, -4, 1]) / np.linalg.norm([3, -4, 1]), atol=1e-5), e
#  d) downloads: verified by checksum, a damaged download never left behind
payload = {"detector": b"Y" * 200_000, "recognizer": b"A" * 300_000}
import hashlib
saved = {k: dict(v) for k, v in fm.MODELS.items()}
for k in fm.MODELS:
    fm.MODELS[k]["sha256"] = hashlib.sha256(payload[k]).hexdigest()
def fake_open(url):
    k = "detector" if url == fm.MODELS["detector"]["url"] else "recognizer"
    data = payload[k]
    return iter([data[i:i + 65536] for i in range(0, len(data), 65536)]), len(data)
fm._open_url = fake_open
assert not fm.models_ready()
fm.ensure_models()
assert fm.models_ready() and fm.download_progress["done"] == fm.download_progress["total"] == 500_000
os.remove(fm.model_path("recognizer"))
fm.MODELS["recognizer"]["sha256"] = "0" * 64
try:
    fm.ensure_models(); raise SystemExit("expected checksum error")
except IOError as exc:
    assert "checksum" in str(exc)
assert not os.path.exists(fm.model_path("recognizer")) and not os.path.exists(fm.model_path("recognizer") + ".part")
for k, v in saved.items(): fm.MODELS[k].update(v)
#  e) caches of different models never mix
assert ("_" + fm.MODEL_TAG) in os.path.basename(se._cache_path(photos))
print("12 ok: alignment, detection scaling, embeddings, verified downloads, cache naming")

# 13. "couldn't open" vs "face model failed"; hidden/trash folders skipped
ph = tmp+"/ph13"; w(ph+"/ok.jpg", "A"); w(ph+"/bad.jpg", "bad"); w(ph+"/boom.jpg", "boom")
w(ph+"/.dtrash/files/old.jpg", "A"); w(ph+"/$RECYCLE.BIN/x.jpg", "A")
f13 = se.list_images(ph)
assert sorted(os.path.basename(x) for x in f13) == ["bad.jpg", "boom.jpg", "ok.jpg"], f13
o13 = tmp+"/out13"
r = se.run_search(None, people, f13, o13, ph)
assert [os.path.basename(x) for x in r["unreadable"]] == ["bad.jpg"]
assert [os.path.basename(x) for x, _ in r["failed"]] == ["boom.jpg"] and "model exploded" in r["failed"][0][1]
txt = open(o13+"/unreadable_photos.txt", encoding="utf-8").read()
assert "bad.jpg" in txt and "face model failed" in txt and "model exploded" in txt
assert "model exploded" in open(os.path.join(tmp, "face_search_diag.log"), encoding="utf-8").read()
print("13 ok: open errors and model errors reported separately; trash folders skipped")

# 14. a model that only takes one face at a time: automatic fallback
class OneAtATime:
    def __init__(self): self.calls = []
    def run(self, _n, feeds):
        blob = next(iter(feeds.values())); self.calls.append(blob.shape[0])
        if blob.shape[0] != 1: raise RuntimeError("Got invalid dimensions for input: index 0 Got: %d Expected: 1" % blob.shape[0])
        return [np.ones((1, 4), np.float32)]
m14 = object.__new__(fm.FaceModel); m14.session = OneAtATime(); m14.input_name = "x"
m14.input_size = 112; m14.batch = 16; m14.mean = 127.5; m14.std = 127.5
faces14 = [fm.Face(np.zeros(4), fm._ARCFACE_DST.copy(), 1.0) for _ in range(5)]
real_align = fm.align_face; fm.align_face = lambda im, kps, size=112: np.zeros((size, size, 3), np.uint8)
m14.embed(np.zeros((200, 200, 3), np.uint8), faces14)
fm.align_face = real_align
assert m14.session.calls == [5, 1, 1, 1, 1, 1] and m14.batch == 1, m14.session.calls
assert all(abs(np.linalg.norm(f.normed_embedding) - 1) < 1e-6 for f in faces14)
print("14 ok: falls back to one face at a time when batches are rejected")
print("ALL ENGINE TESTS PASSED")
