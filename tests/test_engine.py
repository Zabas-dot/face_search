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
se.MODEL_ROOT = os.path.join(tmp, ".insightface", "models")
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
r = run(); assert r["cache_hits"] == len(files) and len(r["unreadable"]) == 1 and r["skipped"] == 1
print("1 ok: unreadable reported, also from cache")

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
assert fake.CALLS == []
fake.CALLS.clear(); se.run_search(None, people2, files, out2, photos)
assert fake.CALLS == [], "standard cache must be untouched"
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

# 11. model loading: graphics card with fallback, only 2 models
class FakeSession:
    def __init__(self, providers): self.providers = providers
    def get_providers(self): return list(self.providers)
class FakeFA:
    broken_gpu = False
    made = []
    def __init__(self, name, providers, allowed_modules=None):
        FakeFA.made.append((tuple(providers), allowed_modules))
        self.det_model = types.SimpleNamespace(session=FakeSession(providers))
        self.models = {}
    def prepare(self, ctx_id, det_size): self.det = det_size
    def get(self, img):
        if FakeFA.broken_gpu and self.det_model.session.providers[0] != "CPUExecutionProvider":
            raise RuntimeError("DirectML device removed")
        return []
se.FaceAnalysis = FakeFA
se._available_providers = lambda: ["DmlExecutionProvider", "CPUExecutionProvider"]
app = se.load_model(detail="thorough", use_gpu=True)
assert app.fs_device == "graphics card" and app.det == (1280, 1280) and app.fs_detail == "thorough"
assert FakeFA.made[-1] == (("DmlExecutionProvider", "CPUExecutionProvider"), ["detection", "recognition"])
FakeFA.broken_gpu = True
app = se.load_model(use_gpu=True)
assert app.fs_device == "processor" and app.det == (640, 640), "must fall back to processor"
app = se.load_model(fast=True)
assert app.fs_device == "processor" and app.det == (320, 320) and FakeFA.made[-1][0] == ("CPUExecutionProvider",)
se._available_providers = lambda: ["CPUExecutionProvider"]
FakeFA.broken_gpu = False
assert se.load_model(use_gpu=True).fs_device == "processor"
print("11 ok: graphics card used when it works, processor fallback otherwise")

# 12. half-downloaded model folder is repaired, a complete one is kept
md = os.path.join(se.MODEL_ROOT, se.MODEL_NAME); os.makedirs(md)
se.repair_model_folder(); assert not os.path.exists(md), "empty model folder removed"
os.makedirs(md)
for f in se._MODEL_FILES: open(os.path.join(md, f), "w").write("x")
se.repair_model_folder(); assert os.path.isdir(md) and se.model_ready()
print("12 ok: model folder repair")
print("ALL ENGINE TESTS PASSED")
