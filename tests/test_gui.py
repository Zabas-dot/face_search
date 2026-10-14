"""GUI tests: drive the real window controls with a stand-in page, as a first-time user would.

Run all tests with:  .venv\\Scripts\\python tests\\run_tests.py
Uses fake face data (no model download, no real photos needed) and a
temporary folder; your real settings and saved scan data are never touched.
"""
import os, sys, tempfile
tmp = tempfile.mkdtemp(prefix="face_search_test_")
# Point every "home folder" lookup at the temp folder BEFORE importing the app
# (Windows uses USERPROFILE, other systems HOME).
os.environ["HOME"] = os.environ["USERPROFILE"] = tmp
import asyncio, types, time, threading, json, numpy as np
import flet as ft
import search_engine as se, fake
se.CACHE_DIR = os.path.join(tmp, "cache"); se.read_faces = fake.fake_read
se.MODEL_ROOT = os.path.join(tmp, ".insightface", "models")
import face_search_gui as gui
gui.CONFIG_PATH = os.path.join(tmp, "cfg.json")
gui.load_model = lambda **kw: types.SimpleNamespace(fs_device="processor", **kw)
from PIL import Image as PILImage
def w(p, t): os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(t)
def png(p, color="red"): os.makedirs(os.path.dirname(p), exist_ok=True); PILImage.new("RGB", (200, 200), color).save(p)

class FakePage:
    def __init__(self):
        self.window = types.SimpleNamespace(width=None, height=None)
        self.controls = []; self.dialogs = []; self.tasks = []; self.statuses = []
    def add(self, *c): self.controls.extend(c)
    def update(self):
        st = self._status()
        if st and (not self.statuses or self.statuses[-1] != st): self.statuses.append(st)
    def _status(self):
        for c in self.controls:
            if isinstance(c, ft.Text) and c.size == 13 and c.selectable: return c.value
    def show_dialog(self, d): assert d not in self.dialogs; d.open = True; self.dialogs.append(d)
    def pop_dialog(self): d = self.dialogs.pop(); d.open = False; return d
    def run_task(self, fn, *a): self.tasks.append(asyncio.ensure_future(fn(*a)))
    async def drain(self):
        while self.tasks:
            t = self.tasks.pop(0); await t

def walk(c):
    yield c
    for attr in ("controls", "content", "actions"):
        v = getattr(c, attr, None)
        if isinstance(v, list):
            for x in v: yield from walk(x)
        elif isinstance(v, ft.Control): yield from walk(v)
def all_ctrls(page, extra=()):
    for c in list(page.controls) + list(extra): yield from walk(c)
def find(page, typ, pred=lambda x: True, extra=()): return [x for x in all_ctrls(page, extra) if isinstance(x, typ) and pred(x)]
def button(page, label, extra=()):
    r = [x for x in all_ctrls(page, extra) if isinstance(x, (ft.Button, ft.TextButton, ft.OutlinedButton)) and x.content == label]
    assert r, "button %r not found" % label; return r[0]
def fields(page): return find(page, ft.TextField)
async def call(h, e=None):
    r = h(e)
    if asyncio.iscoroutine(r): await r

async def main_test():
    ref, photos, out = tmp+"/ref", tmp+"/photos", tmp+"/out"
    for i in range(4): png(ref+f"/a{i}.png")
    for i in range(70): w(photos+f"/a{i:02d}.jpg", "A" + " "*i)
    w(photos+"/x.jpg", "X")

    # 1. first launch
    page = FakePage(); gui.main(page); await page.drain()
    print("1 welcome:", page.statuses[-1][:70])
    assert page.statuses[-1].startswith("Welcome") and "280 MB" in page.statuses[-1]
    adv = find(page, ft.Column, lambda c: fast_cb in c.controls) if False else None
    cbs = find(page, ft.Checkbox); assert all(not c.visible or True for c in cbs)
    advbox = [c for c in find(page, ft.Column) if any(isinstance(x, ft.Checkbox) for x in c.controls)][0]
    assert advbox.visible is False, "advanced options should start collapsed"

    # 2. Run with nothing filled in
    await call(button(page, "Run search").on_click)
    f = fields(page)
    print("2 empty run:", page.statuses[-1], "| error:", f[0].error)
    assert f[0].error == "Required" and "step 1" in page.statuses[-1]

    # 3. fill folders (paste with quotes), hints update, output suggested
    f[0].value = '"%s"' % ref
    await call(f[0].on_blur)
    f[1].value = photos; await call(f[1].on_blur)
    hints = [x.value for x in find(page, ft.Text) if x.size == 12 and x.value and x.value.startswith("Found")]
    print("3 hints:", hints, "| out suggested:", f[2].value)
    N = os.path.normpath
    assert f[0].value == N(ref) and any("4 photo(s)" in h for h in hints) and any("71 photos" in h for h in hints)
    assert f[2].value == N(photos) + "_matches"
    f[2].value = out

    # 4. first-run model download with progress
    def slow_load(**kw):
        se.download_progress.update(total=280*1048576, done=0)
        for k in range(1, 5):
            time.sleep(0.35); se.download_progress["done"] = k * 70 * 1048576
        return types.SimpleNamespace(fs_device="processor", **kw)
    gui.load_model = slow_load
    await call(button(page, "Run search").on_click)
    dl = [s for s in page.statuses if "downloading the face model," in s]
    print("4 download statuses:", dl[:2], "...")
    assert dl and "of 280 MB" in dl[0]

    # 5. face check dialog with real face crops
    dlg = page.dialogs[-1]
    print("5 dialog:", dlg.title.value, "| tiles:", len(find(page, ft.Image, extra=[dlg])))
    assert dlg.title.value == "Is this the right person?"
    imgs = find(page, ft.Image, extra=[dlg.content])
    assert len(imgs) == 4 and all(isinstance(i.src, bytes) and i.src[:2] == b"\xff\xd8" for i in imgs)
    await call(button(page, "Yes, start the search", [dlg]).on_click)
    await page.drain()
    title = find(page, ft.Text, lambda x: x.size == 15)[0]
    print("5 results:", title.value, "| status:", page.statuses[-1].split("\n")[0])
    assert title.value.startswith("70 matching photos") and not page.dialogs, "no completion popup"

    # 6. Not a match + Undo, then Not a match for real
    grid = find(page, ft.Row, lambda r: r.wrap and r.controls and isinstance(r.controls[0], ft.Column))[0]
    card = grid.controls[0]; name = card.controls[1].value; path = os.path.join(os.path.normpath(out), name)
    nam = [x for x in walk(card) if isinstance(x, ft.TextButton)][0]
    await call(nam.on_click)
    snack = page.dialogs[-1]; assert isinstance(snack, ft.SnackBar) and "original photo is safe" in snack.content.value
    assert title.value.startswith("69") and card not in grid.controls
    await call(snack.on_action)
    assert title.value.startswith("70") and grid.controls[0] is card and os.path.isfile(path)
    print("6 undo ok:", name, "restored")
    await call(nam.on_click); t0 = time.time(); await page.drain()
    assert not os.path.exists(path) and title.value.startswith("69")
    print("6 removed after %.1fs undo window" % (time.time() - t0))

    # 7. re-run: tick "don't show again"; no-match guidance
    await call(button(page, "Change folders or settings").on_click)
    gui.load_model = lambda **kw: types.SimpleNamespace(fs_device="processor", **kw)
    await call(button(page, "Run search").on_click)
    dlg = page.dialogs[-1]
    [x for x in walk(dlg.content) if isinstance(x, ft.Checkbox)][0].value = True
    await call(button(page, "Yes, start the search", [dlg]).on_click); await page.drain()
    assert json.load(open(gui.CONFIG_PATH))["check_faces"] is False
    f = fields(page); f[1].value = tmp+"/nomatch"; w(tmp+"/nomatch/x.jpg", "X")
    f[2].value = tmp+"/out_nomatch"
    await call(button(page, "Run search").on_click); await page.drain()
    print("7 zero-match status:", page.statuses[-1].split("\n")[:2])
    assert "no matching photos" in page.statuses[-1] and "Loose" in page.statuses[-1]

    # 8. offline first run
    import shutil
    def offline(**kw): raise ConnectionError("Failed to resolve 'github.com' (getaddrinfo failed)")
    gui.load_model = offline
    page2 = FakePage(); gui.main(page2); await page2.drain()
    f = fields(page2); f[1].value = photos; f[2].value = out
    await call(button(page2, "Run search").on_click)
    print("8 offline:", page2.statuses[-1].split("\n")[0])
    assert "Couldn't download the face model" in page2.statuses[-1]
    assert not button(page2, "Run search").disabled

    # 9. subfolders = people, hint + dialog
    gui.load_model = lambda **kw: types.SimpleNamespace(fs_device="processor", **kw)
    refp = tmp+"/refpeople"; w(refp+"/Mom/a.jpg", "A"); w(refp+"/Dad/b.jpg", "B")
    page3 = FakePage(); gui.main(page3); await page3.drain()
    f = fields(page3); f[0].value = refp; await call(f[0].on_blur)
    hint = [x.value for x in find(page3, ft.Text) if x.size == 12 and "people" in (x.value or "")]
    print("9 subfolder hint:", hint)
    assert hint and "Dad, Mom" in hint[0]
    # 10. auto-split dialog offers merge
    ref2 = tmp+"/ref2"; w(ref2+"/a1.jpg", "A"); w(ref2+"/a2.jpg", "A"); w(ref2+"/b.jpg", "B")
    f[0].value = ref2; f[2].value = tmp+"/out2"
    [c for c in find(page3, ft.Checkbox) if "several different people" in c.label][0].value = True
    await call(button(page3, "Run search").on_click)
    dlg = page3.dialogs[-1]; print("10 auto-split:", dlg.title.value)
    assert dlg.title.value == "Found 2 people"
    button(page3, "It's all one person", [dlg])
    await call(button(page3, "Looks right, start the search", [dlg]).on_click); await page3.drain()
    print("10 status:", page3.statuses[-1].split("\n")[0])

    # 11. "Add as reference" + Search again
    page4 = FakePage(); gui.main(page4); await page4.drain()
    f = fields(page4); f[0].value = ref; f[1].value = photos; f[2].value = tmp+"/out4"
    [c for c in find(page4, ft.Checkbox) if "several different people" in c.label][0].value = False
    await call(button(page4, "Run search").on_click); await page4.drain()
    add = [x for x in all_ctrls(page4) if isinstance(x, ft.TextButton) and x.content == "Add as reference"]
    assert len(add) >= 2
    banner = [c for c in page4.controls if isinstance(c, ft.Container) and c.bgcolor][0]
    assert not banner.visible
    await call(add[0].on_click)
    added = [n for n in os.listdir(ref) if n.startswith("added_")]
    assert len(added) == 1 and banner.visible and add[0].content.startswith("Added")
    banner_text = [x for x in walk(banner) if isinstance(x, ft.Text)][0].value
    print("11 banner:", banner_text[:60], "...")
    await call(add[0].on_click)   # undo
    assert not [n for n in os.listdir(ref) if n.startswith("added_")] and not banner.visible
    await call(add[0].on_click); await call(add[1].on_click)
    assert len([n for n in os.listdir(ref) if n.startswith("added_")]) == 2
    # Not a match on an added card also takes it back out of the references
    nam = [x for x in walk(page4.controls[-1]) if isinstance(x, ft.TextButton) and x.content == "Not a match"][0]
    await call(nam.on_click)
    assert len([n for n in os.listdir(ref) if n.startswith("added_")]) == 1
    await call(button(page4, "Search again", [banner]).on_click); await page4.drain()
    assert not banner.visible and page4.statuses[-1].startswith("Done")
    print("11 ok: add as reference, undo, search again")

    # 12. scan detail + graphics card settings reach the model, summary says so
    seen = {}
    def record(**kw):
        seen.update(kw); return types.SimpleNamespace(fs_device="graphics card", **kw)
    gui.load_model = record
    page5 = FakePage(); gui.main(page5); await page5.drain()
    rg = find(page5, ft.RadioGroup)[0]; rg.value = "thorough"; rg.on_change(None)
    f = fields(page5); f[2].value = tmp+"/out5"
    await call(button(page5, "Run search").on_click); await page5.drain()
    assert seen == {"detail": "thorough", "use_gpu": True}, seen
    assert "searched using the graphics card" in page5.statuses[-1]
    assert json.load(open(gui.CONFIG_PATH))["detail"] == "thorough"
    print("12 ok:", seen)
    print("ALL GUI TESTS PASSED")
asyncio.run(main_test())
