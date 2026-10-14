"""face_search_gui.py -- desktop GUI for face_search (Flet 1.x; 0.x still works).

Pick a reference folder (photos of the person), a folder of photos to
search, and an output folder, then press Run. Matches are copied to the
output folder and listed here. Everything runs offline on your own machine.

Run it with:
    pip install -r requirements.txt
    python face_search_gui.py
"""

import os
import subprocess
import sys
import threading

import flet as ft
import numpy as np
import asyncio
import base64
import io
import json
import time

# Windowed (PyInstaller --noconsole) exes have sys.stdout/sys.stderr = None;
# any print() by us or a library (e.g. InsightFace's model download) would
# crash with "'NoneType' object has no attribute 'write'". Send it nowhere.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# InsightFace ships a data file (meanshape_68.pkl) that PyInstaller does not
# bundle: in a frozen exe, get_object() looks in sys._MEIPASS/objects/,
# finds nothing, returns None, and the landmark model later crashes with
# "'NoneType' object has no attribute 'shape'". Embed the file (68x3 floats,
# ~5KB) and fall back to it when the real file is missing. The CLI (not
# frozen) keeps using the file from the installed package.
_MEANSHAPE_68_DATA = np.array([
    [-0.6266950368881226, -0.2926996946334839, -0.3140018582344055],
    [-0.5996649265289307, -0.12250272184610367, -0.2924409508705139],
    [-0.571026086807251, 0.05118739977478981, -0.2549465000629425],
    [-0.5338566899299622, 0.21284838020801544, -0.1866333782672882],
    [-0.4797332286834717, 0.3506459593772888, -0.047699932008981705],
    [-0.3957556486129761, 0.4465116560459137, 0.07328663021326065],
    [-0.29880842566490173, 0.5106605887413025, 0.17977793514728546],
    [-0.18838681280612946, 0.554440438747406, 0.3166600465774536],
    [0.0014708322705700994, 0.584439218044281, 0.38841578364372253],
    [0.19099053740501404, 0.5517070889472961, 0.31433114409446716],
    [0.3269285559654236, 0.48957568407058716, 0.16839952766895294],
    [0.4400261342525482, 0.4023579955101013, 0.03596242889761925],
    [0.5068787336349487, 0.3116249442100525, -0.09476063400506973],
    [0.540894627571106, 0.20452618598937988, -0.20267142355442047],
    [0.574118435382843, 0.04570329561829567, -0.2841764986515045],
    [0.5991416573524475, -0.14585931599140167, -0.296495646238327],
    [0.6275436282157898, -0.30774804949760437, -0.30199483036994934],
    [-0.47466811537742615, -0.4376046359539032, 0.23648710548877716],
    [-0.4166599214076996, -0.4717560112476349, 0.3159925639629364],
    [-0.3475387990474701, -0.4840780198574066, 0.366113543510437],
    [-0.26063987612724304, -0.4763909876346588, 0.3992317318916321],
    [-0.16712301969528198, -0.45777544379234314, 0.4166091978549957],
    [0.1231740415096283, -0.45874011516571045, 0.4250619411468506],
    [0.2063615769147873, -0.4804193377494812, 0.41578391194343567],
    [0.28667303919792175, -0.4901494085788727, 0.3919537663459778],
    [0.3623969256877899, -0.4768601655960083, 0.3527863025665283],
    [0.425568163394928, -0.4500580430030823, 0.2953187823295593],
    [-0.007627937477082014, -0.3230886459350586, 0.46194377541542053],
    [-0.007876846939325333, -0.255738765001297, 0.5104694366455078],
    [-0.007687545381486416, -0.19917990267276764, 0.5525456666946411],
    [-0.007345120422542095, -0.14261691272258759, 0.5986667275428772],
    [-0.144961878657341, 0.033126939088106155, 0.4196909964084625],
    [-0.08434253931045532, 0.03127360716462135, 0.4731740951538086],
    [-0.005499685648828745, 0.03975145146250725, 0.5146695971488953],
    [0.06346292793750763, 0.04613516479730606, 0.4792240560054779],
    [0.13398391008377075, 0.02204025723040104, 0.41907867789268494],
    [-0.38675081729888916, -0.3133975565433502, 0.259631872177124],
    [-0.3166607916355133, -0.35007160902023315, 0.3285270035266876],
    [-0.2341379076242447, -0.35491427779197693, 0.3334933817386627],
    [-0.15516234934329987, -0.31524932384490967, 0.3143281042575836],
    [-0.23092176020145416, -0.28427040576934814, 0.3255828619003296],
    [-0.3175090253353119, -0.28516536951065063, 0.3098825216293335],
    [0.13895708322525024, -0.30982404947280884, 0.3182835876941681],
    [0.21945592761039734, -0.35319215059280396, 0.33802759647369385],
    [0.30174651741981506, -0.349665105342865, 0.333102285861969],
    [0.37665316462516785, -0.31351813673973083, 0.26322856545448303],
    [0.2966947853565216, -0.2871439754962921, 0.3220130503177643],
    [0.21462441980838776, -0.2905276417732239, 0.33124178647994995],
    [-0.20143844187259674, 0.23736143112182617, 0.37953662872314453],
    [-0.13732077181339264, 0.18578563630580902, 0.46525245904922485],
    [-0.07648587226867676, 0.15119342505931854, 0.5035716891288757],
    [-0.002535885199904442, 0.16872699558734894, 0.5164376497268677],
    [0.06442102789878845, 0.15088020265102386, 0.5045241117477417],
    [0.126465305685997, 0.17947596311569214, 0.4685916602611542],
    [0.21824784576892853, 0.23899227380752563, 0.37567368149757385],
    [0.13288259506225586, 0.28392839431762695, 0.4400508999824524],
    [0.06802233308553696, 0.2973543107509613, 0.4774041771888733],
    [-0.0004690653004217893, 0.30004069209098816, 0.4871094226837158],
    [-0.06934267282485962, 0.29696860909461975, 0.4805404841899872],
    [-0.14252015948295593, 0.2743033170700073, 0.43808019161224365],
    [-0.1781347393989563, 0.23059049248695374, 0.3963589668273926],
    [-0.07403063774108887, 0.2147187888622284, 0.4653262794017792],
    [-0.002636224264279008, 0.21414154767990112, 0.4832296073436737],
    [0.05981616675853729, 0.21076396107673645, 0.47224411368370056],
    [0.16690002381801605, 0.23127099871635437, 0.396789014339447],
    [0.059809282422065735, 0.22376428544521332, 0.46641337871551514],
    [-0.0014343614457175136, 0.2257590889930725, 0.4752439558506012],
    [-0.07522077113389969, 0.23065608739852905, 0.4671475291252136],
], dtype=np.float32)

def _patch_insightface_data():
    try:
        import insightface.model_zoo.landmark as _lm
    except Exception:
        return
    _orig = _lm.get_object

    def _get_object(name):
        obj = _orig(name)
        if obj is not None:
            return obj
        base = name[:-4] if name.endswith(".pkl") else name
        if base == "meanshape_68":
            return _MEANSHAPE_68_DATA.copy()
        return None

    _lm.get_object = _get_object


from face_search import list_images
from search_engine import (enroll_references, load_model, run_search,
                           blocklist_add, forget_match, _file_hash,
                           UNREADABLE_NAME, model_ready, download_progress,
                           friendly_model_error, MODEL_DOWNLOAD_MB,
                           add_reference, remove_reference, DET_SIZES)

__version__ = "1.80"


CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".face_search_gui.json")

def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(values):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f)
    except Exception:
        pass


def _bring_folder_to_front(path):
    """Bring the just-opened Explorer window to the front (Windows)."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        from ctypes import wintypes
        folder_name = os.path.basename(os.path.normpath(path))
        if not folder_name:
            return
        target = []

        def _cb(hwnd, _lp):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                cls = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetClassNameW(hwnd, cls, 256)
                if (cls.value == "CabinetWClass"
                        and buf.value.lower() == folder_name.lower()):
                    target.append(hwnd)
                    return False
            return True

        cmpfunc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND,
                                     wintypes.LPARAM)
        # Retry a few times: Explorer needs a moment to open.
        import time as _time
        for _ in range(10):
            ctypes.windll.user32.EnumWindows(cmpfunc(_cb), 0)
            if target:
                break
            _time.sleep(0.3)
        if target:
            hwnd = target[0]
            # SetForegroundWindow alone just flashes the taskbar (Windows
            # focus-steal protection). Force it with TOPMOST, then drop
            # the flag so the window behaves normally afterwards.
            HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
            SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW = 0x0002, 0x0001, 0x0040
            flags = SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
            ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST,
                                              0, 0, 0, 0, flags)
            ctypes.windll.user32.SetWindowPos(hwnd, HWND_NOTOPMOST,
                                              0, 0, 0, 0, flags)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def open_folder(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606 -- local folder the user chose
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass


def send_notification(title, message):
    """Show a desktop toast when the scan finishes. Never fails the app."""
    try:
        if sys.platform.startswith("win"):
            try:
                import plyer.platforms.win.notification  # noqa: F401
            except ImportError:
                pass
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="Face Search",
            timeout=10,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------
# Image helpers (run in worker threads)
# --------------------------------------------------------------------------

# Formats the app's image widget can't display; previews are made with PIL.
_NEEDS_PREVIEW = {".heic", ".heif", ".tif", ".tiff"}
# Flet 1.x accepts image bytes in Image.src; 0.x used src_base64.
_FLET_BYTES_SRC = "src_base64" not in getattr(ft.Image, "__dataclass_fields__", {"src_base64": 0})
# 1x1 transparent PNG shown until a preview is ready.
_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def _open_pil(path, draft_px=None):
    """Open any supported photo with PIL, upright (phone rotation applied)."""
    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    im = Image.open(path)
    if draft_px:
        im.draft("RGB", (draft_px, draft_px))  # fast JPEG downscale while decoding
    return ImageOps.exif_transpose(im).convert("RGB")


def _to_jpeg(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _jpeg_thumbnail(path, px):
    """Small JPEG preview of any photo PIL can open (None on failure)."""
    try:
        im = _open_pil(path, draft_px=px)
        im.thumbnail((px, px))
        return _to_jpeg(im)
    except Exception:
        return None


def _face_crop(path, box, px):
    """Square JPEG close-up of one face (box = x1, y1, x2, y2)."""
    try:
        im = _open_pil(path)
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * 1.5
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left = max(0, int(cx - side / 2))
        top = max(0, int(cy - side / 2))
        right = min(im.width, int(cx + side / 2))
        bottom = min(im.height, int(cy + side / 2))
        im = im.crop((left, top, right, bottom))
        im.thumbnail((px, px))
        return _to_jpeg(im)
    except Exception:
        return None


def _describe_reference(path):
    """("people", [names]) if the folder has one subfolder per person,
    otherwise ("one", number_of_photos)."""
    names = []
    try:
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            if os.path.isdir(full) and list_images(full):
                names.append(entry)
    except OSError:
        return ("one", 0)
    if names:
        return ("people", names)
    return ("one", len(list_images(path)))


# --------------------------------------------------------------------------
# Wording helpers
# --------------------------------------------------------------------------

def strictness_text(value):
    if value < 0.40:
        return "Loose (%.2f): finds more photos, but expect some wrong matches" % value
    if value <= 0.50:
        return "Balanced (%.2f): recommended for most searches" % value
    return "Strict (%.2f): fewer wrong matches, but may miss some photos" % value


def match_label(score, threshold):
    margin = score - threshold
    if margin >= 0.15:
        return "Strong match"
    if margin >= 0.05:
        return "Good match"
    return "Borderline match"


def person_label(name):
    if not name:
        return "The person"
    if name.startswith("person") and name.endswith("_matches"):
        num = name[len("person"):-len("_matches")]
        return "Person %s (folder: %s)" % (num, name)
    return name


def _num(n):
    return "{:,}".format(n)


def _clean_path(text):
    """Accept pasted paths, including Explorer's "Copy as path" quotes."""
    text = (text or "").strip().strip('"').strip("'").strip()
    return os.path.normpath(text) if text else ""


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

def main(page: ft.Page):
    _patch_insightface_data()
    page.title = "Face Search"
    try:
        page.window.width = 980   # Flet 1.x
        page.window.height = 900
    except Exception:
        try:
            page.window_width = 980  # Flet 0.x
            page.window_height = 900
        except Exception:
            pass
    page.scroll = ft.ScrollMode.AUTO
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.padding = 20

    SUBTLE = ft.Colors.ON_SURFACE_VARIANT  # readable in light and dark mode
    ERROR = ft.Colors.ERROR

    # ---- Flet 0.x / 1.x compatibility ------------------------------------
    def show_dialog(dlg):
        if hasattr(page, "show_dialog"):
            page.show_dialog(dlg)       # Flet 1.x
        else:
            page.open(dlg)              # Flet 0.x

    def close_dialog(dlg):
        try:
            if hasattr(page, "pop_dialog"):
                page.pop_dialog()       # Flet 1.x
            else:
                page.close(dlg)         # Flet 0.x
        except Exception:
            pass

    _tf_fields = getattr(ft.TextField, "__dataclass_fields__", {})

    def set_error(field, msg):
        if "error" in _tf_fields:
            field.error = msg            # Flet 1.x
        else:
            field.error_text = msg       # Flet 0.x

    def set_label(button, text):
        if hasattr(button, "content") and not hasattr(button, "text"):
            button.content = text        # Flet 1.x
        else:
            button.text = text           # Flet 0.x

    def ms(n):
        return ft.Duration(milliseconds=n) if hasattr(ft, "Duration") else n

    _fit_enum = getattr(ft, "BoxFit", None) or getattr(ft, "ImageFit", None)

    async def _fill_thumb(img, path, px):
        data = await asyncio.to_thread(_jpeg_thumbnail, path, px)
        if data is None:
            return
        if _FLET_BYTES_SRC:
            img.src = data
        else:  # Flet 0.x
            img.src_base64 = base64.b64encode(data).decode("ascii")
        try:
            page.update()
        except Exception:
            pass

    def make_image(src, size, fit_name="CONTAIN", radius=None):
        """Image from a file path or JPEG bytes. Photos are decoded at
        preview size; HEIC/TIFF get a JPEG preview made in the background."""
        convert = isinstance(src, str) and \
            os.path.splitext(src)[1].lower() in _NEEDS_PREVIEW
        if isinstance(src, bytes) and not _FLET_BYTES_SRC:
            kwargs = {"src_base64": base64.b64encode(src).decode("ascii")}
        else:
            kwargs = {"src": _PLACEHOLDER_PNG if convert and _FLET_BYTES_SRC else src}
        kwargs.update(width=size, height=size)
        if _fit_enum is not None:
            kwargs["fit"] = getattr(_fit_enum, fit_name)
        if radius:
            kwargs["border_radius"] = radius
        extras = ({"cache_width": size * 2}, {}) if isinstance(src, str) else ({},)
        for extra in extras:
            try:
                img = ft.Image(**kwargs, **extra)
            except TypeError:
                continue
            if convert:
                page.run_task(_fill_thumb, img, src, size * 2)
            return img
        return ft.Text("(no preview)")

    # ---- Settings ---------------------------------------------------------
    _cfg = load_config()

    def _field(label, hint, key):
        return ft.TextField(label=label, hint_text=hint, expand=True, dense=True,
                            value=_cfg.get(key, ""))

    ref_field = _field("1. Photos of the person to find",
                       "Folder with a few clear photos of their face", "ref")
    photos_field = _field("2. Photos to search through",
                          "Folder to search (subfolders included)", "photos")
    out_field = _field("3. Where to put the matches",
                       "Folder that receives copies of the matching photos", "out")

    REF_HINT = ("Use 5-10 clear photos where the face is easy to see. "
                "Looking for several people? Give each one a subfolder, "
                "e.g. Refs\\Mom and Refs\\Dad: each gets its own folder of matches.")
    PHOTOS_HINT = ("Subfolders are included. Your original photos are never "
                   "changed, moved or deleted.")
    OUT_HINT = ("Matching photos are copied here. Suggested for you after step 2, "
                "change it if you like.")
    ref_hint = ft.Text(REF_HINT, size=12, color=SUBTLE)
    photos_hint = ft.Text(PHOTOS_HINT, size=12, color=SUBTLE)
    out_hint = ft.Text(OUT_HINT, size=12, color=SUBTLE)

    try:
        _thr = min(0.70, max(0.25, float(_cfg.get("threshold", 0.45))))
    except Exception:
        _thr = 0.45
    threshold = ft.Slider(min=0.25, max=0.70, divisions=9, value=_thr,
                          label="{value}")
    strict_desc = ft.Text(strictness_text(_thr), size=13)

    _detail0 = _cfg.get("detail") or ("quick" if _cfg.get("fast") else "standard")
    if _detail0 not in DET_SIZES:
        _detail0 = "standard"
    detail_group = ft.RadioGroup(value=_detail0, content=ft.Column([
        ft.Radio(value="quick",
                 label="Quick: about 3x faster the first time, but misses small or distant faces"),
        ft.Radio(value="standard", label="Standard: recommended"),
        ft.Radio(value="thorough",
                 label="Thorough: also finds small faces in group shots and large "
                       "photos, about 3x slower"),
    ], spacing=0))
    gpu_check = ft.Checkbox(
        label="Use the graphics card to search faster, when possible",
        value=bool(_cfg.get("gpu", True)))
    cache_check = ft.Checkbox(
        label="Remember scanned photos, so searching the same folder again takes seconds",
        value=bool(_cfg.get("cache", True)))
    multi_check = ft.Checkbox(
        label="My reference photos show several different people: sort the matches "
              "into one folder per person",
        value=bool(_cfg.get("multi", False)))
    check_faces = ft.Checkbox(
        label="Show me the faces from my reference photos before searching",
        value=bool(_cfg.get("check_faces", True)))

    def persist(_e=None):
        save_config({"ref": ref_field.value, "photos": photos_field.value,
                     "out": out_field.value,
                     "threshold": round(threshold.value, 2),
                     "multi": multi_check.value, "detail": detail_group.value,
                     "gpu": gpu_check.value, "cache": cache_check.value,
                     "check_faces": check_faces.value})

    def on_threshold(_e):
        strict_desc.value = strictness_text(threshold.value)
        page.update()

    threshold.on_change = on_threshold
    threshold.on_change_end = persist
    for _cb in (detail_group, gpu_check, cache_check, multi_check, check_faces):
        _cb.on_change = persist

    adv_box = ft.Column([
        ft.Text("Scan detail", weight=ft.FontWeight.BOLD, size=13),
        detail_group,
        ft.Container(height=6),
        gpu_check, cache_check, multi_check, check_faces,
    ], spacing=0, visible=False)
    adv_toggle = ft.TextButton("Advanced options ▸")

    def on_adv(_e):
        adv_box.visible = not adv_box.visible
        set_label(adv_toggle, "Advanced options " + ("▾" if adv_box.visible else "▸"))
        page.update()

    adv_toggle.on_click = on_adv

    # ---- Status / results area -------------------------------------------
    status = ft.Text("", size=13, selectable=True)
    progress = ft.ProgressBar(value=0, visible=False)
    results_title = ft.Text("", weight=ft.FontWeight.BOLD, size=15)
    results_note = ft.Text("These are copies: your original photos are untouched. "
                           "Click a photo to open it full size.",
                           size=12, color=SUBTLE, visible=False)
    results_list = ft.Column(spacing=8, visible=False)
    refs_text = ft.Text("", size=13)
    refs_banner = ft.Container(visible=False)  # filled in below, needs on_run

    run_btn = ft.FilledButton("Run search")
    cancel_btn = ft.OutlinedButton("Cancel", disabled=True)
    open_btn = ft.OutlinedButton("Open matches folder", visible=False)
    new_search_btn = ft.OutlinedButton("Change folders or settings", visible=False)
    browse_btns = []

    cancel_event = threading.Event()
    state = {"out_dir": "", "models": {}, "shown": 0, "pending": [],
             "snack": None, "ref_notes": [], "threshold": _thr, "device": "",
             "ref_dir": "", "ref_subfolders": False, "added_refs": 0}

    # ---- Folder pickers and hints ----------------------------------------
    def clear_error(field):
        def _h(_e):
            set_error(field, None)
            page.update()
        return _h

    async def refresh_hints():
        """Describe what's in the chosen folders, so problems show up
        before pressing Run."""
        ref = _clean_path(ref_field.value)
        if ref and os.path.isdir(ref):
            kind, info = await asyncio.to_thread(_describe_reference, ref)
            if kind == "people":
                shown_names = ", ".join(info[:6]) + (" ..." if len(info) > 6 else "")
                ref_hint.value = ("Found %d people (one subfolder each): %s. "
                                  "Each gets its own folder of matches."
                                  % (len(info), shown_names))
                ref_hint.color = SUBTLE
            elif info == 0:
                ref_hint.value = "No photos found in this folder."
                ref_hint.color = ERROR
            else:
                ref_hint.value = "Found %d photo(s).%s" % (
                    info, " Tip: 5-10 clear face photos give the best results."
                    if info < 3 else "")
                ref_hint.color = SUBTLE
        elif ref:
            ref_hint.value, ref_hint.color = "This folder doesn't exist.", ERROR
        else:
            ref_hint.value, ref_hint.color = REF_HINT, SUBTLE
        photos = _clean_path(photos_field.value)
        if photos and os.path.isdir(photos):
            photos_hint.value, photos_hint.color = "Counting photos...", SUBTLE
            page.update()
            n = len(await asyncio.to_thread(list_images, photos))
            if n:
                photos_hint.value = ("Found %s photos to search (subfolders included). "
                                     "Your originals are never changed." % _num(n))
                photos_hint.color = SUBTLE
            else:
                photos_hint.value, photos_hint.color = "No photos found in this folder.", ERROR
        elif photos:
            photos_hint.value, photos_hint.color = "This folder doesn't exist.", ERROR
        else:
            photos_hint.value, photos_hint.color = PHOTOS_HINT, SUBTLE
        page.update()

    def suggest_out():
        photos = _clean_path(photos_field.value)
        if photos and not out_field.value:
            parent = os.path.dirname(photos.rstrip(os.sep))
            name = os.path.basename(photos.rstrip(os.sep))
            out_field.value = os.path.join(parent, name + "_matches")

    async def on_field_done(_e):
        for f in (ref_field, photos_field, out_field):
            f.value = _clean_path(f.value)
        suggest_out()
        persist()
        await refresh_hints()

    for f in (ref_field, photos_field, out_field):
        f.on_change = clear_error(f)
        f.on_blur = on_field_done
        f.on_submit = on_field_done

    async def browse(field):
        import tkinter as tk
        from tkinter import filedialog as tkfd

        def _pick():
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)  # in front of the Flet window
            root.update_idletasks()
            root.lift()
            try:
                root.focus_force()
            except Exception:
                pass
            try:
                cur = _clean_path(field.value)
                start = cur if cur and os.path.isdir(cur) else None
                return tkfd.askdirectory(title="Choose a folder", parent=root,
                                         initialdir=start)
            finally:
                root.destroy()
        path = await asyncio.to_thread(_pick)
        if path:
            field.value = os.path.normpath(path)
            set_error(field, None)
            suggest_out()
            persist()
            page.update()
            await refresh_hints()

    async def browse_ref(_e):
        await browse(ref_field)

    async def browse_photos(_e):
        await browse(photos_field)

    async def browse_out(_e):
        await browse(out_field)

    def set_running(running):
        run_btn.disabled = running
        cancel_btn.disabled = not running
        progress.visible = running
        for c in browse_btns + [ref_field, photos_field, out_field, threshold,
                                detail_group, gpu_check, cache_check, multi_check,
                                check_faces]:
            c.disabled = running
        page.update()

    def on_cancel(_e):
        cancel_event.set()
        status.value = "Cancelling..."
        page.update()

    # ---- "Not a match" with undo -----------------------------------------
    UNDO_SECONDS = 6

    def _finalize(rec):
        """Actually delete the copy and remember the rejection."""
        if rec["done"] or rec["undone"]:
            return
        rec["done"] = True
        path = rec["path"]
        try:
            blocklist_add(rec["person_dir"], _file_hash(path))
        except Exception:
            pass
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass
        forget_match(rec["out_dir"], path)
        if rec in state["pending"]:
            state["pending"].remove(rec)

    def flush_pending():
        for rec in list(state["pending"]):
            _finalize(rec)

    async def _finalize_later(rec):
        await asyncio.sleep(UNDO_SECONDS + 0.5)
        _finalize(rec)

    def _close_snack():
        snack = state.get("snack")
        if snack is not None and getattr(snack, "open", False):
            snack.open = False
            try:
                snack.update()
            except Exception:
                pass

    def _undo(rec):
        if rec["done"] or rec["undone"]:
            return
        rec["undone"] = True
        if rec in state["pending"]:
            state["pending"].remove(rec)
        grid = rec["grid"]
        grid.controls.insert(min(rec["index"], len(grid.controls)), rec["card"])
        state["shown"] += 1
        _update_title()
        _close_snack()
        page.update()

    def _snack(text, action=None, on_action=None, seconds=4):
        _close_snack()
        kwargs = {"content": ft.Text(text), "duration": ms(int(seconds * 1000))}
        if action:
            kwargs.update(action=action, on_action=on_action)
        snack = ft.SnackBar(**kwargs)
        state["snack"] = snack
        show_dialog(snack)
        page.update()

    def _reject(card, grid, path, person_dir, card_state):
        if card not in grid.controls:
            return
        if card_state.get("ref"):  # it was also added as a reference: undo that
            _toggle_ref(card_state, card_state.get("button"), quiet=True)
        rec = {"path": path, "person_dir": person_dir, "out_dir": state["out_dir"],
               "card": card, "grid": grid, "index": grid.controls.index(card),
               "done": False, "undone": False}
        grid.controls.remove(card)
        state["shown"] = max(0, state["shown"] - 1)
        _update_title()
        state["pending"].append(rec)
        _snack("Removed %s from the matches. Your original photo is safe."
               % os.path.basename(path), "Undo", lambda _e: _undo(rec), UNDO_SECONDS)
        page.run_task(_finalize_later, rec)

    # ---- "Add as reference": learn from confirmed matches -----------------
    ADD_LABEL = "Add as reference"
    ADDED_LABEL = "Added \u2713 (undo)"

    def _update_refs_banner():
        n = state["added_refs"]
        refs_banner.visible = n > 0
        refs_text.value = ("%d photo%s added to your reference photos. Search again to "
                           "also find photos that look like %s; photos already scanned "
                           "are re-checked in seconds."
                           % (n, "" if n == 1 else "s", "them" if n != 1 else "it"))

    def _toggle_ref(card_state, button, quiet=False):
        if card_state.get("ref"):
            remove_reference(card_state["ref"])
            card_state["ref"] = None
            state["added_refs"] = max(0, state["added_refs"] - 1)
            if button is not None:
                set_label(button, ADD_LABEL)
        else:
            try:
                dest = add_reference(state["ref_dir"], card_state["person"],
                                     card_state["path"], state["ref_subfolders"])
            except OSError as exc:
                _snack("Couldn't copy the photo into your reference folder: %s" % exc)
                return
            if dest is None:
                _snack("This photo is already one of your reference photos.")
                return
            card_state["ref"] = dest
            state["added_refs"] += 1
            if button is not None:
                set_label(button, ADDED_LABEL)
        _update_refs_banner()
        if not quiet:
            page.update()

    # ---- Results -----------------------------------------------------------
    PAGE_SIZE = 60  # cards rendered at a time; "Show more" adds the next batch

    def _photo_card(path, score, person, person_dir, grid):
        card = ft.Column(spacing=2, width=250,
                         horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        card_state = {"path": path, "person": person, "ref": None}
        img = make_image(path, 240)
        add_btn = ft.TextButton(
            ADD_LABEL,
            tooltip="It's them: copies this photo into your reference photos, so the "
                    "next search also finds photos like it. Most useful for good "
                    "matches with a different age, angle or lighting.")
        add_btn.on_click = lambda _e: _toggle_ref(card_state, add_btn)
        card_state["button"] = add_btn
        card.controls = [
            ft.Container(content=img, on_click=lambda _e: open_folder(path),
                         ink=True, tooltip="Click to open the full photo"),
            ft.Text(os.path.basename(path), size=12, max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    text_align=ft.TextAlign.CENTER, width=240),
            ft.Text("%s (%.2f)" % (match_label(score, state["threshold"]), score),
                    size=12, weight=ft.FontWeight.BOLD,
                    tooltip="How closely this face matches your reference photos. "
                            "Higher is more certain."),
            ft.Row([
                ft.TextButton("Not a match",
                              tooltip="Removes this copy from the matches and won't add "
                                      "it again. Your original photo is not touched.",
                              on_click=lambda _e: _reject(card, grid, path, person_dir,
                                                          card_state)),
                add_btn,
            ], spacing=0, wrap=True, alignment=ft.MainAxisAlignment.CENTER),
        ]
        return card

    def _update_title():
        n = state["shown"]
        results_title.value = ("%s matching photo%s, most certain first"
                               % (_num(n), "" if n == 1 else "s")) if n \
            else "No matches"

    def _add_person_section(name, items, person_dir, multi):
        if multi:
            results_list.controls.append(
                ft.Text("%s: %d photo(s)" % (person_label(name), len(items)),
                        weight=ft.FontWeight.BOLD, size=14))
        grid = ft.Row(wrap=True, spacing=10, run_spacing=14)
        results_list.controls.append(grid)
        more_btn = ft.OutlinedButton("Show more")
        pos = {"i": 0}

        def _add_batch(_e=None):
            batch = items[pos["i"]:pos["i"] + PAGE_SIZE]
            grid.controls.extend(_photo_card(p, s, name, person_dir, grid)
                                 for p, s in batch)
            pos["i"] += len(batch)
            left = len(items) - pos["i"]
            more_btn.visible = left > 0
            set_label(more_btn, "Show %d more" % min(left, PAGE_SIZE))
            if _e is not None:
                page.update()

        more_btn.on_click = _add_batch
        _add_batch()
        results_list.controls.append(more_btn)

    def _summary(result):
        details = []
        if result.get("cache_hits"):
            details.append("%s already scanned before (instant)" % _num(result["cache_hits"]))
        if result.get("unreadable"):
            details.append("%d couldn't be opened (listed in %s)"
                           % (len(result["unreadable"]), UNREADABLE_NAME))
        copy_failed = result.get("skipped", 0) - len(result.get("unreadable", []))
        if copy_failed > 0:
            details.append("%d couldn't be copied (is the disk full?)" % copy_failed)
        if result.get("duplicates"):
            details.append("%d were already in the matches folder" % result["duplicates"])
        if result.get("cleaned"):
            details.append("%d duplicate copies removed" % result["cleaned"])
        if result.get("blocklisted"):
            details.append("%d you marked 'Not a match' were left out" % result["blocklisted"])
        stale = result.get("stale_removed", [])
        if stale:
            details.append("%d earlier match(es) no longer pass the strictness "
                           "setting and were removed" % len(stale))
        details.extend(state.get("ref_notes", []))
        if state.get("device") == "graphics card":
            details.append("searched using the graphics card")
        return details

    def _render_results(result):
        out_dir = state["out_dir"]
        matches = result["matches"]
        total = sum(len(v) for v in matches.values())
        multi = any(name for name in matches)
        details = _summary(result)
        detail_str = ("\n" + "\n".join("• " + d for d in details)) if details else ""

        if result["cancelled"]:
            status.value = ("Search cancelled. The %d match(es) found so far are in "
                            "the matches folder." % total)
        elif total:
            breakdown = ""
            if multi:
                breakdown = ":" + "".join(
                    "\n    %s: %d" % (person_label(n), len(v))
                    for n, v in sorted(matches.items()))
            status.value = ("Done! Found %s matching photo%s%s%s"
                            % (_num(total), "" if total == 1 else "s",
                               breakdown or ".", detail_str))
        else:
            status.value = ("Done, but no matching photos were found.\n"
                            "Try moving the strictness slider toward Loose and searching "
                            "again (photos already scanned are re-checked in seconds), "
                            "or add more reference photos of the person." + detail_str)
        send_notification("Face Search",
                          "Search cancelled." if result["cancelled"] else
                          "Done: %d matching photo(s) found." % total)

        results_list.controls = []
        for name in sorted(matches):
            items = matches[name]
            if items:
                person_dir = os.path.join(out_dir, name) if name else out_dir
                _add_person_section(name, items, person_dir, multi)
        state["shown"] = total
        _update_title()
        results_title.visible = True
        results_list.visible = results_note.visible = bool(total)
        open_btn.visible = True
        progress.value = 1.0
        if total:
            # Collapse the setup so the matches sit near the top.
            setup_section.visible = False
            new_search_btn.visible = True
        page.update()

    # ---- Scanning ----------------------------------------------------------
    progress_state = {"active": False, "done": 0, "total": 0, "matches": 0,
                      "from_cache": False, "result": None, "error": None}

    def progress_cb(done, total, matches, from_cache):
        progress_state.update(done=done, total=total, matches=matches,
                              from_cache=from_cache)

    async def _poll_progress():
        # The worker thread only writes progress_state; all UI updates happen
        # here on the UI loop (updates from a thread don't redraw reliably).
        last_done = -1
        while progress_state["active"]:
            done, total = progress_state["done"], progress_state["total"]
            progress.value = done / max(1, total) if total else None
            if done != last_done:
                eta = ""
                start_t = progress_state.get("start_time", 0)
                if start_t and done > 5:
                    elapsed = time.time() - start_t
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (total - done) / rate if rate > 0 else 0
                    if remaining >= 90:
                        eta = " · about %d min left" % int(remaining / 60 + 0.5)
                    elif remaining >= 5:
                        eta = " · under 2 min left"
                status.value = ("Searching: %s of %s photos checked · %d match(es) "
                                "so far%s" % (_num(done), _num(total),
                                              progress_state["matches"], eta))
                last_done = done
                page.update()
            await asyncio.sleep(0.25)

        set_running(False)
        err = progress_state.pop("error", None)
        result = progress_state.pop("result", None)
        if err is not None:
            status.value = (str(err) if isinstance(err, ValueError)
                            else "Something went wrong during the search: %s" % err)
            page.update()
            return
        if result is not None:
            _render_results(result)

    def _worker(app, ref_matrices, photo_files, out_dir, photos_dir, opts):
        try:
            progress_state["result"] = run_search(
                app, ref_matrices, photo_files, out_dir, photos_dir,
                threshold=opts["threshold"], use_cache=opts["cache"],
                progress_cb=progress_cb, cancel_event=cancel_event,
                detail=opts["detail"])
        except Exception as exc:  # never leave the UI stuck on Run
            progress_state["error"] = exc
        finally:
            progress_state["active"] = False

    def _start_scan(app, ref_matrices, photo_files, out_dir, photos_dir, opts):
        ref_matrices = {k: v for k, v in ref_matrices.items()
                        if not k.startswith("_")}
        state["out_dir"] = out_dir
        state["threshold"] = opts["threshold"]
        status.value = "Starting the search..."
        progress_state.update(active=True, done=0, total=len(photo_files),
                              matches=0, from_cache=False, result=None,
                              error=None, start_time=time.time())
        page.update()
        threading.Thread(target=_worker,
                         args=(app, ref_matrices, photo_files, out_dir,
                               photos_dir, opts),
                         daemon=True).start()
        page.run_task(_poll_progress)

    async def _get_model(detail, use_gpu):
        """Load the face model; on first use, download it with progress."""
        key = (detail, use_gpu)
        app = state["models"].get(key)
        if app is not None:
            return app
        first_time = not model_ready()
        if first_time:
            status.value = ("First-time setup: downloading the face model "
                            "(about %d MB). This happens only once." % MODEL_DOWNLOAD_MB)
        else:
            status.value = "Loading the face model..."
        download_progress.update(done=0, total=0)
        page.update()
        task = asyncio.ensure_future(asyncio.to_thread(
            load_model, detail=detail, use_gpu=use_gpu))
        while not task.done():
            if first_time:
                done, total = download_progress["done"], download_progress["total"]
                if total:
                    progress.value = min(1.0, done / total)
                    if done >= total:
                        status.value = "First-time setup: unpacking the face model..."
                    else:
                        status.value = ("First-time setup: downloading the face model, "
                                        "%d of %d MB. This happens only once."
                                        % (done // 1048576, total // 1048576))
                    page.update()
            await asyncio.sleep(0.3)
        app = task.result()  # raises on failure
        state["models"] = {key: app}  # keep only one model in memory
        progress.value = None
        return app

    def _validate():
        """Return (ref, photos, out) or None after pointing at the problem."""
        for f in (ref_field, photos_field, out_field):
            f.value = _clean_path(f.value)
        checks = [
            (ref_field, "Choose the folder with photos of the person (step 1)."),
            (photos_field, "Choose the folder of photos to search through (step 2)."),
            (out_field, "Choose where to put the matches (step 3)."),
        ]
        for field, msg in checks:
            if not field.value:
                set_error(field, "Required")
                status.value = msg
                page.update()
                return None
        for field, step in ((ref_field, 1), (photos_field, 2)):
            if not os.path.isdir(field.value):
                set_error(field, "Folder not found")
                status.value = "The folder in step %d doesn't exist anymore." % step
                page.update()
                return None
        ref, photos, out = ref_field.value, photos_field.value, out_field.value
        n_out = os.path.normcase(os.path.abspath(out))
        if n_out in (os.path.normcase(os.path.abspath(ref)),
                     os.path.normcase(os.path.abspath(photos))):
            set_error(out_field, "Pick a different folder")
            status.value = ("The matches folder (step 3) must be different from the "
                            "folders in steps 1 and 2.")
            page.update()
            return None
        return ref, photos, out

    async def on_run(_e):
        flush_pending()
        _close_snack()
        folders = _validate()
        if folders is None:
            return
        ref_dir, photos_dir, out_dir = folders
        persist()
        # Settings are read once here, so changing them mid-run has no effect.
        opts = {"threshold": round(threshold.value, 2),
                "cache": bool(cache_check.value),
                "detail": detail_group.value or "standard",
                "gpu": bool(gpu_check.value)}

        results_list.visible = results_note.visible = False
        results_list.controls = []
        refs_banner.visible = False
        results_title.value = ""
        open_btn.visible = False
        progress.value = None  # indeterminate while preparing
        cancel_event.clear()
        set_running(True)

        def _stop(msg=None):
            status.value = msg or "Cancelled."
            progress.value = 0
            set_running(False)

        try:
            app = await _get_model(opts["detail"], opts["gpu"])
            state["device"] = getattr(app, "fs_device", "")
        except Exception as exc:
            _stop(friendly_model_error(exc))
            return
        if cancel_event.is_set():
            return _stop()

        status.value = "Looking through your folders..."
        page.update()
        ref_files, photo_files = await asyncio.to_thread(
            lambda: (list_images(ref_dir), list_images(photos_dir)))
        # Never search the matches folder itself (it may sit inside the
        # photos folder), or matches would match themselves on re-runs.
        out_prefix = os.path.normcase(os.path.abspath(out_dir)) + os.sep
        photo_files = [p for p in photo_files
                       if not os.path.normcase(os.path.abspath(p)).startswith(out_prefix)]
        if not ref_files:
            set_error(ref_field, "No photos here")
            return _stop("No photos found in the folder in step 1.")
        if not photo_files:
            set_error(photos_field, "No photos here")
            return _stop("No photos found in the folder in step 2.")
        if cancel_event.is_set():
            return _stop()

        status.value = "Finding faces in your %d reference photo(s)..." % len(ref_files)
        page.update()
        try:
            ref_matrices = await asyncio.to_thread(
                enroll_references, app, ref_dir, bool(multi_check.value))
        except Exception as exc:
            return _stop(str(exc) if isinstance(exc, ValueError) else "Error: %s" % exc)
        if cancel_event.is_set():
            return _stop()

        info = ref_matrices.get("_ref_info") or {}
        notes = []
        if info.get("faces_ignored"):
            notes.append("%d other face(s) in group reference photos were ignored"
                         % info["faces_ignored"])
        if info.get("no_face"):
            notes.append("no face found in %d reference photo(s)" % len(info["no_face"]))
        if info.get("unreadable"):
            notes.append("%d reference photo(s) couldn't be opened" % len(info["unreadable"]))
        state["ref_notes"] = notes
        state["ref_dir"] = ref_dir
        # Subfolder mode = named people without auto-separation.
        state["ref_subfolders"] = (not ref_matrices.get("_cluster_previews")
                                   and any(k for k in ref_matrices if not k.startswith("_")))
        state["added_refs"] = 0
        _update_refs_banner()

        people = [k for k in ref_matrices if not k.startswith("_")]
        auto_split = bool(ref_matrices.get("_cluster_previews")) and len(people) > 1
        if check_faces.value or auto_split:
            await _show_face_check(app, ref_matrices, people, info, auto_split,
                                   photo_files, out_dir, photos_dir, opts)
        else:
            _start_scan(app, ref_matrices, photo_files, out_dir, photos_dir, opts)

    async def _show_face_check(app, ref_matrices, people, info, auto_split,
                               photo_files, out_dir, photos_dir, opts):
        """Show the faces that will be searched for, before the long scan."""
        MAX_FACES = 12
        status.value = "Preparing a preview of the faces found..."
        page.update()
        faces = ref_matrices.get("_faces") or {}

        def _crops():
            out = {}
            for name in people:
                items = faces.get(name, [])
                out[name] = ([c for c in (_face_crop(fp, box, 144)
                                          for fp, box in items[:MAX_FACES]) if c],
                             len(items))
            return out
        crops = await asyncio.to_thread(_crops)

        rows = []
        if len(people) == 1:
            n = crops[people[0]][1]
            intro = ("These %d face(s) are who the app will look for. If any of them "
                     "is someone else, press Cancel and remove that photo from the "
                     "reference folder." % n)
        else:
            intro = ("Each row is one person; their matches go into a folder named "
                     "after them. If a row mixes different people, press Cancel and "
                     "sort the reference photos into one subfolder per person.")
        rows.append(ft.Text(intro, size=13))
        for name in people:
            imgs, count = crops[name]
            if len(people) > 1:
                rows.append(ft.Text("%s: %d face(s)" % (person_label(name), count),
                                    weight=ft.FontWeight.BOLD, size=13))
            tiles = [make_image(b, 72, "COVER", 6) for b in imgs]
            if count > len(imgs):
                tiles.append(ft.Text("+%d more" % (count - len(imgs)), size=12,
                                     color=SUBTLE))
            rows.append(ft.Row(tiles, wrap=True, spacing=6, run_spacing=6))
        warn = []
        if info.get("no_face"):
            warn.append("No face found in: " + ", ".join(
                os.path.basename(p) for p in info["no_face"][:5]) +
                (" ..." if len(info["no_face"]) > 5 else "") + " (skipped)")
        if info.get("unreadable"):
            warn.append("Couldn't open: " + ", ".join(
                os.path.basename(p) for p in info["unreadable"][:5]))
        if info.get("faces_ignored"):
            warn.append("Ignored %d other face(s) in group photos."
                        % info["faces_ignored"])
        for w_ in warn:
            rows.append(ft.Text(w_, size=12, color=SUBTLE))
        dont_ask = ft.Checkbox(label="Don't show this check again", value=False)
        if not auto_split:
            rows.append(dont_ask)

        def _go(matrices):
            if dont_ask.value:
                check_faces.value = False
                persist()
            _start_scan(app, matrices, photo_files, out_dir, photos_dir, opts)

        def _proceed(_e):
            close_dialog(dlg)
            _go(ref_matrices)

        def _merge_one(_e):
            close_dialog(dlg)
            _go({"": np.vstack([ref_matrices[k] for k in people])})

        def _cancel(_e):
            close_dialog(dlg)
            status.value = "Search cancelled. Adjust your reference photos and press Run again."
            progress.value = 0
            set_running(False)

        actions = [ft.TextButton("Cancel", on_click=_cancel)]
        if auto_split:
            actions.append(ft.TextButton("It's all one person", on_click=_merge_one))
        actions.append(ft.FilledButton(
            "Yes, start the search" if len(people) == 1 else "Looks right, start the search",
            on_click=_proceed))
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Is this the right person?" if len(people) == 1
                          else "Found %d people" % len(people)),
            content=ft.Column(rows, spacing=10, scroll=ft.ScrollMode.AUTO,
                              tight=True, width=520),
            actions=actions,
            actions_alignment=ft.MainAxisAlignment.END,
        )
        status.value = "Check the faces, then start the search."
        page.update()
        show_dialog(dlg)

    def on_open(_e):
        if state["out_dir"]:
            open_folder(state["out_dir"])
            threading.Thread(target=_bring_folder_to_front,
                             args=(state["out_dir"],), daemon=True).start()

    def on_new_search(_e):
        flush_pending()
        setup_section.visible = True
        new_search_btn.visible = False
        page.update()

    refs_banner.content = ft.Row([
        ft.Container(content=refs_text, expand=True),
        ft.FilledButton("Search again", on_click=on_run),
    ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
    refs_banner.padding = 12
    refs_banner.border_radius = 8
    refs_banner.bgcolor = ft.Colors.SECONDARY_CONTAINER
    run_btn.on_click = on_run
    cancel_btn.on_click = on_cancel
    open_btn.on_click = on_open
    new_search_btn.on_click = on_new_search

    browse_btns[:] = [ft.OutlinedButton("Browse...", on_click=browse_ref),
                      ft.OutlinedButton("Browse...", on_click=browse_photos),
                      ft.OutlinedButton("Browse...", on_click=browse_out)]

    def _step(field, btn, hint):
        return ft.Column([ft.Row([field, btn],
                                 vertical_alignment=ft.CrossAxisAlignment.CENTER),
                          hint], spacing=2)

    setup_section = ft.Column([
        _step(ref_field, browse_btns[0], ref_hint),
        _step(photos_field, browse_btns[1], photos_hint),
        _step(out_field, browse_btns[2], out_hint),
        ft.Divider(),
        ft.Text("How strict should matching be?", weight=ft.FontWeight.BOLD, size=14),
        ft.Row([ft.Text("More matches", size=12, color=SUBTLE),
                ft.Container(content=threshold, expand=True),
                ft.Text("Fewer wrong matches", size=12, color=SUBTLE)],
               vertical_alignment=ft.CrossAxisAlignment.CENTER),
        strict_desc,
        adv_toggle,
        adv_box,
    ], spacing=10)

    page.add(
        ft.Text("Face Search", size=24, weight=ft.FontWeight.BOLD),
        ft.Text("Finds every photo of a person in a folder, using a few photos of "
                "their face. Everything runs on this computer: no photos are uploaded.",
                size=13, color=SUBTLE),
        ft.Divider(),
        setup_section,
        ft.Row([run_btn, cancel_btn, open_btn, new_search_btn], spacing=10,
               wrap=True),
        progress,
        status,
        results_title,
        results_note,
        refs_banner,
        results_list,
    )
    if not model_ready():
        status.value = ("Welcome! Pick your three folders and press Run search. "
                        "The first search downloads the face model (about %d MB, "
                        "once only)." % MODEL_DOWNLOAD_MB)
    else:
        status.value = "Pick your three folders, then press Run search."
    page.update()
    page.run_task(refresh_hints)


if __name__ == "__main__":
    # Flet 1.x renamed ft.app() to ft.run(); support both.
    if hasattr(ft, "run"):
        ft.run(main)
    else:  # Flet 0.x
        ft.app(target=main)
