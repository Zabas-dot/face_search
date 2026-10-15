"""face_model.py -- face detection + recognition with commercially usable models.

Detection:   YuNet (OpenCV Zoo, MIT License), run with OpenCV's FaceDetectorYN.
Recognition: AuraFace v1 (fal.ai, Apache License 2.0), an ArcFace-style
             ResNet-100, run with onnxruntime (graphics card when available).

The two model files (about 262 MB) are downloaded once into
~/.face_search/models and checked against known SHA-256 hashes.

Usage:
    ensure_models()                 # downloads on first use
    model = FaceModel(det_size=640)
    for face in model.get(bgr_image):
        face.bbox, face.kps, face.det_score, face.normed_embedding

No InsightFace code or models are used (InsightFace's models are licensed
for non-commercial research only).
"""

import hashlib
import json
import os

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.expanduser("~"), ".face_search", "models")

MODELS = {
    "detector": {
        "file": "face_detection_yunet_2023mar.onnx",
        "url": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
               "face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "license": "MIT (OpenCV Zoo)",
    },
    "recognizer": {
        "file": "auraface_v1.onnx",
        "url": "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
        "sha256": "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60",
        "license": "Apache-2.0 (fal.ai AuraFace v1)",
    },
}
DOWNLOAD_MB = 262  # approximate total, shown before the real size is known
MODEL_TAG = "af1"  # embedding space id; caches from other models never mix

# Detector settings. YuNet's own default score threshold (0.9) misses many
# real faces in casual photos; 0.8 catches more and the recognition step
# ignores stray non-faces anyway (they match nobody).
DET_SCORE_THRESHOLD = 0.8
DET_NMS_THRESHOLD = 0.3
MIN_FACE_PX = 16          # faces smaller than this (in the photo) are skipped
EMBED_BATCH = 16

# Filled in while models download: {"done": bytes, "total": bytes or 0}
download_progress = {"done": 0, "total": 0}

# ArcFace 5-point template for 112x112 crops (eyes, nose, mouth corners,
# left-to-right as seen in the image). Public, model-independent geometry.
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


# --------------------------------------------------------------------------
# Model files
# --------------------------------------------------------------------------

def model_path(key):
    return os.path.join(MODEL_DIR, MODELS[key]["file"])


def models_ready():
    """True if both model files are present (and not obviously truncated)."""
    for key in MODELS:
        p = model_path(key)
        if not os.path.isfile(p) or os.path.getsize(p) < 100_000:
            return False
    return True


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_url(url):
    """Return (response iterator of byte chunks, content length or 0)."""
    try:
        import requests  # honours system proxy settings, bundled CA certs
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        return r.iter_content(chunk_size=1 << 16), int(r.headers.get("content-length") or 0)
    except ImportError:
        import urllib.request
        resp = urllib.request.urlopen(url, timeout=30)
        length = int(resp.headers.get("Content-Length") or 0)
        return iter(lambda: resp.read(1 << 16), b""), length


def ensure_models(log=None):
    """Download any missing model file, verifying its SHA-256. Progress is
    reported in download_progress. Raises on network or checksum errors
    (a failed download never leaves a broken file behind)."""
    missing = [k for k in MODELS
               if not os.path.isfile(model_path(k)) or os.path.getsize(model_path(k)) < 100_000]
    if not missing:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_progress.update(done=0, total=0)
    streams = []
    for key in missing:
        chunks, length = _open_url(MODELS[key]["url"])
        streams.append((key, chunks))
        download_progress["total"] += length
    for key, chunks in streams:
        final = model_path(key)
        part = final + ".part"
        if log:
            log("Downloading %s..." % MODELS[key]["file"])
        try:
            with open(part, "wb") as f:
                for chunk in chunks:
                    if chunk:
                        f.write(chunk)
                        download_progress["done"] += len(chunk)
            if _sha256(part) != MODELS[key]["sha256"]:
                raise IOError("The downloaded %s is damaged or not the expected file "
                              "(checksum mismatch)." % MODELS[key]["file"])
            os.replace(part, final)
        finally:
            if os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

def similarity_transform(src, dst):
    """Least-squares similarity transform (rotation, uniform scale,
    translation) mapping src points to dst points (Umeyama, 1991).
    Returns a 2x3 matrix for cv2.warpAffine."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / n
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    u, s, vt = np.linalg.svd(cov)
    r = u @ np.diag(d) @ vt
    var_src = (src_c ** 2).sum() / n
    scale = (s * d).sum() / var_src if var_src > 0 else 1.0
    t = dst_mean - scale * (r @ src_mean)
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = t
    return m


def align_face(img, kps, size=112):
    """Crop the face to the standard size x size ArcFace layout."""
    m = similarity_transform(kps, _ARCFACE_DST * (size / 112.0))
    return cv2.warpAffine(img, m, (size, size), borderValue=0.0)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

class Face:
    __slots__ = ("bbox", "kps", "det_score", "normed_embedding")

    def __init__(self, bbox, kps, det_score):
        self.bbox = bbox                # np.array([x1, y1, x2, y2])
        self.kps = kps                  # 5x2: eyes, nose, mouth corners
        self.det_score = det_score
        self.normed_embedding = None    # unit-length float32 vector


def _input_normalization(model_file):
    """ArcFace exports differ in whether the network normalizes pixels
    itself. Check the first graph nodes like the reference ArcFace loader
    does; remember the answer next to the model file."""
    note = model_file + ".json"
    try:
        with open(note, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d["mean"]), float(d["std"])
    except Exception:
        pass
    mean, std = 127.5, 127.5
    try:
        import onnx
        graph = onnx.load(model_file, load_external_data=False).graph
        names = [n.name for n in graph.node[:8]]
        has_sub = any(n.startswith(("Sub", "_minus")) for n in names)
        has_mul = any(n.startswith(("Mul", "_mul")) for n in names)
        if has_sub and has_mul:
            mean, std = 0.0, 1.0
    except Exception:
        pass
    try:
        with open(note, "w", encoding="utf-8") as f:
            json.dump({"mean": mean, "std": std}, f)
    except OSError:
        pass
    return mean, std


class FaceModel:
    """YuNet detection + AuraFace recognition.

    det_size: the photo is scaled so its longer side is this many pixels
    before detection (320 quick, 640 standard, 1280 thorough). Landmarks are
    mapped back and faces are cropped from the full-resolution photo.
    providers: onnxruntime providers for recognition, best first.
    """

    def __init__(self, det_size=640, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        self.det_size = int(det_size)
        self.detector = cv2.FaceDetectorYN.create(
            model_path("detector"), "", (self.det_size, self.det_size),
            DET_SCORE_THRESHOLD, DET_NMS_THRESHOLD, 5000)
        rec_file = model_path("recognizer")
        opts = ort.SessionOptions()
        if providers and providers[0] == "DmlExecutionProvider":
            # DirectML doesn't support these optimizations.
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(rec_file, sess_options=opts,
                                            providers=list(providers))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 112
        batch = inp.shape[0]
        self.batch = EMBED_BATCH if not isinstance(batch, int) or batch < 1 else batch
        self.mean, self.std = _input_normalization(rec_file)

    @property
    def providers(self):
        return self.session.get_providers()

    def detect(self, img):
        h, w = img.shape[:2]
        if h < 8 or w < 8:
            return []
        scale = self.det_size / float(max(h, w))
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1
                           else cv2.INTER_LINEAR) if (sw, sh) != (w, h) else img
        self.detector.setInputSize((sw, sh))
        _ok, rows = self.detector.detect(small)
        faces = []
        if rows is None:
            return faces
        for row in rows:
            x, y, bw, bh = (float(v) / scale for v in row[:4])
            if min(bw, bh) < MIN_FACE_PX:
                continue
            kps = (np.asarray(row[4:14], dtype=np.float64) / scale).reshape(5, 2)
            faces.append(Face(np.array([x, y, x + bw, y + bh], dtype=np.float32),
                              kps, float(row[14])))
        return faces

    def embed(self, img, faces):
        """Fill in face.normed_embedding for each face."""
        if not faces:
            return faces
        crops = [align_face(img, f.kps, self.input_size) for f in faces]
        start = 0
        while start < len(crops):
            n = self.batch
            try:
                out = self._run(crops[start:start + n])
            except Exception:
                if n == 1:
                    raise
                # Some model exports / graphics-card back ends only accept
                # one face at a time: switch to that for good and retry.
                self.batch = 1
                continue
            for f, e in zip(faces[start:start + n], out):
                f.normed_embedding = e
            start += n
        return faces

    def _run(self, crops):
        chunk = np.stack(crops)[..., ::-1]  # BGR -> RGB
        blob = ((chunk.astype(np.float32) - self.mean) / self.std).transpose(0, 3, 1, 2)
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        out = out.reshape(out.shape[0], -1).astype(np.float32)
        if out.shape[0] != len(crops):
            raise ValueError("model returned %d embeddings for %d faces"
                             % (out.shape[0], len(crops)))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def get(self, img):
        """All faces in a BGR image, with embeddings."""
        return self.embed(img, self.detect(img))

    def warm_up(self):
        """Run both models once, so a graphics card that can't run them
        fails here instead of mid-search."""
        self.detect(np.zeros((self.det_size, self.det_size, 3), dtype=np.uint8))
        f = Face(np.zeros(4, np.float32), _ARCFACE_DST.copy(), 1.0)
        self.embed(np.zeros((112, 112, 3), dtype=np.uint8), [f])
