import numpy as np
D = 8
def vec(i, cos=1.0):
    v = np.zeros(D, np.float32); v[i] = cos; v[7] = np.sqrt(max(0, 1 - cos*cos)); return v
CALLS = []
BOX = (10.0, 10.0, 60.0, 70.0)
def fake_read(app, fp):
    """Text files: tokens like  A  A:0.6  B@500  (person[:cos][@area]); 'bad' -> unreadable.
    Real images (.png/.tif): one face of person A."""
    CALLS.append(fp)
    if fp.lower().endswith((".tif", ".png")):
        return [(vec(0), 100.0, BOX)]
    t = open(fp).read().strip()
    if t == "bad": raise IOError("could not read image")
    if t == "boom": raise RuntimeError("model exploded")
    out = []
    for tok in t.split():
        tok, _, area = tok.partition("@")
        name, _, cos = tok.partition(":")
        out.append((vec("ABX".index(name), float(cos or 1)), float(area or 100), BOX))
    return out
