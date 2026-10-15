# Third-party notices

Face Search's own code is MIT-licensed (see `LICENSE`). It uses the
following third-party models and libraries, which keep their own licenses.

## Face models (downloaded on first use, not redistributed)

- **YuNet face detector** (`face_detection_yunet_2023mar.onnx`): OpenCV Zoo,
  MIT License. https://huggingface.co/opencv/face_detection_yunet
- **AuraFace v1 face recognizer** (`glintr100.onnx`, saved as
  `auraface_v1.onnx`): fal.ai, Apache License 2.0.
  https://huggingface.co/fal/AuraFace-v1
  Only this file is used. The other files in that repository
  (`scrfd_10g_bnkps.onnx`, `1k3d68.onnx`, `2d106det.onnx`, `genderage.onnx`)
  come from InsightFace under a non-commercial license and are not
  downloaded or used.

The face alignment uses the standard ArcFace 5-point template (public
geometry), implemented in `face_model.py`.

## Libraries (bundled in FaceSearchGUI.exe)

| Library | License |
|---|---|
| OpenCV (opencv-python) | Apache 2.0 |
| ONNX Runtime / ONNX Runtime DirectML | MIT |
| ONNX | Apache 2.0 |
| NumPy | BSD 3-Clause |
| Flet | Apache 2.0 |
| Requests | Apache 2.0 |
| Pillow | MIT-CMU (HPND) |
| pi-heif (HEIC decoding) | BSD 3-Clause; its bundled libheif and libde265 are LGPL 3.0 |
| plyer | MIT |
| Python | PSF License |

The LGPL libraries (libheif, libde265) are used as separate DLLs. The full
source of Face Search is published, so anyone can rebuild the exe with
their own versions of them.
