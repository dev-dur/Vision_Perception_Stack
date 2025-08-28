# Vision-Perception-Stack
This repository implements a modular perception stack featuring QR code scanning, YOLOv8-OBB object detection, monocular depth estimation using MiDaS, numeric ratio-based height estimation, and angle detection. Designed for general-purpose robotics and automation tasks requiring spatial awareness and visual processing.

🚀 Features
- YOLOv8-OBB Object Detection – Uses oriented bounding boxes (OBB) for precise localization.
- Monocular Depth Estimation (MiDaS) – Generates per-pixel depth maps from a single RGB image.
- Numeric Ratio-based Height Estimation – Estimates object height using pixel-to-real-world scaling.
- Angle Detection – Computes object orientation relative to the camera or world reference frame.

🛠️ Tech Stack
- Python 3.9+
- PyTorch – Model inference (YOLOv8-OBB & MiDaS)
- Ultralytics YOLOv8 (with OBB extension)
- OpenCV – QR detection, image preprocessing, and visualization
- NumPy / SciPy – Math utilities for ratio & angle calculations

📊 Workflow
- YOLOv8-OBB → Detect objects with oriented bounding boxes.
- MiDaS Depth → Generate per-pixel depth estimation.
- Height Estimation → Apply ratio-based scaling to infer real height.
- Angle Detection → Compute tilt/rotation angles.

