# Vision-Guided Perception Layer for UGV-Based Meal Delivery

A final year project exploring computer vision and robotics for autonomous hospital meal delivery.

## 🌍 Context

In hospitals, timely and accurate meal delivery is essential but labor-intensive. This project reimagines the process using an Unmanned Ground Vehicle (UGV) equipped with a vision-guided perception system to:

- Identify patient trays via QR codes
- Localize and orient tables using object detection + depth sensing
- Place trays with centimeter-level precision using wireless-controlled actuators

## 🧩 My Role

- Vision Design: Developed and trained custom YOLOv8n-OBB model for table detection
- Depth Estimation: Implemented MiDaS + geometric projection fusion for robust height calculation
- Hardware Interface: Assist in designing ESP32 ESP-NOW wireless network for real-time actuator control
- Evaluation: Conducted accuracy tests (confusion matrix, PR/F1 analysis) to optimize thresholds

## ⚡ Problem

Manual meal delivery is error-prone, repetitive, and time-consuming. Existing robotic solutions often lack the precision and adaptability required in dynamic hospital environments.

### 🔬 Process
#### 1. Research & Planning
- Mapped out workflow from meal tray pickup → patient room delivery → tray placement
- Defined success criteria: accuracy, latency, modularity

#### 2. Vision Pipeline
- QR Recognition: Decode patient IDs and verify against hospital records
- Table Detection: YOLOv8n-OBB model trained on hospital-like datasets
- Pose & Depth Estimation: Combined monocular depth + projection math to achieve robust height readings

#### 3. Wireless Actuation
- Designed modular ESP-NOW setup with Master + multiple Slaves
- Each actuator (slider, lifter, rotator) managed by a dedicated ESP32

#### 4. Testing & Iteration
- Benchmarked object detection with precision/recall
- Validated depth estimation against physical measurements
- Measured ESP-NOW latency in real environments

## 🎯 Outcome
- Accurate tray identification via QR vision
- Reliable table localization with orientation detection
- Wireless modular control achieving near real-time response
- A scalable perception layer ready for integration into hospital UGV prototypes

| Actual Height (cm) | Vanish Point | Calculated Height (cm) | Depth Height (cm) | Final Height (cm) | Error % | Angle from Camera (°) |
| ------------------ | ------------ | ---------------------- | ----------------- | ----------------- | ------- | --------------------- |
| 60                 | -20.48       | 63.5                   | 57.2              | 60.6              | 9.9     | 16.06                 |
| 62                 | -39.79       | 79.0                   | 65.8              | 58.73             | 18.8    | 17.38                 |
| 60                 | -37.17       | 65.2                   | 58.1              | 63.2              | 10.7    | 15.32                 |
| 65                 | -51.70       | 67.7                   | 56.6              | 64.03             | 16.3    | 10.41                 |
| 67                 | -64.47       | 68.0                   | 56.8              | 66.4              | 16.34   | 14.15                 |

### Insights
- Depth-only methods produced underestimation in certain cases.
- Projection-based methods tended to overshoot at larger vanish points.
- Fusion & correction yielded Final Height values closer to ground truth with reduced error.
- Angular offsets in the 10–17° range confirmed the need for tray rotation before placement.


## 📷 Visuals
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4b073c92-14a6-4008-bcf0-7a19278a6125" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b84087b5-f19b-44ad-8a80-4281b29f90db" />


## 🔮 Future Scope
- Handle occlusion and crowded spaces
- Multi-tray delivery logic
- Integration with hospital logistics systems
- Real-world deployment & long-term trials

## 🛠 Tech Stack
- Vision: OpenCV, PyTorch, YOLOv8n-OBB, MiDaS
- Robotics: ROS Melodic, Jetson Nano
- Hardware: ESP32, stepper/servo/DC actuators
- Wireless: ESP-NOW protocol

## 📝 Reflection

This project was a deep dive into the intersection of computer vision, robotics, and system design. I learned how to balance accuracy, real-time performance, and hardware constraints, while also designing with real-world hospital environments in mind.
