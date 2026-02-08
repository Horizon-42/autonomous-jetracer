# Learning Guide: JetRacer Autonomous Stack

This guide is a narrative walkthrough of how the project fits together.
It is intended to be read alongside the code.

---

## 1) Big Picture

The system runs two neural networks on a Jetson Nano:

- **Driving model (ResNet-18)**: predicts steering (and throttle output, though
  throttle is overridden by the state machine).
- **Detection model (YOLOv11 or MobileNet-SSD)**: detects signs/people and
  influences throttle behavior.

A simple **Kalman filter** smooths the steering output, and a **state machine**
uses detections to slow/stop the car. Telemetry + an annotated frame are
published over **ZMQ** to a Flask dashboard.

---

## 2) Runtime Flow (Jetson)

Entry point:
- `jetson-inference/infer_drive_detect_v6_yolo.py`

Per-frame pipeline:
1. **Capture** a 640x480 frame from CSI camera.
2. **Detect objects** with YOLO TensorRT.
3. **Preprocess driving crop** (center strip -> resize -> grayscale -> normalize).
4. **Predict steering** with ResNet TensorRT.
5. **Filter steering** with Kalman (angle + rate).
6. **Apply throttle state machine** using detections and timers.
7. **Publish telemetry** + JPEG frame over ZMQ.

Key code paths:
- Detection engine: `YoloTRT` class
- Driving engine: `TRTInference` class
- State machine: logic in `main_loop`
- Telemetry publisher: ZMQ PUB in `main_loop`

---

## 3) Dashboard Flow (any machine)

Entry point:
- `live_feed_v3.py`

Flow:
1. Subscribe to ZMQ on `DATA_PORT` (topic: `dashboard`).
2. Decode JSON telemetry + JPEG bytes.
3. Store latest frame + telemetry in memory.
4. Serve a Flask UI that polls `/data` and streams `/video_feed`.

---

## 4) Training Flow

### 4.1 Driving Model (ResNet-18)
- Script: `train_resnet18.py`
- Dataset: DonkeyCar-style catalog + images
- Model: `timm` ResNet-18, 1-channel input, 2 outputs
- Output: `.pth` weights

Export to ONNX:
- `onnx/export_resnet.py`

TensorRT engine is created on Jetson (outside this repo).

### 4.2 Detection Model (YOLOv11)
- Script: `train_yolo11.py`
- Input: `data.yaml` + train list
- Optional: class-balanced resampling to oversample underrepresented classes

### 4.3 Detection Model (MobileNet-SSD)
- Script: `train_mobilenet_ssd.py`
- Input: YOLO-format dataset
- Conversion: YOLO -> VOC XML
- Training: `pytorch-ssd`

---

## 5) Dataset Utilities

- `combine_dataset_instances.py`: merges multiple DonkeyCar datasets into a
  single `catalog.csv` + `images/` folder, renaming images to avoid collisions.

---

## 6) Key Concepts (Quick Reference)

- **Letterbox**: resize + pad to preserve aspect ratio (YOLO input).
- **NMS**: remove overlapping boxes (postprocessing).
- **Kalman filter**: smooths noisy steering predictions.
- **ZMQ PUB/SUB**: telemetry transport between Jetson and dashboard.

### 6.1 TensorRT Memory + Bindings (Deep Dive)

TensorRT engines expose a set of **bindings** (inputs + outputs). Each binding
has a fixed shape and dtype at build time. The runtime expects an ordered list
of device pointers that matches the engine's binding order.

Typical flow in this repo:
1. **Allocate once**: page-locked host buffers + device buffers for each binding.
2. **Copy H->D**: copy input tensor into the host buffer, then async to device.
3. **Execute**: call `execute_async_v2` with the binding pointer list.
4. **Copy D->H**: async copy outputs back to host.
5. **Synchronize**: wait on the CUDA stream so results are ready.

Where to see it:
- `jetson-inference/infer_drive_detect_v6_yolo.py` (`TRTInference`, `YoloTRT`)
- `jetson-inference/utils/trt_detector.py` (`TrtYoloDetector`)

### 6.2 Kalman Filter (Deep Dive)

The steering filter models a 2D state:
- **x = [angle, rate]** (rate = angular velocity)

Process model (constant velocity):
- angle_k = angle_{k-1} + rate_{k-1} * dt
- rate_k = rate_{k-1}

Measurement model:
- z_k (raw steering from the network) measures the **angle** only.

Key matrices in code:
- **F**: state transition (depends on dt)
- **H**: measurement projection (extracts angle)
- **Q**: process noise (how much we trust the dynamics)
- **R**: measurement noise (how noisy the model output is)

Where to see it:
- `jetson-inference/infer_drive_detect_v6_yolo.py` (`KalmanSteering`)

### 6.3 Dataset Preprocessing (Deep Dive)

**Driving dataset (ResNet-18):**
- Crop center lane region (480px wide), resize to 224x224.
- Convert to grayscale.
- Normalize to [-1, 1] (same as inference).
- Augment: random flip (invert steering), blur, color jitter, noise.

Where to see it:
- `train_resnet18.py` (`AutonomousDataset`, `train_transform`)

**Detection dataset (MobileNet-SSD):**
- YOLO labels are normalized: `class cx cy w h` (relative to image size).
- Converted to VOC XML with absolute pixel coords.
- VOC expects `ImageSets/Main/*.txt` lists with image IDs (no extension).

Where to see it:
- `train_mobilenet_ssd.py` (YOLO -> VOC conversion)

---

## 7) Where to Look First (Suggested Reading Order)

1. `jetson-inference/infer_drive_detect_v6_yolo.py`
2. `jetson-inference/utils/trt_detector.py`
3. `live_feed_v3.py`
4. `train_resnet18.py`
5. `train_yolo11.py`
6. `train_mobilenet_ssd.py`

---

## 8) Typical Execution (High Level)

Jetson:
- `python jetson-inference/infer_drive_detect_v6_yolo.py`

Dashboard:
- `python live_feed_v3.py`

---

## 9) Glossary

- **Engine**: TensorRT compiled model file (`.engine`).
- **Binding**: TensorRT input/output buffer attached to the engine.
- **CHW**: Channel-Height-Width tensor layout (PyTorch / TensorRT style).
- **HWC**: Height-Width-Channel layout (OpenCV image style).
