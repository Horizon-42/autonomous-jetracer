# JetRacer Autonomous Driving Stack

Autonomous racecar platform running on NVIDIA Jetson Nano. Combines a ResNet-18 steering model with YOLOv11 or MobileNet-V1 SSD object detection to navigate a track while reacting to traffic signs, pedestrians, and speed limit zones. A companion Flask dashboard provides real-time telemetry over ZMQ.

---

## Architecture

```
CSI Camera (640×480 @ 30fps)
        │
        ├──► ResNet-18 (TensorRT)            ──► Kalman Filter ──► Steering
        │
        └──► YOLOv11 / MobileNet SSD (TensorRT) ──► State Machine  ──► Throttle
                                                          │
                                                     ZMQ PUB/SUB
                                                          │
                                                   Flask Dashboard
```

The inference node runs entirely on the Jetson Nano. The dashboard can run on any machine on the same network.

---

## Components

### Inference Node (`main.py`)

Runs on the Jetson Nano. Captures frames from the CSI camera and runs two TensorRT engines in sequence each loop iteration:

- **Driving Model** — ResNet-18 trained on labeled steering data. Takes a 224×224 grayscale crop of the center region and outputs a continuous steering value in [-1, 1].
- **Detection Model** — YOLOv11 or MobileNet-V1 SSD (interchangeable) trained on custom classes (stop signs, children, adults, speed limit zones). Runs on the full 640×480 frame. The active backend is selected via `DETECTION_ENGINE_PATH` in `main.py`.

A Kalman filter (2-state: angle + rate) smooths the raw steering output with dt-aware prediction and the Joseph form covariance update for numerical stability.

A state machine governs throttle based on detections:

| Detection          | Behavior                          | Duration |
|--------------------|-----------------------------------|----------|
| Stop sign          | Full stop (throttle = 0)          | 2.0s     |
| Child              | Reduce speed to 80%               | 2.0s     |
| Adult              | Reduce speed to 90%               | 2.0s     |
| Speed limit zone   | Reduce speed to 80% (latched)     | 10.0s    |

Priorities are enforced: stop overrides child, child overrides adult. A cooldown prevents repeated stop triggers from the same sign.

### Dashboard (`dashboard.py`)

Flask web application that subscribes to the ZMQ telemetry stream. Displays:

- Live camera feed with detection overlays and white-balance correction
- System status, speed mode, active incident, throttle, steering, and FPS
- Real-time signal graph (raw steering, Kalman-filtered steering, throttle)
- Live detection tags with danger-class highlighting

Runs on port 5000 by default. Includes a debug mode with synthetic data for UI development without hardware.

---

## Setup

### Jetson Nano

```bash
# Dependencies (JetPack 4.x assumed)
pip install pyzmq pycuda numpy opencv-python

# JetRacer and JetCam
pip install jetracer jetcam
```

Place TensorRT engine files in `./models/`:
- `resnet18_merged_v3.engine` — driving model
- `yolo11.engine` — detection model

Create `labels.txt` with one class name per line matching the YOLO training configuration.

### Dashboard (any machine)

```bash
pip install flask pyzmq opencv-python numpy
```

Update `JETSON_IP` in `dashboard.py` to match the Jetson's address on your network.

---

## Usage

On the Jetson:
```bash
python main.py
```

On the dashboard machine:
```bash
python dashboard.py
```

Open `http://localhost:5000` in a browser.

---

## Configuration

Key parameters are defined as constants at the top of each script.

### Inference Node

| Parameter            | Default | Description                              |
|----------------------|---------|------------------------------------------|
| `MAX_THROTTLE`       | 0.375   | Base throttle value                      |
| `STEERING_GAIN`      | -0.65   | Steering servo gain                      |
| `STEERING_OFFSET`    | 0.20    | Steering center trim                     |
| `CONF_THRES`         | 0.25    | YOLO confidence threshold                |
| `IOU_THRES`          | 0.45    | YOLO NMS IoU threshold                   |

### Dashboard

| Parameter    | Default         | Description                        |
|--------------|-----------------|------------------------------------|
| `JETSON_IP`  | 192.168.0.103   | Jetson Nano IP address             |
| `DATA_PORT`  | 5555            | ZMQ subscriber port                |
| `DEBUG_MODE` | False           | Use synthetic data without Jetson  |

---

## Telemetry Protocol

The inference node publishes ZMQ multipart messages on a PUB socket (port 5555) with topic `dashboard`. Each message contains three frames:

1. **Topic** — `b"dashboard"`
2. **JSON payload** — `raw_steer`, `smooth_steer`, `throttle`, `fps`, `mode`, `incident`, `detections`
3. **JPEG frame** — annotated camera image with bounding boxes

---

## Training

### Driving Model — ResNet-18 (`train_resnet18.py`)

Supervised regression on manually collected steering and throttle labels. The dataset is a zip archive containing a `catalog.csv` and an `images/` directory, collected via the DonkeyCar data pipeline.

**Preprocessing:** center crop to 480px, resize to 224×224, grayscale, normalized to [-1, 1].

**Augmentation:** random horizontal flip (with steering sign inversion), Gaussian blur, brightness/contrast jitter, and salt-and-pepper noise.

**Training:** uses `timm` pretrained ResNet-18 with `in_chans=1` and `num_classes=2` (steering + throttle). Trained with MSE loss and Adam optimizer. 90/10 train/val split.

| Parameter       | Default |
|-----------------|---------|
| `BATCH_SIZE`    | 64      |
| `EPOCHS`        | 35      |
| `LEARNING_RATE` | 1e-4    |

```bash
python train_resnet18.py
```

Output: `output_models/resnet18_merged_csv_v1.pth`

The `.pth` file is then exported to ONNX and converted to a TensorRT engine on the Jetson for inference.

### Detection Model — MobileNet-V1 SSD (`train_mobilenet_ssd.py`)

Trains a MobileNet-V1 SSD detector using the [pytorch-ssd](https://github.com/dusty-nv/pytorch-ssd) repo from Dusty NV. Accepts a YOLO-format dataset (images + `.txt` label files + `classes.txt`) and automatically converts it to Pascal VOC format before training.

**Pipeline:** unzip → scan class IDs → convert YOLO annotations to VOC XML → 90/10 split → patch pytorch-ssd for current PyTorch compatibility → train.

| Parameter       | Default |
|-----------------|---------|
| `BATCH_SIZE`    | 16      |
| `EPOCHS`        | 30      |
| `LEARNING_RATE` | 0.001   |

```bash
python train_mobilenet_ssd.py
```

Output: `output_models/mb1-ssd-Epoch-*-Loss-*.pth` + `labels.txt`

### Detection Model — YOLOv11 (`train_yolo11.py`)

Finetunes a YOLOv11 model using the Ultralytics training pipeline. Includes automatic class-balanced resampling — underrepresented classes are oversampled (weighted by per-image instance count) until all classes reach parity with the most frequent class.

**Pipeline:** parse `data.yaml` → load train image list → scan label files for per-class instance counts → build balanced train list via weighted resampling → write `data_balanced.yaml` → train with Ultralytics.

**Augmentation:** HSV jitter, rotation (±2°), translation, scale, shear, perspective, horizontal flip, mosaic, mixup, copy-paste, and erasing.

| Parameter         | Default                              |
|-------------------|--------------------------------------|
| `--model`         | `yolo11n.pt`                         |
| `--epochs`        | 50                                   |
| `--imgsz`         | 320                                  |
| `--batch`         | 32                                   |
| `--balance-seed`  | 42                                   |
| `--no-balance`    | disabled (balancing on by default)   |

```bash
python train_yolo11.py --data-yaml finetune_dataset/splits/data.yaml --model yolo11n.pt
```

Output: `runs/finetune/<model>_signs/weights/best.pt`

The resulting `.pt` is exported to ONNX and then converted to a TensorRT engine on the Jetson.

### Swapping Detection Backends

The inference node supports two interchangeable detection backends. Both the MobileNet-V1 SSD and YOLOv11 engines are available in the repo — swap between them by changing `DETECTION_ENGINE_PATH` in `main.py`:


```python
# Option A: YOLOv11 (default)
DETECTION_ENGINE_PATH = "./models/yolo11.engine"

# Option B: MobileNet-V1 SSD
DETECTION_ENGINE_PATH = "./models/mobilenet_ssd.engine"
# *all models can be found in "onnx" folder
```

Both models are trained on the same class set defined in `labels.txt`. The YOLO engine uses the `YoloTRT` class with built-in pre/postprocessing, while the MobileNet SSD uses the generic `TRTInference` class. The rest of the pipeline (state machine, throttle control, telemetry) remains identical regardless of which backend is active.

Trained PyTorch checkpoints are exported to ONNX and then converted to TensorRT engines on the target Jetson hardware.

---

## License

This project is for educational and research purposes.
