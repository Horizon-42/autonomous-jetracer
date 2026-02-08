"""TensorRT YOLO detector with preprocessing, NMS, and visualization helpers.

This module loads a TensorRT engine, runs inference on images, applies
confidence filtering + NMS, and returns a list of detections:
  {"box": [x1, y1, x2, y2], "conf": float, "class_id": int}
"""

import atexit
import time
from pathlib import Path

import cv2
import numpy as np

# --- 1. IMPORTS & SETUP ---
try:
    import tensorrt as trt
    import pycuda.driver as cuda
except Exception as exc:  
    trt = None
    cuda = None
    _TRT_IMPORT_ERROR = exc
else:
    _TRT_IMPORT_ERROR = None

PALETTE = [
    (255, 82, 82), (0, 189, 214), (65, 179, 123), (255, 196, 0),
    (146, 104, 255), (255, 112, 67), (38, 198, 218), (156, 204, 101),
    (171, 71, 188), (66, 165, 245),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# --- 2. HELPER FUNCTIONS ---

def load_class_names(source=None):
    """Load class labels from a list or a newline-delimited text file."""
    if source is None:
        return []
    if isinstance(source, (list, tuple)):
        return list(source)
    path = Path(source)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _color_for_class(class_id):
    return PALETTE[class_id % len(PALETTE)]

def _text_color_for_bg(bgr):
    """Choose black/white text for contrast against a background color."""
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)

def _draw_text_with_bg(image, text, origin, bg_color, font_scale, thickness):
    """Draw label text with a filled rectangle background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    x = max(0, min(x, image.shape[1] - text_w - 1))
    y = max(text_h + baseline + 1, min(y, image.shape[0] - 1))
    top_left = (x, y - text_h - baseline - 1)
    bottom_right = (x + text_w + 2, y + baseline + 1)
    cv2.rectangle(image, top_left, bottom_right, bg_color, -1)
    text_color = _text_color_for_bg(bg_color)
    cv2.putText(image, text, (x + 1, y - 1), font, font_scale, text_color, thickness, cv2.LINE_AA)

def _letterbox(image, new_shape, color=(114, 114, 114)):
    """Resize while preserving aspect ratio and pad to target shape."""
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)

def _xywh_to_xyxy(xywh):
    """Convert center-x/y, width/height boxes to x1/y1/x2/y2."""
    xyxy = xywh.copy()
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy

def _clip_boxes(boxes, shape):
    """Clamp boxes to image boundaries."""
    h, w = shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes

def _scale_boxes(boxes, ratio, pad, shape):
    """Undo letterbox scaling/padding to map boxes to original image size."""
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    return _clip_boxes(boxes, shape)

def _nms(boxes, scores, iou_thres):
    """Basic CPU NMS for a single class."""
    if len(boxes) == 0: return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        remaining = np.where(iou <= iou_thres)[0]
        order = order[remaining + 1]
    return keep

def _normalize_output(outputs):
    """Handle different TensorRT output layouts and return a 2D array."""
    output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    output = np.asarray(output)
    if output.ndim == 3:
        # Some engines output (1, anchors, dims) or (1, dims, anchors).
        # Normalize to (anchors, dims).
        if output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))
        output = output[0]
    elif output.ndim == 2:
        pass
    else:
        # Flatten unknown layouts into (N, D).
        output = output.reshape(-1, output.shape[-1])
    return output

def _decode_predictions(preds, conf_thres, num_classes=None, box_format="auto"):
    """Decode raw model output into boxes/scores/class_ids."""
    if preds.size == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    dim = preds.shape[1]
    if dim in (6, 7):
        # Already includes [x, y, w, h, conf, cls_id] or similar.
        coords = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(int)
        if box_format == "xywh": coords = _xywh_to_xyxy(coords)
    else:
        # Typical YOLO export: [x, y, w, h, cls0, cls1, ...].
        coords = preds[:, :4]
        scores_raw = preds[:, 4:]
        if num_classes is not None:
            scores = scores_raw[:, :num_classes] if dim == 4 + num_classes else scores_raw
        else:
            scores = scores_raw
        class_ids = scores.argmax(axis=1)
        scores = scores.max(axis=1)
        if box_format in ("auto", "xywh"): coords = _xywh_to_xyxy(coords)
    mask = scores >= conf_thres
    return coords[mask], scores[mask], class_ids[mask]

def draw_detections(image_bgr, detections, class_names=None):
    """Render bounding boxes + labels on a BGR image."""
    output = image_bgr.copy()
    h, w = output.shape[:2]
    thickness = max(1, int(round(min(h, w) / 400)))
    font_scale = max(0.5, min(h, w) / 1200)
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        conf, class_id = det["conf"], det["class_id"]
        color = _color_for_class(class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        name = class_names[class_id] if class_names and 0 <= class_id < len(class_names) else f"class_{class_id}"
        label_text = f"{name} {conf:.2f}"
        _draw_text_with_bg(output, label_text, (x1, y1 - 4), color, font_scale, thickness)
    return output

def collect_images(path):
    """Return a list of images from a folder or a single image path."""
    path = Path(path)
    if path.is_dir():
        return [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return []

# --- 3. MAIN DETECTOR CLASS ---

class TrtYoloDetector:
    def __init__(self, engine_path, class_names=None, conf_thres=0.25, iou_thres=0.45, 
                 input_size=None, box_format="auto", device_id=0, trt_logger_severity=None):
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT + pycuda are required.") from _TRT_IMPORT_ERROR

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {self.engine_path}")

        # TensorRT/pycuda context setup must happen before any GPU allocations.
        cuda.init()
        self._cuda_context = cuda.Device(device_id).make_context()
        atexit.register(self._cleanup)

        self.class_names = load_class_names(class_names)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.box_format = box_format

        severity = trt_logger_severity if trt_logger_severity is not None else trt.Logger.WARNING
        self._logger = trt.Logger(severity)
        self._runtime = trt.Runtime(self._logger)
        
        with self.engine_path.open("rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        
        # Find the input binding index (TensorRT engines can have multiple bindings).
        self._input_index = 0
        for i in range(self._engine.num_bindings):
            if self._engine.binding_is_input(i):
                self._input_index = i
                break
        
        self.input_dtype = trt.nptype(self._engine.get_binding_dtype(self._input_index))
        self.input_size = self._resolve_input_size(input_size)
        self._input_shape = (1, 3, self.input_size[0], self.input_size[1])
        self._context.set_binding_shape(self._input_index, self._input_shape)
        
        self._scale = np.array(1.0 / 255.0, dtype=self.input_dtype)
        self._allocate_buffers()
        self.last_inference_time = None

    def _cleanup(self):
        """Release the CUDA context cleanly on exit."""
        if self._cuda_context:
            try: self._cuda_context.pop()
            except: pass
            try: self._cuda_context.detach()
            except: pass
            self._cuda_context = None

    def _resolve_input_size(self, input_size):
        """Use caller-provided size or infer from the engine bindings."""
        if input_size: return (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        shape = self._engine.get_binding_shape(self._input_index)
        if len(shape) == 4 and shape[2] > 0 and shape[3] > 0: return (int(shape[2]), int(shape[3]))
        return (640, 640)

    def _allocate_buffers(self):
        """Allocate page-locked host buffers and device buffers for each binding."""
        self._bindings, self._host_inputs, self._device_inputs = [], [], []
        self._host_outputs, self._device_outputs, self._output_shapes = [], [], []
        
        for idx in range(self._engine.num_bindings):
            dtype = trt.nptype(self._engine.get_binding_dtype(idx))
            shape = self._context.get_binding_shape(idx)
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Track device memory addresses for execute_async_v2.
            # The order of this list MUST match engine bindings.
            self._bindings.append(int(device_mem)) 
            
            if self._engine.binding_is_input(idx):
                self._host_inputs.append(host_mem)
                self._device_inputs.append(device_mem)
            else:
                self._host_outputs.append(host_mem)
                self._device_outputs.append(device_mem)
                self._output_shapes.append(tuple(int(v) for v in shape))

    def preprocess(self, image_bgr):
        """Prepare input tensor: letterbox, normalize, CHW, batch."""
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized, ratio, pad = _letterbox(image, self.input_size)
        blob = resized.astype(self.input_dtype, copy=False)
        blob *= self._scale
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)
        return blob, ratio, pad

    def predict(self, image_bgr, conf_thres=None, iou_thres=None, verbose=True):
        """Run inference and return a list of detections."""
        conf_thres = conf_thres or self.conf_thres
        iou_thres = iou_thres or self.iou_thres

        # 1. Preprocess
        blob, ratio, pad = self.preprocess(image_bgr)
        np.copyto(self._host_inputs[0], blob.ravel())
        cuda.memcpy_htod_async(self._device_inputs[0], self._host_inputs[0], self._stream)

        # 2. Inference
        # Everything runs on one CUDA stream to keep H<->D copies ordered.
        start = time.perf_counter()
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        for h, d in zip(self._host_outputs, self._device_outputs):
            cuda.memcpy_dtoh_async(h, d, self._stream)
        self._stream.synchronize()
        self.last_inference_time = time.perf_counter() - start

        # 3. Postprocess
        outputs = [np.asarray(h).reshape(s) for h, s in zip(self._host_outputs, self._output_shapes)]
        preds = _normalize_output(outputs).astype(np.float32, copy=False)
        
        boxes, scores, class_ids = _decode_predictions(preds, conf_thres, len(self.class_names) if self.class_names else None)
        
        # --- DEBUG INFO ---
        raw_count = len(boxes)
        if raw_count == 0:
            if verbose: print(f"[DEBUG] No objects above confidence {conf_thres}.")
            return []

        boxes = _scale_boxes(boxes, ratio, pad, image_bgr.shape)
        detections = []
        for class_id in np.unique(class_ids):
            idxs = np.where(class_ids == class_id)[0]
            keep = _nms(boxes[idxs], scores[idxs], iou_thres)
            for k in keep:
                i = idxs[k]
                detections.append({"box": boxes[i].tolist(), "conf": float(scores[i]), "class_id": int(class_ids[i])})

        detections.sort(key=lambda d: d["conf"], reverse=True)
        
        if verbose:
            print(f"[DEBUG] Raw Candidates: {raw_count} | Post-NMS Detections: {len(detections)} | Time: {self.last_inference_time:.4f}s")
            
        return detections

    def visualize(self, image_bgr, detections, class_names=None):
        return draw_detections(image_bgr, detections, class_names or self.class_names)

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # ================= SETTINGS =================
    ENGINE_FILE = "./models/yolo11.engine"   # <--- Your engine file
    IMAGE_PATH = "test_images"       # <--- Input folder
    OUTPUT_PATH = "test_results"     # <--- Output folder to save results
    CLASSES_FILE = "classes.txt"     # <--- (Optional) Path to classes.txt
    # ============================================

    # 1. Setup Directories
    out_dir = Path(OUTPUT_PATH)
    out_dir.mkdir(parents=True, exist_ok=True) # Create output folder if it doesn't exist

    print("Initializing TensorRT Detector...")
    try:
        detector = TrtYoloDetector(
            engine_path=ENGINE_FILE, 
            class_names=CLASSES_FILE, # Pass the classes file if you have one
            conf_thres=0.3,
            iou_thres=0.5
        )
    except FileNotFoundError:
        print(f"ERROR: Could not find engine file at: {ENGINE_FILE}")
        exit()
    except Exception as e:
        print(f"Error: {e}")
        exit()

    # 2. Collect Images
    images = collect_images(IMAGE_PATH)
    print(f"Found {len(images)} images in '{IMAGE_PATH}'")

    if not images:
        print(f"No images found in {IMAGE_PATH}. Please create the folder and add images.")
        exit()

    # 3. Process Batch
    total_objects_all_images = 0
    start_time_all = time.perf_counter()

    for img_file in images:
        print(f"\nProcessing: {img_file.name}")
        frame = cv2.imread(str(img_file))
        
        if frame is None:
            print("Failed to load image.")
            continue

        # Run Inference
        results = detector.predict(frame, verbose=True)
        count = len(results)
        total_objects_all_images += count

        # Visualize and Save
        if count > 0:
            # Draw boxes
            annotated_frame = detector.visualize(frame, results)
        else:
            # Save original if nothing detected
            annotated_frame = frame
            print("No detections found.")

        # Save to test_results folder
        save_file = out_dir / img_file.name
        cv2.imwrite(str(save_file), annotated_frame)
        print(f"Saved result to: {save_file}")

    total_time = time.perf_counter() - start_time_all
    
    # 4. Final Summary
    print("\n" + "="*40)
    print(f"BATCH COMPLETE")
    print(f"Images Processed: {len(images)}")
    print(f"Total Objects Detected: {total_objects_all_images}")
    print(f"Total Time: {total_time:.2f}s")
    print("="*40)
    
    cv2.destroyAllWindows()
