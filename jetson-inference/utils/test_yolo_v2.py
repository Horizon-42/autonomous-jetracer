import os
import sys
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

# --- CONFIGURATION ---
REPO_ROOT = Path.cwd()
ENGINE_PATH = REPO_ROOT / "models" / "yolo11.engine"  # Adjust if needed
LABELS_PATH = REPO_ROOT / "labels.txt"
IMAGE_FOLDER = REPO_ROOT / "test_images"
OUTPUT_FOLDER = REPO_ROOT / "test_results"

# YOLO Thresholds
CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT_SIZE = 640


def load_class_names(namesfile):
    if isinstance(namesfile, (str, Path)) and os.path.exists(namesfile):
        with open(namesfile, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []


class TrtYoloDetector:
    def __init__(self, engine_path, class_names, conf_thres=0.25, iou_thres=0.45, input_size=640):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size
        self.last_inference_time = None
        
        print(f"\n🚀 [INIT] Loading YOLO engine: {engine_path}")
        if not os.path.exists(engine_path):
             print(f"❌ [ERROR] Engine file not found at {engine_path}")
             sys.exit(1)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        print(f"   [INIT] Engine Bindings:")
        # Allocate buffers
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            
            is_input = self.engine.binding_is_input(i)
            type_str = "Input" if is_input else "Output"
            print(f"     - Binding {i} ({type_str}): Name='{name}', Shape={shape}, Dtype={dtype}")

            if dtype == trt.float32: nptype = np.float32
            elif dtype == trt.float16: nptype = np.float16
            else: nptype = np.float32
            
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, nptype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.allocations.append(int(device_mem))
            self.bindings.append(int(device_mem))
            
            binding = {
                "index": i, "name": name, "dtype": nptype,
                "shape": list(shape), "host": host_mem, "device": device_mem,
            }
            
            if is_input:
                self.inputs.append(binding)
                self.input_h = shape[2]
                self.input_w = shape[3]
            else:
                self.outputs.append(binding)

    def preprocess(self, img):
        """ Resize image with Letterbox padding """
        h, w, _ = img.shape
        target_h, target_w = self.input_h, self.input_w
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        
        image_resized = cv2.resize(img, (nw, nh))
        
        # Create padded image (Grey 114)
        image_padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dw, dh = (target_w - nw) // 2, (target_h - nh) // 2
        image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
        
        # Normalize and Transpose
        blob = image_padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)
        
        return blob, scale, (dw, dh)

    def predict(self, img, measure_time=True):
        t_start = time.time()

        # 1. Preprocess
        blob, scale, (dw, dh) = self.preprocess(img)
        
        # 2. Copy to GPU
        np.copyto(self.inputs[0]['host'], blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 4. Copy back
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # 5. Reshape Output
        output = self.outputs[0]['host']
        output = output.reshape(self.outputs[0]['shape'])
        output = np.transpose(output[0], (1, 0)) # Transpose to (Anchors, 84)
        
        # 6. Postprocess
        results = self.postprocess(output, scale, (dw, dh))

        t_end = time.time()
        if measure_time:
            self.last_inference_time = t_end - t_start
            
        return results

    def postprocess(self, prediction, scale, pad):
        dw, dh = pad
        
        # Filter by confidence
        max_scores = np.max(prediction[:, 4:], axis=1)
        mask = max_scores > self.conf_thres
        prediction = prediction[mask]
        
        if len(prediction) == 0: return []

        # Extract Boxes and Scores
        xc, yc, w, h = prediction[:, 0], prediction[:, 1], prediction[:, 2], prediction[:, 3]
        cls_scores = prediction[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        confidences = np.max(cls_scores, axis=1)

        # Convert Center-WH to TopLeft-BottomRight
        x1 = (xc - w / 2) - dw
        y1 = (yc - h / 2) - dh
        x2 = (xc + w / 2) - dw
        y2 = (yc + h / 2) - dh
        
        # Scale back to original image coordinates
        x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.conf_thres, self.iou_thres)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "box": boxes[i], # [x1, y1, x2, y2]
                    "conf": confidences[i],
                    "class_id": cls_ids[i]
                })
        return results

    def visualize(self, img, detections):
        vis_img = img.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d['box'])
            conf = d['conf']
            cls_id = d['class_id']
            
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"ID_{cls_id}"
            
            # Draw Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(vis_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return vis_img


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("="*50)
    print("       YOLO TENSORRT INFERENCE DIAGNOSTICS       ")
    print("="*50)

    # 1. Setup
    if not IMAGE_FOLDER.exists():
        print(f"❌ [ERROR] Folder '{IMAGE_FOLDER}' not found.")
        sys.exit(1)
        
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Labels
    labels = load_class_names(LABELS_PATH)
    print(f"\n📂 [SETUP] Loaded Labels: {len(labels)} classes found.")
    if len(labels) > 0:
        print(f"   - First 3 classes: {labels[:3]}...")
    else:
        print("   - ⚠️ Warning: Labels file empty or not found. Using numeric IDs.")

    # 3. Load Model
    detector = TrtYoloDetector(
        str(ENGINE_PATH),
        class_names=labels,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        input_size=INPUT_SIZE
    )

    # 4. Process Images
    image_files = sorted([f for f in IMAGE_FOLDER.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    print(f"\n📸 [SETUP] Found {len(image_files)} images in '{IMAGE_FOLDER}'")

    total_time = 0
    total_objects = 0

    print("\n" + "-"*50)
    print(" STARTING INFERENCE LOOP")
    print("-"*50)

    for i, img_path in enumerate(image_files):
        print(f"\n🔸 [{i+1}/{len(image_files)}] Processing: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None: 
            print("   ❌ Failed to read image")
            continue
        
        h, w = image.shape[:2]
        print(f"   - Image Size: {w}x{h}")

        # Inference
        detections = detector.predict(image, measure_time=True)
        
        # Logging
        if detector.last_inference_time is not None:
            inf_ms = detector.last_inference_time * 1000.0
            total_time += detector.last_inference_time
            print(f"   - Inference Time: {inf_ms:.2f} ms")
        
        num_dets = len(detections)
        total_objects += num_dets
        print(f"   - Objects Detected: {num_dets}")

        # --- DETAILED DIAGNOSTICS FOR THIS IMAGE ---
        if num_dets > 0:
            print("   - Detailed Results:")
            for d in detections:
                x1, y1, x2, y2 = map(int, d['box'])
                conf = d['conf']
                cls_id = d['class_id']
                label = labels[cls_id] if cls_id < len(labels) else f"ID_{cls_id}"
                
                print(f"      ➡️ {label:<15} (ID:{cls_id}) | Conf: {conf:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]")
        else:
             print("      (No objects passed confidence threshold)")
        # -------------------------------------------

        # Visualize and Save
        vis = detector.visualize(image, detections)
        out_path = OUTPUT_FOLDER / ("res_" + img_path.name)
        cv2.imwrite(str(out_path), vis)
        print(f"   - Saved result to: {out_path.name}")

    # Summary
    print("\n" + "="*50)
    print("       FINAL SUMMARY       ")
    print("="*50)
    print(f"✅ Processed:     {len(image_files)} images")
    print(f"✅ Total Objects: {total_objects}")
    if len(image_files) > 0:
        avg_time = (total_time / len(image_files)) * 1000
        print(f"✅ Avg Inf Time:  {avg_time:.2f} ms per image")
    print(f"✅ Output Folder: {OUTPUT_FOLDER}")
    print("="*50 + "\n")
