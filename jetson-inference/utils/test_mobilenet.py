
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import sys


# --- CONFIGURATION ---

ENGINE_PATH = "./models/mb1-ssd.engine"   # Path to your TensorRT engine

LABELS_PATH = "labels_ori.txt"                # Path to your labels file

IMAGE_FOLDER = "test_images"              # Folder with input images

OUTPUT_FOLDER = "test_results"            # Folder to save output images

CONFIDENCE_THRESHOLD = 0.3                # Filter out weak detections

NMS_THRESHOLD = 0.45                      # Filter out overlapping boxes


# --- TENSORRT INFERENCE CLASS ---

class TRTInference:

    def __init__(self, engine_path):

        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:

            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

        

    def allocate_buffers(self, engine):

        inputs, outputs, bindings = [], [], []

        stream = cuda.Stream()

        for b in engine:

            size = trt.volume(engine.get_binding_shape(b))

            dtype = trt.nptype(engine.get_binding_dtype(b))

            host_mem = cuda.pagelocked_empty(size, dtype)

            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if engine.binding_is_input(b): 

                inputs.append({'host': host_mem, 'device': device_mem, 'shape': engine.get_binding_shape(b)})

            else: 

                outputs.append({'host': host_mem, 'device': device_mem, 'shape': engine.get_binding_shape(b)})

        return inputs, outputs, bindings, stream

        

    def infer(self, img_np):

        np.copyto(self.inputs[0]['host'], img_np.ravel())

        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:

            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        return [out['host'] for out in self.outputs]


# --- PREPROCESSING (The Fixes) ---


def preprocess(image):

    # 1. Resize to 300x300 (Standard SSD Input)

    resized = cv2.resize(image, (300, 300))

    

    # 2. Convert BGR to RGB (OpenCV uses BGR, Model usually needs RGB)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    

    # 3. Normalize (Standard MobileNet-SSD method)

    # If this yields bad results, try: rgb.astype(np.float32) / 255.0

    norm = (rgb.astype(np.float32) - 127.0) / 128.0

    

    # 4. HWC -> CHW (CRITICAL FIX)

    # Moves the Channels (3) from the end to the front. 

    # Without this, the model sees "static" and outputs garbage.

    chw = np.transpose(norm, (2, 0, 1))

    

    return chw


def postprocess(outputs, labels, w, h):

    # 1. Identify which output is Boxes vs Scores

    arr1, arr2 = outputs[0], outputs[1]

    if arr1.size < arr2.size:

        raw_boxes, raw_scores = arr1, arr2

    else:

        raw_boxes, raw_scores = arr2, arr1


    num_anchors = 3000

    num_classes = int(raw_scores.size / num_anchors)

    

    raw_boxes = raw_boxes.reshape(num_anchors, 4)

    raw_scores = raw_scores.reshape(num_anchors, num_classes)


    # 2. Filter Weak Detections

    class_ids = np.argmax(raw_scores, axis=1)

    scores = np.max(raw_scores, axis=1)

    

    # DEBUG: Print max score seen to verify model is working

    max_score = np.max(scores)

    # print(f"   (Debug) Max Score in image: {max_score:.4f}")


    mask = (scores > CONFIDENCE_THRESHOLD) & (class_ids > 0)

    filtered_boxes = raw_boxes[mask]

    filtered_scores = scores[mask]

    filtered_classes = class_ids[mask]


    if len(filtered_scores) == 0:

        return []


    # 3. Convert to Pixels (SMART FIX)

    boxes_pixels = []

    for box in filtered_boxes:

        xmin, ymin, xmax, ymax = box

        

        # SMART CHECK: Is the model outputting 0-300 or 0-1?

        # If any value is > 1.0, it is definitely using pixel coordinates (0-300).

        if xmin > 1.1 or ymin > 1.1 or xmax > 1.1:

            # Scale from Model Size (300) to Image Size

            scale_x = w / 300.0

            scale_y = h / 300.0

            x = int(xmin * scale_x)

            y = int(ymin * scale_y)

            bw = int((xmax - xmin) * scale_x)

            bh = int((ymax - ymin) * scale_y)

        else:

            # Standard Normalized (0.0 - 1.0)

            x = int(xmin * w)

            y = int(ymin * h)

            bw = int((xmax - xmin) * w)

            bh = int((ymax - ymin) * h)


        boxes_pixels.append([x, y, bw, bh])

    

    # 4. NMS (Remove Overlaps)

    indices = cv2.dnn.NMSBoxes(boxes_pixels, filtered_scores.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)


    results = []

    if len(indices) > 0:

        for i in indices.flatten():

            cls_id = filtered_classes[i]

            # Safety check for labels index

            label_name = labels[cls_id] if cls_id < len(labels) else f"ID_{cls_id}"

            results.append((label_name, float(filtered_scores[i]), boxes_pixels[i]))

            

    return results


# --- MAIN EXECUTION ---


if __name__ == "__main__":

    # 1. Setup Folders

    if not os.path.exists(IMAGE_FOLDER):

        print(f"❌ Error: Folder '{IMAGE_FOLDER}' not found.")

        print("   Please create it and put .jpg images inside.")

        sys.exit(1)

        

    if not os.path.exists(OUTPUT_FOLDER):

        os.makedirs(OUTPUT_FOLDER)


    # 2. Load Labels

    if not os.path.exists(LABELS_PATH):

         print(f"❌ Error: '{LABELS_PATH}' not found.")

         sys.exit(1)

         

    labels = []

    with open(LABELS_PATH, 'r') as f:

        # Strip newlines and empty lines

        labels = [line.strip() for line in f.readlines() if line.strip()]

    print(f"📂 Loaded {len(labels)} labels.")


    # 3. Load Model

    if not os.path.exists(ENGINE_PATH):

        print(f"❌ Error: Engine '{ENGINE_PATH}' not found.")

        sys.exit(1)

        

    print(f"🚀 Loading Engine: {ENGINE_PATH}...")

    model = TRTInference(ENGINE_PATH)


    # 4. Process Images

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"📸 Found {len(image_files)} images to test.")


    if len(image_files) == 0:

        print(f"⚠️ No images found in {IMAGE_FOLDER}!")


    for img_name in image_files:

        img_path = os.path.join(IMAGE_FOLDER, img_name)

        image = cv2.imread(img_path)

        if image is None: continue


        h, w = image.shape[:2]

        

        # Inference

        input_tensor = preprocess(image)

        raw_output = model.infer(input_tensor)

        detections = postprocess(raw_output, labels, w, h)


        # Draw Results

        print(f"🔍 {img_name}: Found {len(detections)} objects.")

        for label, score, box in detections:

            x, y, bw, bh = box

            

            # Print precise coords to debug

            print(f"   ➡️ {label}: {score:.2f} at [{x}, {y}, {bw}x{bh}]")

            

            # Draw Box

            cv2.rectangle(image, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

            # Draw Label Background

            cv2.rectangle(image, (x, y-20), (x+100, y), (0, 255, 0), -1)

            # Draw Text

            text = f"{label} {score:.2f}"

            cv2.putText(image, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


        # Save Result

        save_path = os.path.join(OUTPUT_FOLDER, "res_" + img_name)

        cv2.imwrite(save_path, image)


    print(f"\n✅ Done! Results saved to '{OUTPUT_FOLDER}' folder.") 
