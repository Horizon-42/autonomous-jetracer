"""Deprecated: export MobileNet-V1 SSD weights to ONNX.

This script lives under deprecated/ because newer pipelines use YOLO.
It is kept for reference when converting legacy MobileNet SSD models.
"""

import sys
import os
import torch

# --- 1. FIX THE IMPORT PATH ---
# This tells Python: "Look inside the 'pytorch-ssd' folder for modules".
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(current_dir, 'pytorch-ssd')
sys.path.append(repo_path)

# Now this import will work!
try:
    from vision.ssd.mobilenet_v1_ssd import create_mobilenetv1_ssd
except ImportError:
    print("❌ Error: Could not find 'vision' module.")
    print(f"   Checked in: {repo_path}")
    print("   Make sure the 'pytorch-ssd' folder is in this directory!")
    sys.exit(1)

# --- CONFIGURATION ---
INPUT_MODEL = "mb1-ssd.pth"      # The file listed in your ls
LABELS_FILE = "labels.txt"       # The file listed in your ls
OUTPUT_ONNX = "ssd-mobilenet.onnx"

def export():
    # 2. Auto-Detect Class Count
    if not os.path.exists(LABELS_FILE):
        print(f"❌ Error: {LABELS_FILE} not found.")
        return

    with open(LABELS_FILE, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    # IMPORTANT: pytorch-ssd adds a 'BACKGROUND' class at index 0
    # So if you have 3 classes (car, person, sign), the model actually has 4.
    num_classes = len(class_names) + 1
    print(f"✅ Found {len(class_names)} classes in {LABELS_FILE}.")
    print(f"ℹ️  Model Architecture will use {num_classes} classes (including background).")

    # 3. Create Model Architecture
    net = create_mobilenetv1_ssd(num_classes, is_test=True)

    # 4. Load Weights
    print(f"🔄 Loading weights from {INPUT_MODEL}...")
    try:
        net.load(INPUT_MODEL)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    net.eval()
    net.to("cpu")

    # 5. Export to ONNX
    # MobileNet-V1-SSD standard resolution is 300x300
    dummy_input = torch.randn(1, 3, 300, 300)

    print(f"📤 Exporting to {OUTPUT_ONNX}...")
    try:
        torch.onnx.export(
            net,
            dummy_input,
            OUTPUT_ONNX,
            input_names=['input_0'],
            output_names=['scores', 'boxes'],
            opset_version=11,
            do_constant_folding=True
        )
        print(f"🎉 SUCCESS! Saved to {OUTPUT_ONNX}")
    except Exception as e:
        print(f"❌ ONNX Export Failed: {e}")

if __name__ == "__main__":
    export()
