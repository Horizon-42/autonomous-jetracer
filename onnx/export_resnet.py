"""Export a trained ResNet-18 steering model to ONNX for TensorRT."""

import torch
import timm
import os
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_to_onnx(pth_path, output_path):
    # 1. Define architecture.
    # Must match training EXACTLY: resnet18, 2 outputs, 1-channel (grayscale).
    device = torch.device("cpu")
    model = timm.create_model('resnet18', pretrained=False, num_classes=2, in_chans=1)
    
    # 2. Load weights.
    if not os.path.exists(pth_path):
        logger.error(f"❌ File not found: {pth_path}")
        return

    try:
        # Load directly (no wrapper class needed)
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"✅ Weights loaded from {pth_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load weights: {e}")
        return

    # 3. Set to eval mode.
    model.to(device).eval()

    # 4. Create dummy input with the fixed model input shape.
    # Shape: [batch=1, channels=1, height=224, width=224]
    dummy_input = torch.randn(1, 1, 224, 224, device=device)

    # 5. Export to ONNX.
    logger.info(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=11,          # Stable for Jetson/TensorRT
        do_constant_folding=True,
        input_names=['input_0'],
        output_names=['output_0'],
        dynamic_axes=None          # Fixed shape is faster on Jetson
    )
    logger.info(f"🎉 Success! ONNX model saved to {output_path}")

if __name__ == "__main__":
    # Ensure this matches the file name from your training script
    PTH_FILE = 'resnet18_best_v3.pth' 
    ONNX_FILE = 'resnet18_merged_v3.onnx'
    
    export_to_onnx(PTH_FILE, ONNX_FILE)
