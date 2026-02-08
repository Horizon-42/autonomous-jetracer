import torch
import timm
import os
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_to_onnx(pth_path, output_path):
    # 1. Define Architecture 
    # Must match training EXACTLY: resnet18, 2 classes, 1 channel (grayscale)
    device = torch.device("cpu")
    model = timm.create_model('resnet18', pretrained=False, num_classes=2, in_chans=1)
    
    # 2. Load Weights
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

    # 3. Set to Eval Mode
    model.to(device).eval()

    # 4. Create Dummy Input
    # Shape: [Batch=1, Channels=1, Height=224, Width=224]
    dummy_input = torch.randn(1, 1, 224, 224, device=device)

    # 5. Export
    logger.info(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=11,          # Best for Jetson/TensorRT
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
