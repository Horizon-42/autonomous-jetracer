"""Train a MobileNet-V1 SSD detector using a YOLO-format dataset.

Pipeline:
1) Unzip YOLO dataset.
2) Convert YOLO labels to Pascal VOC (pytorch-ssd expects VOC).
3) Split into train/val and write ImageSets files.
4) Patch pytorch-ssd for newer PyTorch load semantics.
5) Launch training and copy the newest checkpoint to output_models/.
"""

import os
import shutil
import zipfile
import glob
import sys
import subprocess
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Input dataset zip in YOLO format (images + .txt labels + classes.txt).
ZIP_PATH = './yolo_dataset.zip'         
DATA_DIR = './data/voc_dataset'         
OUTPUT_DIR = './output_models'          
REPO_DIR = 'pytorch-ssd'

# Training hyperparameters for pytorch-ssd.
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

def run_command(cmd):
    """Executes a shell command and exits on failure."""
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        sys.exit(1)

# 1. Setup Environment
# Clone pytorch-ssd if missing and ensure boto3 is installed.
print("Setting up environment...")
if not os.path.exists(REPO_DIR):
    print(f"Cloning {REPO_DIR}...")
    run_command('git clone https://github.com/dusty-nv/pytorch-ssd')
    # Install boto3 locally if needed
    run_command(f'{sys.executable} -m pip install boto3')

# 2. Extract Data
# Work in a temp folder so we can inspect/convert freely.
if os.path.exists('./temp_yolo'): shutil.rmtree('./temp_yolo')
if os.path.exists(DATA_DIR): shutil.rmtree(DATA_DIR)
os.makedirs('./temp_yolo', exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Unzipping {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall('./temp_yolo')

# 3. Analyze Class IDs
# Read classes.txt if present, otherwise use fallback classes.
print("Analyzing Class IDs...")
classes_file = None
for root, dirs, files in os.walk('./temp_yolo'):
    if 'classes.txt' in files:
        classes_file = os.path.join(root, 'classes.txt')
        break

if classes_file:
    with open(classes_file, 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
else:
    # Fallback class list if classes.txt is missing.
    CLASS_NAMES = ['stop_sign', 'person', 'car']

# Scan for max ID to ensure compatibility with datasets that have missing classes.
max_id_found = -1
all_txts = []
for root, dirs, files in os.walk('./temp_yolo'):
    for file in files:
        if file.endswith('.txt') and file != 'classes.txt':
            all_txts.append(os.path.join(root, file))

for txt_file in all_txts:
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5: 
                try:
                    cid = int(parts[0])
                    if cid > max_id_found: max_id_found = cid
                except ValueError: continue

# Extend the class list if labels reference IDs beyond classes.txt.
if max_id_found >= len(CLASS_NAMES):
    print(f"Extending class list to cover ID {max_id_found}...")
    for i in range(len(CLASS_NAMES), max_id_found + 1):
        CLASS_NAMES.append(f"class_{i}")

print(f"Final Class List: {CLASS_NAMES}")

# 4. Convert Data (YOLO -> VOC)
# YOLO txt -> VOC XML so pytorch-ssd can train.
print("Converting YOLO data to Pascal VOC format...")
img_dir = os.path.join(DATA_DIR, 'JPEGImages')
ann_dir = os.path.join(DATA_DIR, 'Annotations')
sets_dir = os.path.join(DATA_DIR, 'ImageSets', 'Main')
os.makedirs(img_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)
os.makedirs(sets_dir, exist_ok=True)

def write_xml(xml_path, img_name, w, h, boxes):
    """Write a Pascal VOC XML file for a single image."""
    root = ET.Element('annotation')
    ET.SubElement(root, 'filename').text = img_name
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(w)
    ET.SubElement(size, 'height').text = str(h)
    ET.SubElement(size, 'depth').text = '3'
    for cls_id, xmin, ymin, xmax, ymax in boxes:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = CLASS_NAMES[int(cls_id)]
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(xml_path, "w") as f: f.write(xmlstr)

valid_ids = []
image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
all_images = []
for root, dirs, files in os.walk('./temp_yolo'):
    for file in files:
        if os.path.splitext(file)[1].lower() in image_exts:
            all_images.append(os.path.join(root, file))

for img_path in all_images:
    filename = os.path.basename(img_path)
    file_id = os.path.splitext(filename)[0]
    label_path = os.path.join(os.path.dirname(img_path), file_id + '.txt')
    if not os.path.exists(label_path): label_path = label_path.replace('images', 'labels')
    if not os.path.exists(label_path): continue 

    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape
    shutil.copy(img_path, os.path.join(img_dir, filename))

    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                cid = int(p[0])
                # YOLO labels are normalized: class cx cy w h (relative to image).
                mx, my, mw, mh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                xmin = int((mx - mw/2) * w); ymin = int((my - mh/2) * h)
                xmax = int((mx + mw/2) * w); ymax = int((my + mh/2) * h)
                xmin=max(0,xmin); ymin=max(0,ymin); xmax=min(w,xmax); ymax=min(h,ymax)
                boxes.append([cid, xmin, ymin, xmax, ymax])

    # Only include images that actually have at least one labeled object.
    if boxes:
        write_xml(os.path.join(ann_dir, file_id + '.xml'), filename, w, h, boxes)
        valid_ids.append(file_id)

# Create Splits
# VOC uses ImageSets/Main/*.txt files listing image ids (no extensions).
train_ids, val_ids = train_test_split(valid_ids, test_size=0.1, random_state=42)
with open(os.path.join(sets_dir, 'train.txt'), 'w') as f: f.write('\n'.join(train_ids))
with open(os.path.join(sets_dir, 'val.txt'), 'w') as f: f.write('\n'.join(val_ids))
with open(os.path.join(sets_dir, 'test.txt'), 'w') as f: f.write('\n'.join(val_ids))
with open(os.path.join(sets_dir, 'trainval.txt'), 'w') as f: f.write('\n'.join(train_ids + val_ids))

with open(os.path.join(DATA_DIR, 'labels.txt'), 'w') as f: f.write('\n'.join(CLASS_NAMES))
print("Conversion Complete.")

# 5. Patch PyTorch Compatibility (Local File Edit)
# Newer PyTorch versions require weights_only=False for older checkpoints.
print("Patching source code for PyTorch compatibility...")
target_file = os.path.join(REPO_DIR, 'vision/ssd/ssd.py')

if os.path.exists(target_file):
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Replace the legacy load call with safe weights_only=False
    old_code = "state_dict = torch.load(model, map_location=lambda storage, loc: storage)"
    new_code = "state_dict = torch.load(model, map_location=lambda storage, loc: storage, weights_only=False)"
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(target_file, 'w') as f:
            f.write(content)
        print("Patch applied successfully.")
    elif "weights_only=False" in content:
        print("Patch already applied.")
    else:
        print("Warning: Could not locate code to patch.")
else:
    print(f"Error: Could not find {target_file}")

# 6. Run Training
print("Starting MobileNet-V1 SSD Training...")

# Construct the absolute path for data to avoid directory confusion
abs_data_dir = os.path.abspath(DATA_DIR)
abs_model_dir = os.path.abspath(os.path.join('models', 'myssd'))

# Execute training script from within the repo directory.
cmd = (
    f"cd {REPO_DIR} && {sys.executable} train_ssd.py "
    f"--dataset-type=voc "
    f"--data='{abs_data_dir}' "
    f"--model-dir='{abs_model_dir}' "
    f"--batch-size={BATCH_SIZE} "
    f"--epochs={EPOCHS} "
    f"--lr={LEARNING_RATE}"
)

run_command(cmd)

# 7. Save Results
print("Saving Checkpoint to output directory...")
# Look for result inside the repo structure
checkpoint_pattern = os.path.join(REPO_DIR, 'models/myssd', "mb1-ssd-Epoch-*-Loss-*.pth")
checkpoints = glob.glob(checkpoint_pattern)

if checkpoints:
    latest_pth = sorted(checkpoints)[-1] 
    filename = os.path.basename(latest_pth)
    
    dest_path = os.path.join(OUTPUT_DIR, filename)
    shutil.copy(latest_pth, dest_path)
    print(f"Saved Checkpoint to: {dest_path}")
    
    shutil.copy(os.path.join(DATA_DIR, 'labels.txt'), os.path.join(OUTPUT_DIR, 'labels.txt'))
    print("Saved labels.txt")
else:
    print("No trained model found.")
