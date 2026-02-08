"""Merge multiple DonkeyCar-style datasets into a single catalog + images folder.

This script accepts one or more dataset roots, each of which may be:
1) A pre-merged dataset containing `catalog.csv` + `images/`, or
2) A directory of individual session folders, each with its own catalog.

It copies all images into one output folder and rewrites the catalog so each
image filename is globally unique (source + session + original filename).
"""

import os
import shutil
import json
import csv
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
# List all folders that contain sessions you want to include.
# Each source can be either a single merged dataset or a directory of sessions.
INPUT_SOURCES = [
    './merged_dataset_v2',  # Your existing stable data
    './new_dataset',        # High-speed data 1
]

# The name of your new, grand-master dataset.
OUTPUT_DIR = 'merged_dataset_v3'

# 1. Setup Output Directories
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
output_csv_path = os.path.join(OUTPUT_DIR, 'catalog.csv')

# Accumulate all rows across all sources into a single catalog.
master_catalog = []

# 2. Iterate through every source root
for source_root in INPUT_SOURCES:
    if not os.path.exists(source_root):
        print(f"Skipping {source_root}, directory not found.")
        continue

    # Identify if the source is a direct dataset or a folder containing sessions.
    # We check for a catalog.csv directly in the source_root first.
    direct_catalog = os.path.join(source_root, 'catalog.csv')
    
    if os.path.exists(direct_catalog):
        # This is a pre-merged dataset (like merged_dataset_v1).
        session_folders = [source_root]
        is_pre_merged = True
    else:
        # This is a folder containing many session folders (like genuine_dataset).
        session_folders = [os.path.join(source_root, f) for f in os.listdir(source_root) 
                           if os.path.isdir(os.path.join(source_root, f))]
        is_pre_merged = False

    print(f"Processing source: {source_root} ({len(session_folders)} sessions/folders)...")

    for session_path in tqdm(session_folders, desc=f"Merging {os.path.basename(source_root)}"):
        session_id = os.path.basename(session_path)
        
        # Avoid recursive merging (e.g., if OUTPUT_DIR lives inside INPUT_SOURCES).
        if session_id == OUTPUT_DIR:
            continue

        # Each session can store metadata in either catalog.csv or catalog.json.
        csv_path = os.path.join(session_path, 'catalog.csv')
        json_path = os.path.join(session_path, 'catalog.json')
        
        session_data = []
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                session_data = df.to_dict('records')
            except:
                # Skip corrupted CSVs instead of hard failing.
                continue
        elif os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    session_data = json.load(f)
            except:
                # Skip malformed JSON sessions.
                continue
        else:
            continue 

        # Process Images & Metadata
        for row in session_data:
            original_img_name = row['cam/image_array']
            # If it was already merged, images are in 'images/', otherwise
            # they are relative to the session folder.
            img_subfolder = 'images' 
            original_img_path = os.path.join(session_path, img_subfolder, original_img_name)
            
            # Create a globally unique name using source name + session + original name.
            source_prefix = os.path.basename(source_root)
            new_img_name = f"{source_prefix}_{session_id}_{original_img_name}"
            new_img_path = os.path.join(OUTPUT_DIR, 'images', new_img_name)
            
            if os.path.exists(original_img_path):
                shutil.copy2(original_img_path, new_img_path)
                
                # Keep only the fields required by the training pipeline.
                master_catalog.append({
                    'cam/image_array': new_img_name,
                    'user/angle': row['user/angle'],
                    'user/throttle': row['user/throttle'],
                    'timestamp': row.get('timestamp', 0)
                })

# 3. Save Master CSV
print(f"\nSaving Master CSV with {len(master_catalog)} frames...")
keys = ['cam/image_array', 'user/angle', 'user/throttle', 'timestamp']

with open(output_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(master_catalog)

# 4. Zip it for easy transport/backups.
print("Zipping dataset...")
shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
print(f"DONE! Merged everything into {OUTPUT_DIR}.zip")
