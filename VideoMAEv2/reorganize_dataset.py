import os
import shutil

base_path = r"C:\Users\V\Desktop\AMRITA\FINAL_YEAR_PROJECT\crime_dataset"
dataset_path = os.path.join(base_path, "DATASET")
unified_path = os.path.join(base_path, "unified")

os.makedirs(unified_path, exist_ok=True)

mappings = {
    r"Ankith P\Dragging & Pulling": "dragging",
    r"Ankith P\Hitting & Throwing": "hitting",
    r"Ankith P\Stealing & Snatching": "snatching",
    r"Hemanth.B\Kicking": "kicking",
    r"Hemanth.B\Pushing": "pushing",
    r"Hemanth.B\robbery": "robbery",
    r"N Praveen\hair pulling": "hair_pulling",
    r"N Praveen\punch": "punching",
    r"N Praveen\slap": "slapping",
    r"P hemanth\abuse": "abuse",
    r"P hemanth\neck grabbing": "neck_grabbing",
    r"P hemanth\shooting": "shooting",
    r"R yasaswini\stabing": "stabbing",
}

for src_rel, dest_folder in mappings.items():
    src = os.path.join(dataset_path, src_rel)
    dest = os.path.join(unified_path, dest_folder)
    if os.path.isdir(src):
        print(f"Creating directory: {dest}")
        os.makedirs(dest, exist_ok=True)
        print(f"Copying files from {src} to {dest}...")
        for item in os.listdir(src):
            s_file = os.path.join(src, item)
            d_file = os.path.join(dest, item)
            if os.path.isfile(s_file):
                shutil.copy2(s_file, d_file)
    else:
        print(f"[WARNING] Source folder not found: {src}")

print("\nDataset reorganization complete! You are ready to start training.")
