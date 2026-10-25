import os
import pandas as pd
from tqdm import tqdm

ROOT = "/home/y_rajeshwar/final_year_project_29/ucfcrime_lvmae_dataset/train"

SEGMENT_VERSION = "2_second_segments"

segment_details = []

print("=" * 60)
print("Scanning segmented videos...")
print("=" * 60)

video_idx = 0

for cls in sorted(os.listdir(ROOT)):

    cls_path = os.path.join(ROOT, cls)

    if not os.path.isdir(cls_path):
        continue

    for video_folder in tqdm(sorted(os.listdir(cls_path)), desc=cls):

        video_dir = os.path.join(cls_path, video_folder)

        segment_root = os.path.join(video_dir, SEGMENT_VERSION)

        if not os.path.isdir(segment_root):
            continue

        for timestamp in sorted(os.listdir(segment_root)):

            segment_dir = os.path.join(segment_root, timestamp)

            if not os.path.isdir(segment_dir):
                continue

            segment_details.append({
                "segment_idx": len(segment_details),
                "segment_timestamp": timestamp,
                "segment_version": SEGMENT_VERSION,
                "class": cls,
                "video_folder": video_folder,
                "segment_path": segment_dir,
            })

OUTPUT_DIR = os.path.join(ROOT, "2_second_segments_metadata")
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = os.path.join(
    OUTPUT_DIR,
    "segments_metadata.csv"
)

df = pd.DataFrame(segment_details)

df.to_csv(csv_path, index=False)

print("\nFinished.")
print(f"Videos scanned : {video_idx}")
print(f"Segments       : {len(df)}")
print(f"Saved CSV      : {csv_path}")
