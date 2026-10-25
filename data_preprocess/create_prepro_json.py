import os
import json

# -------------------------------------------------------
# Dataset root
# -------------------------------------------------------

DATASET_ROOT = "/home/y_rajeshwar/final_year_project_29/ucfcrime_lvmae_dataset"

# -------------------------------------------------------
# Where LanguageBind embeddings will be saved
# -------------------------------------------------------

OUTPUT_ROOT = "/home/y_rajeshwar/final_year_project_29/ucfcrime_langbind_features"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

file_paths = []

# -------------------------------------------------------
# Scan train/test videos
# -------------------------------------------------------

for split in ["train", "test"]:

    split_dir = os.path.join(DATASET_ROOT, split)

    if not os.path.exists(split_dir):
        continue

    for video_folder in sorted(os.listdir(split_dir)):

        video_dir = os.path.join(split_dir, video_folder)

        if not os.path.isdir(video_dir):
            continue

        input_path = os.path.join(video_dir, "video.mp4")

        if not os.path.isfile(input_path):
            continue

        output_path = os.path.join(
            OUTPUT_ROOT,
            split,
            video_folder
        )

        file_paths.append([
            input_path,
            output_path
        ])

# -------------------------------------------------------
# Save JSON
# -------------------------------------------------------

os.makedirs("data_preprocess/test_assets", exist_ok=True)

json_path = "data_preprocess/test_assets/prepro_path_tuples.json"

with open(json_path, "w") as f:
    json.dump(
        {
            "file_paths": file_paths
        },
        f,
        indent=4
    )

print("=" * 60)
print(f"Total videos : {len(file_paths)}")
print(f"JSON saved to: {json_path}")
print("=" * 60)
