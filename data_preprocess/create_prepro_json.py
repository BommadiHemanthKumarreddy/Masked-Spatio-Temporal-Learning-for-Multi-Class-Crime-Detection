import os
import json

# Dataset root
DATASET_ROOT = "/home/y_rajeshwar/final_year_project_29/dataset"

# Where LanguageBind embeddings will be saved
OUTPUT_ROOT = "/home/y_rajeshwar/final_year_project_29/dataset/langbind_features"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

file_paths = []

for split in ["train", "test", "val"]:
    split_dir = os.path.join(DATASET_ROOT, split)

    if not os.path.exists(split_dir):
        continue

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith(video_extensions):

                input_path = os.path.join(class_dir, filename)

                video_name = os.path.splitext(filename)[0]

                output_path = os.path.join(
                    OUTPUT_ROOT,
                    split,
                    class_name,
                    video_name
                )

                file_paths.append([input_path, output_path])

os.makedirs("data_preprocess/test_assets", exist_ok=True)

json_path = "data_preprocess/test_assets/prepro_path_tuples.json"

with open(json_path, "w") as f:
    json.dump({"file_paths": file_paths}, f, indent=4)

print("=" * 60)
print(f"Total videos : {len(file_paths)}")
print(f"JSON saved to: {json_path}")
print("=" * 60)
