from pathlib import Path
import json

DATASET_ROOT = Path("/home/y_rajeshwar/final_year_project_29/ucf_dataset")

video_index = {}

for video in DATASET_ROOT.rglob("*"):
    if video.suffix.lower() not in [".mp4", ".avi"]:
        continue

    cls = video.parent.name
    key = f"{cls}/{video.name}"

    video_index[key] = str(video.resolve())

print(f"Indexed {len(video_index)} videos")

with open("video_index.json", "w") as f:
    json.dump(video_index, f, indent=2)

print("Saved video_index.json")
