import os
import ast
import pandas as pd

ROOT = "/home/y_rajeshwar/final_year_project_29/dataset/2sec/langbind_features"
OUTPUT_ROOT = "/home/y_rajeshwar/final_year_project_29/dataset/2sec/metadata"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for split in ["train", "test"]:

    split_root = os.path.join(ROOT, split)

    rows = []

    if not os.path.exists(split_root):
        print(f"Missing split: {split_root}")
        continue

    classes = sorted(
        d for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
    )

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    print(f"\n{split} classes:")
    print(class_to_idx)

    for class_name in classes:

        class_root = os.path.join(split_root, class_name)

        for video_name in sorted(os.listdir(class_root)):

            video_root = os.path.join(class_root, video_name)

            if not os.path.isdir(video_root):
                continue

            segment_paths = []

            for segment_name in sorted(os.listdir(video_root)):

                segment_root = os.path.join(
                    video_root,
                    segment_name
                )

                if not os.path.isdir(segment_root):
                    continue

                pt_path = os.path.join(
                    segment_root,
                    "langbind_output.pt"
                )

                if os.path.isfile(pt_path):
                    segment_paths.append(pt_path)

            if not segment_paths:
                continue

            rows.append({
                "segment_idx": len(rows),
                "video_filename": video_name,
                "segment_paths": segment_paths,
                "number_of_segments": len(segment_paths),
                "label": class_to_idx[class_name],
                "label_name": class_name,
            })

    df = pd.DataFrame(rows)

    output_file = os.path.join(
        OUTPUT_ROOT,
        f"{split}.csv"
    )

    df.to_csv(output_file, index=False)

    print("\nSaved:", output_file)
    print("Videos:", len(df))
    print("Total segments:", df["number_of_segments"].sum())

    print("\nVideos per class:")
    print(df["label_name"].value_counts().sort_index())
