import os
import pandas as pd

FEATURE_ROOT = "/home/y_rajeshwar/final_year_project_29/dataset/langbind_features"
OUTPUT_DIR = "/home/y_rajeshwar/final_year_project_29/dataset"


def create_csv(split):
    rows = []

    split_root = os.path.join(FEATURE_ROOT, split)

    for root, dirs, files in os.walk(split_root):
        if "langbind_output.pt" in files:
            pt_path = os.path.join(root, "langbind_output.pt")

            rows.append({
                "segment_paths": str([pt_path]),
                "number_of_segments": 1
            })

    df = pd.DataFrame(rows)

    out_csv = os.path.join(OUTPUT_DIR, f"{split}_segments.csv")
    df.to_csv(out_csv, index=False)

    print(f"{split}: {len(df)} samples")
    print(f"Saved -> {out_csv}")


for split in ["train", "val", "test"]:
    create_csv(split)
