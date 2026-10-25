import json
import shutil
from pathlib import Path

# -------------------------------------------------
# Paths
# -------------------------------------------------

VIDEO_INDEX = Path("video_index.json")

TRAIN_SPLIT = Path(
    "/home/y_rajeshwar/final_year_project_29/ucf_dataset/"
    "UCF_Crimes-Train-Test-Split/Action_Regnition_splits/train_001.txt"
)

TEST_SPLIT = Path(
    "/home/y_rajeshwar/final_year_project_29/ucf_dataset/"
    "UCF_Crimes-Train-Test-Split/Action_Regnition_splits/test_001.txt"
)

# Official dataset (class-wise)
OUTPUT_ROOT = Path(
    "/home/y_rajeshwar/final_year_project_29/ucfcrime_official_split"
)

# LV-MAE dataset (one folder per video)
LVMAE_OUTPUT_ROOT = Path(
    "/home/y_rajeshwar/final_year_project_29/ucfcrime_lvmae_dataset"
)

# Ignore these classes
IGNORE_CLASSES = {
    "RoadAccidents",
    "Normal_Videos_event"
}

# -------------------------------------------------
# Load video index
# -------------------------------------------------

with open(VIDEO_INDEX) as f:
    video_index = json.load(f)


# -------------------------------------------------
# Create official train/test dataset
# -------------------------------------------------

def process_split(split_file, split_name):

    copied = 0

    with open(split_file) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            cls = line.split("/")[0]

            if cls in IGNORE_CLASSES:
                continue

            if line not in video_index:
                print(f"Missing: {line}")
                continue

            src = Path(video_index[line])

            dst = OUTPUT_ROOT / split_name / cls / src.name

            dst.parent.mkdir(parents=True, exist_ok=True)

            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1

    print(f"{split_name}: copied {copied} new videos")


# -------------------------------------------------
# Create LV-MAE dataset
# -------------------------------------------------

def create_lvmae_dataset():

    copied = 0

    for split in ["train", "test"]:

        split_dir = OUTPUT_ROOT / split

        if not split_dir.exists():
            continue

        for cls_dir in sorted(split_dir.iterdir()):

            if not cls_dir.is_dir():
                continue

            for video in sorted(cls_dir.iterdir()):

                if not video.is_file():
                    continue

                video_name = video.stem

                out_dir = LVMAE_OUTPUT_ROOT / split / video_name
                out_dir.mkdir(parents=True, exist_ok=True)

                dst_video = out_dir / "video.mp4"

                if not dst_video.exists():
                    shutil.copy2(video, dst_video)
                    copied += 1

    print(f"LV-MAE dataset: copied {copied} new videos")


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CREATING OFFICIAL DATASET")
    print("=" * 60)

    process_split(TRAIN_SPLIT, "train")
    process_split(TEST_SPLIT, "test")

    print("\n" + "=" * 60)
    print("CREATING LV-MAE DATASET")
    print("=" * 60)

    create_lvmae_dataset()

    print("\nFinished.")
