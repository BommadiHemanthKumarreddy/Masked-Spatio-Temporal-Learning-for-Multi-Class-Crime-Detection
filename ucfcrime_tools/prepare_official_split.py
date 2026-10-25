import json
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
VIDEO_INDEX = Path("video_index.json")

TRAIN_SPLIT = Path(
    "/home/y_rajeshwar/final_year_project_29/ucf_dataset/"
    "UCF_Crimes-Train-Test-Split/Action_Regnition_splits/train_001.txt"
)

TEST_SPLIT = Path(
    "/home/y_rajeshwar/final_year_project_29/ucf_dataset/"
    "UCF_Crimes-Train-Test-Split/Action_Regnition_splits/test_001.txt"
)

# -----------------------------
# Ignore these classes
# -----------------------------
IGNORE_CLASSES = {
    "RoadAccidents",
    "Normal_Videos_event"
}

# -----------------------------
# Load index
# -----------------------------
with open(VIDEO_INDEX) as f:
    video_index = json.load(f)

# -----------------------------
# Function
# -----------------------------
def load_split(split_file):

    videos = []

    missing = []

    skipped = 0

    with open(split_file) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            cls = line.split("/")[0]

            if cls in IGNORE_CLASSES:
                skipped += 1
                continue

            if line in video_index:

                videos.append(
                    (
                        cls,
                        line,
                        video_index[line]
                    )
                )

            else:

                missing.append(line)

    return videos, missing, skipped


train_videos, train_missing, train_skipped = load_split(TRAIN_SPLIT)
test_videos, test_missing, test_skipped = load_split(TEST_SPLIT)

print("="*60)
print("TRAIN")
print("="*60)
print("Videos :", len(train_videos))
print("Skipped:", train_skipped)
print("Missing:", len(train_missing))

print()

print("="*60)
print("TEST")
print("="*60)
print("Videos :", len(test_videos))
print("Skipped:", test_skipped)
print("Missing:", len(test_missing))

if train_missing:
    print("\nFirst missing train:", train_missing[:5])

if test_missing:
    print("\nFirst missing test:", test_missing[:5])
