import os
import shutil
import hashlib
import random
import torch
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = "/home/y_rajeshwar/final_year_project_29/dataset/2sec/langbind_features"

OUTPUT_ROOT = "/home/y_rajeshwar/final_year_project_29/dataset/2sec/langbind_features_clean"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)


# ============================================================
# 1. COLLECT VIDEOS
# ============================================================

print("=" * 70)
print("COLLECTING VIDEOS")
print("=" * 70)

videos = []

for old_split in ["train", "test"]:

    split_dir = os.path.join(SOURCE_ROOT, old_split)

    for cls in sorted(os.listdir(split_dir)):

        cls_dir = os.path.join(split_dir, cls)

        if not os.path.isdir(cls_dir):
            continue

        for video in sorted(os.listdir(cls_dir)):

            video_dir = os.path.join(cls_dir, video)

            if not os.path.isdir(video_dir):
                continue

            segment_files = []

            for segment in sorted(os.listdir(video_dir)):

                pt = os.path.join(
                    video_dir,
                    segment,
                    "langbind_output.pt"
                )

                if os.path.isfile(pt):
                    segment_files.append(pt)

            if segment_files:

                videos.append({
                    "old_split": old_split,
                    "class": cls,
                    "video": video,
                    "path": video_dir,
                    "segments": segment_files
                })

print("Total videos found:", len(videos))


# ============================================================
# 2. CREATE HASHES FOR EVERY EMBEDDING
# ============================================================

print("\n" + "=" * 70)
print("HASHING SEGMENT EMBEDDINGS")
print("=" * 70)

hash_to_videos = defaultdict(set)

for i, video in enumerate(videos):

    for pt in video["segments"]:

        data = torch.load(pt, map_location="cpu")

        emb = data["embedding"].float().flatten()

        h = hashlib.sha256(
            emb.numpy().tobytes()
        ).hexdigest()

        hash_to_videos[h].add(i)

print("Unique embedding hashes:", len(hash_to_videos))


# ============================================================
# 3. BUILD CONNECTED COMPONENTS
#
# If video A shares an embedding with video B,
# they MUST stay in the same split.
#
# Transitive duplicates are also handled:
# A == B and B == C -> A,B,C stay together.
# ============================================================

print("\n" + "=" * 70)
print("BUILDING DUPLICATE GROUPS")
print("=" * 70)

parent = list(range(len(videos)))


def find(x):

    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]

    return x


def union(a, b):

    ra = find(a)
    rb = find(b)

    if ra != rb:
        parent[rb] = ra


for indices in hash_to_videos.values():

    indices = list(indices)

    if len(indices) > 1:

        first = indices[0]

        for other in indices[1:]:
            union(first, other)


components = defaultdict(list)

for i in range(len(videos)):
    components[find(i)].append(i)

groups = list(components.values())

print("Total unique video groups:", len(groups))

duplicate_video_groups = [
    g for g in groups
    if len(g) > 1
]

print("Duplicate-linked groups:", len(duplicate_video_groups))


# ============================================================
# 4. CHECK FOR LABEL CONFLICTS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING LABEL CONFLICTS")
print("=" * 70)

conflict_groups = []

for group in groups:

    labels = {
        videos[i]["class"]
        for i in group
    }

    if len(labels) > 1:
        conflict_groups.append(group)

print("Groups containing multiple labels:", len(conflict_groups))

for n, group in enumerate(conflict_groups, 1):

    print(f"\nCONFLICT GROUP {n}")

    for i in group:

        print(
            videos[i]["old_split"],
            videos[i]["class"],
            videos[i]["video"]
        )


# ============================================================
# IMPORTANT:
# Remove ambiguous duplicate groups having different labels.
# ============================================================

conflict_indices = set()

for group in conflict_groups:

    for i in group:
        conflict_indices.add(i)

clean_groups = []

for group in groups:

    if not any(i in conflict_indices for i in group):
        clean_groups.append(group)

print("\nGroups after removing label conflicts:", len(clean_groups))


# ============================================================
# 5. GROUP BY CLASS
# ============================================================

class_groups = defaultdict(list)

for group in clean_groups:

    # Safe because conflicting-label groups were removed
    cls = videos[group[0]]["class"]

    class_groups[cls].append(group)


# ============================================================
# 6. SPLIT EACH CLASS
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAIN / VAL / TEST SPLIT")
print("=" * 70)

assignments = {
    "train": [],
    "val": [],
    "test": []
}


for cls, cls_groups in sorted(class_groups.items()):

    random.shuffle(cls_groups)

    total_videos = sum(
        len(g)
        for g in cls_groups
    )

    target_train = total_videos * TRAIN_RATIO
    target_val = total_videos * VAL_RATIO

    counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    for group in cls_groups:

        size = len(group)

        if counts["train"] < target_train:

            split = "train"

        elif counts["val"] < target_val:

            split = "val"

        else:

            split = "test"

        assignments[split].extend(group)

        counts[split] += size

    print(
        f"{cls:20s} "
        f"train={counts['train']:3d} "
        f"val={counts['val']:3d} "
        f"test={counts['test']:3d}"
    )


# ============================================================
# 7. CREATE NEW DIRECTORY
# ============================================================

print("\n" + "=" * 70)
print("COPYING CLEAN DATASET")
print("=" * 70)

if os.path.exists(OUTPUT_ROOT):

    print("Removing previous clean output:")
    print(OUTPUT_ROOT)

    shutil.rmtree(OUTPUT_ROOT)


for split, indices in assignments.items():

    for i in indices:

        video = videos[i]

        destination = os.path.join(
            OUTPUT_ROOT,
            split,
            video["class"],
            video["video"]
        )

        # Protect against same video directory names
        if os.path.exists(destination):

            destination += (
                "__from_" +
                video["old_split"]
            )

        os.makedirs(
            os.path.dirname(destination),
            exist_ok=True
        )

        shutil.copytree(
            video["path"],
            destination
        )


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEAN SPLIT")
print("=" * 70)

for split in ["train", "val", "test"]:

    print(
        f"{split.upper():5s}: "
        f"{len(assignments[split])} videos"
    )

print(
    "\nRemoved because of conflicting duplicate labels:",
    len(conflict_indices),
    "videos"
)

print("\nOutput:")
print(OUTPUT_ROOT)

print("\nDONE.")
