#!/usr/bin/env python3

"""
Human-Aware 2-Second LanguageBind Preprocessing
================================================

Pipeline:

    Original Video
          |
          v
    Split into 2-second clips
          |
          v
    Sample frames from each clip
          |
          v
    Faster R-CNN human detection
          |
          v
    Create soft human-region mask
          |
          v
    Enhance human regions
          |
          v
    LanguageBind Video Encoder
          |
          v
    1024-D embedding
          |
          v
    Save langbind_output.pt

Important:
- Uses the same LanguageBind implementation as lv-mae-2sec.
- Uses LanguageBind_Video_Huge_V1.5_FT.
- Human detection is performed using Faster R-CNN.
- Human regions are enhanced, not background-deleted.
- Output directory preserves the original extension to avoid
  collisions between files such as:
      video.MOV
      video.MP4

Expected JSON format:

{
    "file_paths": [
        [
            "/path/to/video.mp4",
            "/path/to/output/directory"
        ]
    ]
}

The second path is treated as the DIRECT output directory.
"""

import os
import sys
import cv2
import math
import json
import time
import shutil
import tempfile
import argparse
import subprocess
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
LVMAE_UCF_DIR = os.path.dirname(CURRENT_DIR)

# Current script is inside:
#   v_mea2/lv-mae-ucfcrime/data_preprocess/
#
# We use LanguageBind from:
#   v_mea2/lv-mae-2sec/
#
LVMAE_2SEC_DIR = os.path.join(
    os.path.dirname(LVMAE_UCF_DIR),
    "lv-mae-2sec"
)

if LVMAE_2SEC_DIR not in sys.path:
    sys.path.insert(0, LVMAE_2SEC_DIR)

# Also add current project directory
if LVMAE_UCF_DIR not in sys.path:
    sys.path.insert(0, LVMAE_UCF_DIR)


# ============================================================
# LANGUAGEBIND IMPORT
# ============================================================

try:
    from languagebind_2sec import (
        LanguageBind,
        to_device,
        transform_dict,
    )
except Exception as e:
    print("=" * 80)
    print("ERROR: Could not import languagebind_2sec")
    print("=" * 80)
    print("Expected location:")
    print(LVMAE_2SEC_DIR)
    print()
    print("Original error:")
    print(e)
    print()
    print("Make sure the lv-mae-2sec directory contains")
    print("the languagebind_2sec package.")
    print("=" * 80)
    raise


# ============================================================
# TORCHVISION IMPORT
# ============================================================

try:
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
except Exception as e:
    print("=" * 80)
    print("ERROR: Could not import torchvision detection model")
    print("=" * 80)
    print(e)
    raise


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_VIDEO_PATH_FILE = os.path.join(
    CURRENT_DIR,
    "test_assets",
    "prepro_path_tuples.json"
)

DEFAULT_OUTPUT_ROOT = (
    "/home/y_rajeshwar/final_year_project_29/"
    "langbind_features_human"
)

DEFAULT_END_JOB_FILE = "~/tmp/tmp_end_job_file.txt"

# LanguageBind
VIDEO_MODEL_NAME = "LanguageBind_Video_Huge_V1.5_FT"

# Device
DEFAULT_DEVICE = "cuda"

# Two-second segmentation
SEGMENT_DURATION = 2.0

# Faster R-CNN
DETECTION_SCORE_THRESHOLD = 0.50

# Number of frames sampled from every 2-second clip
DETECTION_SAMPLE_FRAMES = 8

# Human-region enhancement:
#
# I' = I * (1 + HUMAN_ALPHA * M)
#
# where M is a soft human mask.
#
# alpha = 0.5 means:
#   human pixels can be enhanced by up to 50%.
HUMAN_ALPHA = 0.50

# Gaussian blur size for soft mask
MASK_BLUR_KERNEL = 21

# Minimum bounding-box area relative to image area.
# Very tiny detections are ignored.
MIN_BOX_AREA_RATIO = 0.0005

# Person class in COCO
PERSON_CLASS_ID = 1


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_json(path):
    """
    Load JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def safe_remove(path):
    """
    Remove a file safely.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def format_seconds(seconds):
    """
    Pretty formatting for log output.
    """
    return f"{seconds:.2f}s"


# ============================================================
# COLLISION-SAFE OUTPUT DIRECTORY
# ============================================================

def collision_safe_output_name(video_path):
    """
    Preserve the original file extension in the output directory name.

    Example:

        video.MOV
        video.MP4

    would otherwise both become:

        video

    which creates an output collision.

    Instead:

        video__mov
        video__mp4
    """

    filename = os.path.basename(video_path)

    stem, ext = os.path.splitext(filename)

    safe_ext = ext.replace(".", "").lower()

    if safe_ext:
        return f"{stem}__{safe_ext}"

    return stem


def build_output_dir(output_root, video_path):
    """
    Build a collision-safe output directory.

    Expected structure:

        output_root/
            train/
                abuse/
                    video__mov/
                    video__mp4/

    OR:

        output_root/
            test/
                abuse/
                    video__mov/
    """

    video_dir = os.path.dirname(video_path)

    class_name = os.path.basename(video_dir)

    # Detect whether this video is inside train/test.
    parent_dir = os.path.dirname(video_dir)
    split_name = os.path.basename(parent_dir)

    if split_name not in ("train", "test", "val", "validation"):
        split_name = "unknown"

    output_name = collision_safe_output_name(video_path)

    return os.path.join(
        output_root,
        split_name,
        class_name,
        output_name
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):
    """
    Get FPS, frame count and duration.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps is None or fps <= 0:
        return None

    if frame_count is None or frame_count <= 0:
        return None

    duration = frame_count / fps

    return {
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration": float(duration),
    }


# ============================================================
# FFMPEG CLIP EXTRACTION
# ============================================================

def extract_clip(
    input_video,
    output_video,
    start_time,
    duration
):
    """
    Extract one clip using FFmpeg.

    The video stream is re-encoded to ensure that the temporary
    clip is readable by both OpenCV and LanguageBind.
    """

    cmd = [
        "/usr/bin/ffmpeg",
        "-y",

        "-ss",
        str(start_time),

        "-i",
        input_video,

        "-t",
        str(duration),

        "-c:v",
        "libx264",

        "-preset",
        "ultrafast",

        "-crf",
        "23",

        "-an",

        output_video,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed:\n" +
            result.stderr[-2000:]
        )


# ============================================================
# READ FRAMES
# ============================================================

def read_all_frames(video_path):
    """
    Read all frames from a short 2-second clip.

    Returns:
        frames: list of BGR NumPy arrays
        fps: video FPS
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open clip: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30.0

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames, float(fps)


# ============================================================
# SAMPLE FRAME INDICES
# ============================================================

def get_sample_indices(
    num_frames,
    num_samples=DETECTION_SAMPLE_FRAMES
):
    """
    Uniformly sample frame indices.

    Example:
        60 frames, 8 samples

    approximately gives:
        0, 8, 17, 25, ...
    """

    if num_frames <= 0:
        return []

    if num_frames <= num_samples:
        return list(range(num_frames))

    indices = np.linspace(
        0,
        num_frames - 1,
        num=num_samples,
        dtype=np.int64
    )

    return sorted(set(indices.tolist()))


# ============================================================
# FASTER R-CNN MODEL
# ============================================================

def load_detector(device):
    """
    Load pretrained Faster R-CNN.

    COCO person class:
        class ID = 1
    """

    print()
    print("=" * 80)
    print("Loading Faster R-CNN human detector")
    print("=" * 80)

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    detector = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    detector = detector.to(device)

    detector.eval()

    # TorchVision preprocessing.
    # Important:
    # weights.transforms() expects PIL images.
    detector_transform = weights.transforms()

    print("Detector      : Faster R-CNN ResNet50 FPN")
    print("Weights       : COCO pretrained")
    print("Person class  : 1")
    print("Threshold     :", DETECTION_SCORE_THRESHOLD)
    print("Device        :", device)

    return detector, detector_transform


# ============================================================
# DETECT PEOPLE
# ============================================================

@torch.no_grad()
def detect_people(
    frame,
    detector,
    detector_transform,
    device,
    score_threshold=DETECTION_SCORE_THRESHOLD
):
    """
    Detect humans in one frame.

    Input:
        frame = BGR NumPy array from OpenCV.

    Important:
        TorchVision weights.transforms() expects PIL Image,
        so we explicitly convert:

            BGR -> RGB -> PIL

    Returns:
        list of bounding boxes:

            [
                [x1, y1, x2, y2],
                ...
            ]
    """

    if frame is None:
        return []

    # OpenCV:
    #   BGR
    #
    # Faster R-CNN preprocessing:
    #   PIL RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)

    image_tensor = detector_transform(
        pil_image
    ).to(device)

    outputs = detector(
        [image_tensor]
    )

    output = outputs[0]

    boxes = output["boxes"]
    labels = output["labels"]
    scores = output["scores"]

    h, w = frame.shape[:2]

    min_area = (
        h *
        w *
        MIN_BOX_AREA_RATIO
    )

    human_boxes = []

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        score_value = float(
            score.detach().cpu().item()
        )

        label_value = int(
            label.detach().cpu().item()
        )

        if label_value != PERSON_CLASS_ID:
            continue

        if score_value < score_threshold:
            continue

        x1, y1, x2, y2 = (
            box.detach()
            .cpu()
            .tolist()
        )

        # Clamp coordinates
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        area = (
            (x2 - x1) *
            (y2 - y1)
        )

        if area < min_area:
            continue

        human_boxes.append(
            [x1, y1, x2, y2]
        )

    return human_boxes


# ============================================================
# DETECT HUMANS IN SAMPLED FRAMES
# ============================================================

def detect_sampled_frames(
    frames,
    detector,
    detector_transform,
    device
):
    """
    Detect people only on uniformly sampled frames.

    Returns:

        sampled_results = {
            frame_index: [
                [x1,y1,x2,y2],
                ...
            ]
        }

    This significantly reduces detector computation compared
    with running Faster R-CNN on every frame.
    """

    sample_indices = get_sample_indices(
        len(frames),
        DETECTION_SAMPLE_FRAMES
    )

    sampled_results = {}

    total_persons = 0

    for frame_idx in sample_indices:

        boxes = detect_people(
            frames[frame_idx],
            detector,
            detector_transform,
            device
        )

        sampled_results[frame_idx] = boxes

        total_persons += len(boxes)

    return (
        sampled_results,
        total_persons
    )


# ============================================================
# NEAREST FRAME PROPAGATION
# ============================================================

def propagate_boxes(
    num_frames,
    sampled_results
):
    """
    Propagate the detection from the nearest sampled frame
    to all frames.

    Example:

        sampled frames:
            0, 8, 17, 25

    Frame 7 gets boxes from frame 8.
    Frame 12 gets boxes from frame 8.
    Frame 21 gets boxes from frame 17.

    This is a lightweight approximation intended to avoid
    running Faster R-CNN on every frame.
    """

    if num_frames <= 0:
        return []

    if not sampled_results:
        return [[] for _ in range(num_frames)]

    sampled_indices = sorted(
        sampled_results.keys()
    )

    propagated = []

    for frame_idx in range(num_frames):

        nearest_idx = min(
            sampled_indices,
            key=lambda x: abs(x - frame_idx)
        )

        propagated.append(
            sampled_results[nearest_idx]
        )

    return propagated


# ============================================================
# CREATE HUMAN MASK
# ============================================================

def create_human_mask(
    frame_shape,
    boxes
):
    """
    Create a soft human-region mask.

    Bounding boxes are filled with 1.

    Then Gaussian blur converts the hard mask into a soft
    spatial weighting map.

    Returns:
        float32 mask in [0,1]
    """

    h, w = frame_shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for box in boxes:

        x1, y1, x2, y2 = box

        x1 = max(
            0,
            min(w - 1, int(round(x1)))
        )

        y1 = max(
            0,
            min(h - 1, int(round(y1)))
        )

        x2 = max(
            0,
            min(w, int(round(x2)))
        )

        y2 = max(
            0,
            min(h, int(round(y2)))
        )

        if x2 <= x1 or y2 <= y1:
            continue

        mask[
            y1:y2,
            x1:x2
        ] = 1.0

    # Gaussian smoothing
    kernel = MASK_BLUR_KERNEL

    if kernel % 2 == 0:
        kernel += 1

    if kernel > 1 and np.max(mask) > 0:

        mask = cv2.GaussianBlur(
            mask,
            (kernel, kernel),
            0
        )

        max_value = mask.max()

        if max_value > 0:
            mask = mask / max_value

    return np.clip(
        mask,
        0.0,
        1.0
    )


# ============================================================
# HUMAN REGION ENHANCEMENT
# ============================================================

def enhance_human_regions(
    frame,
    boxes,
    alpha=HUMAN_ALPHA
):
    """
    Enhance human regions without deleting background.

    Formula:

        I' = I * (1 + alpha * M)

    where:

        I = original frame
        M = soft human mask
        alpha = enhancement strength

    With alpha=0.5:

        background:
            approximately unchanged

        human center:
            approximately 1.5x

    Values are clipped to valid uint8 range.
    """

    if frame is None:
        return frame, None

    if len(boxes) == 0:

        # No human detected.
        return (
            frame.copy(),
            np.zeros(
                frame.shape[:2],
                dtype=np.float32
            )
        )

    mask = create_human_mask(
        frame.shape,
        boxes
    )

    frame_float = frame.astype(
        np.float32
    )

    # Convert:
    #
    # mask:
    #   H x W
    #
    # to:
    #   H x W x 1
    mask_3d = mask[..., None]

    enhanced = (
        frame_float *
        (
            1.0 +
            alpha *
            mask_3d
        )
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(
        np.uint8
    )

    return enhanced, mask


# ============================================================
# CREATE HUMAN-ENHANCED VIDEO
# ============================================================

def create_human_enhanced_clip(
    frames,
    propagated_boxes,
    fps
):
    """
    Apply human-aware enhancement to every frame.

    Returns:

        enhanced_frames
        metadata
    """

    enhanced_frames = []

    total_frames = len(frames)
    frames_with_humans = 0
    total_persons = 0
    total_human_area = 0.0

    for frame_idx, frame in enumerate(frames):

        boxes = propagated_boxes[
            frame_idx
        ]

        if len(boxes) > 0:
            frames_with_humans += 1

        total_persons += len(boxes)

        enhanced_frame, mask = (
            enhance_human_regions(
                frame,
                boxes,
                alpha=HUMAN_ALPHA
            )
        )

        if mask is not None:

            human_pixels = np.sum(
                mask > 0.05
            )

            total_human_area += (
                human_pixels /
                (
                    mask.shape[0] *
                    mask.shape[1]
                )
            )

        enhanced_frames.append(
            enhanced_frame
        )

    if total_frames > 0:

        human_frame_ratio = (
            frames_with_humans /
            total_frames
        )

        avg_human_area = (
            total_human_area /
            total_frames
        )

    else:

        human_frame_ratio = 0.0
        avg_human_area = 0.0

    metadata = {
        "total_frames": int(
            total_frames
        ),

        "frames_with_humans": int(
            frames_with_humans
        ),

        "human_frame_ratio": float(
            human_frame_ratio
        ),

        "total_person_detections": int(
            total_persons
        ),

        "average_human_area_ratio": float(
            avg_human_area
        ),

        "detection_sample_frames": int(
            DETECTION_SAMPLE_FRAMES
        ),

        "detection_score_threshold": float(
            DETECTION_SCORE_THRESHOLD
        ),

        "human_alpha": float(
            HUMAN_ALPHA
        ),

        "mask_blur_kernel": int(
            MASK_BLUR_KERNEL
        ),
    }

    return (
        enhanced_frames,
        metadata
    )


# ============================================================
# WRITE TEMPORARY VIDEO
# ============================================================

def write_frames_to_video(
    frames,
    output_path,
    fps
):
    """
    Write enhanced frames to an MP4 file.

    The temporary video is then passed to LanguageBind.
    """

    if not frames:
        raise RuntimeError(
            "No frames available for video writing."
        )

    h, w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (w, h)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter: {output_path}"
        )

    try:

        for frame in frames:
            writer.write(frame)

    finally:

        writer.release()


# ============================================================
# LANGUAGEBIND MODEL
# ============================================================

def load_languagebind(device):
    """
    Load LanguageBind using the same implementation as the
    working lv-mae-2sec preprocessing script.
    """

    print()
    print("=" * 80)
    print("Loading LanguageBind")
    print("=" * 80)

    clip_type = {
        "video": VIDEO_MODEL_NAME
    }

    model = LanguageBind(
        clip_type=clip_type,
        cache_dir="./cache_dir"
    )

    model = model.to(device)

    model.eval()

    modality_transform = {
        c: transform_dict[c](
            model.modality_config[c]
        )
        for c in clip_type.keys()
    }

    print(
        "Video model:",
        VIDEO_MODEL_NAME
    )

    print(
        "Device:",
        device
    )

    return (
        model,
        modality_transform
    )


# ============================================================
# LANGUAGEBIND EMBEDDING
# ============================================================

@torch.no_grad()
def generate_languagebind_embedding(
    clip_path,
    model,
    modality_transform,
    device
):
    """
    Generate one 1024-D LanguageBind embedding.

    This follows the same call pattern as the working
    lv-mae-2sec pipeline.
    """

    inputs = {
        "video": to_device(
            modality_transform["video"](
                [clip_path]
            ),
            device
        )
    }

    outputs = model(inputs)

    embedding = outputs[
        "video"
    ][0].detach().cpu().float()

    embedding = embedding.flatten()

    if embedding.numel() != 1024:

        raise RuntimeError(
            "Unexpected LanguageBind embedding "
            f"dimension: {embedding.shape}. "
            "Expected 1024."
        )

    return embedding


# ============================================================
# SAVE EMBEDDING
# ============================================================

def save_embedding(
    output_dir,
    embedding,
    video_path,
    start_time,
    end_time,
    human_detection
):
    """
    Save LanguageBind embedding and metadata.
    """

    ensure_dir(output_dir)

    output_path = os.path.join(
        output_dir,
        "langbind_output.pt"
    )

    torch.save(
        {
            "embedding": embedding,

            "video_name": os.path.basename(
                video_path
            ),

            "video_path": video_path,

            "start": float(
                start_time
            ),

            "end": float(
                end_time
            ),

            "human_detection": human_detection,
        },
        output_path
    )

    return output_path


# ============================================================
# PROCESS ONE SEGMENT
# ============================================================

def process_segment(
    video_path,
    segment_output_dir,
    start_time,
    end_time,
    detector,
    detector_transform,
    languagebind_model,
    modality_transform,
    device
):
    """
    Process one 2-second segment.

    Steps:

        FFmpeg
          |
          v
        frames
          |
          v
        Faster R-CNN
          |
          v
        human-aware enhancement
          |
          v
        temporary MP4
          |
          v
        LanguageBind
          |
          v
        save 1024-D embedding
    """

    temp_clip = None
    enhanced_clip = None

    segment_duration = (
        end_time -
        start_time
    )

    if segment_duration <= 0:
        raise RuntimeError(
            "Invalid segment duration."
        )

    try:

        # ----------------------------------------------------
        # Original temporary clip
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as tmp:

            temp_clip = tmp.name

        extract_clip(
            input_video=video_path,
            output_video=temp_clip,
            start_time=start_time,
            duration=segment_duration
        )

        # ----------------------------------------------------
        # Read frames
        # ----------------------------------------------------

        frames, fps = read_all_frames(
            temp_clip
        )

        if not frames:
            raise RuntimeError(
                "No frames extracted from clip."
            )

        # ----------------------------------------------------
        # Human detection
        # ----------------------------------------------------

        sampled_results, sampled_persons = (
            detect_sampled_frames(
                frames,
                detector,
                detector_transform,
                device
            )
        )

        # ----------------------------------------------------
        # Propagate detections
        # ----------------------------------------------------

        propagated_boxes = (
            propagate_boxes(
                len(frames),
                sampled_results
            )
        )

        # ----------------------------------------------------
        # Human enhancement
        # ----------------------------------------------------

        (
            enhanced_frames,
            human_metadata
        ) = create_human_enhanced_clip(
            frames,
            propagated_boxes,
            fps
        )

        human_metadata[
            "sampled_person_detections"
        ] = int(sampled_persons)

        human_metadata[
            "sampled_frame_count"
        ] = int(
            len(sampled_results)
        )

        # ----------------------------------------------------
        # Write enhanced temporary clip
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as tmp:

            enhanced_clip = tmp.name

        write_frames_to_video(
            enhanced_frames,
            enhanced_clip,
            fps
        )

        # ----------------------------------------------------
        # LanguageBind
        # ----------------------------------------------------

        embedding = (
            generate_languagebind_embedding(
                enhanced_clip,
                languagebind_model,
                modality_transform,
                device
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = save_embedding(
            output_dir=segment_output_dir,
            embedding=embedding,
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            human_detection=human_metadata
        )

        return {
            "embedding": embedding,
            "output_path": output_path,
            "human_detection": human_metadata,
        }

    finally:

        safe_remove(temp_clip)
        safe_remove(enhanced_clip)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    output_dir,
    detector,
    detector_transform,
    languagebind_model,
    modality_transform,
    device,
    overwrite=False
):
    """
    Process one complete video.

    The video is split into:

        [0,2]
        [2,4]
        [4,6]
        ...

    The last segment can be shorter than 2 seconds.
    """

    start_time_wall = time.time()

    print()
    print("-" * 80)
    print("VIDEO")
    print("-" * 80)
    print(video_path)
    print()
    print("OUTPUT")
    print(output_dir)

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    info = get_video_info(
        video_path
    )

    if info is None:

        raise RuntimeError(
            f"Could not read video information: "
            f"{video_path}"
        )

    duration = info["duration"]
    fps = info["fps"]
    frame_count = info["frame_count"]

    expected_segments = math.ceil(
        duration / SEGMENT_DURATION
    )

    print(
        f"Duration        : {duration:.2f} sec"
    )

    print(
        f"FPS             : {fps:.2f}"
    )

    print(
        f"Frame count     : {frame_count}"
    )

    print(
        f"Expected clips  : {expected_segments}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    ensure_dir(output_dir)

    # --------------------------------------------------------
    # Existing segments
    # --------------------------------------------------------

    existing_segments = set()

    if os.path.isdir(output_dir):

        for segment_name in os.listdir(
            output_dir
        ):

            segment_path = os.path.join(
                output_dir,
                segment_name
            )

            if not os.path.isdir(
                segment_path
            ):
                continue

            embedding_path = os.path.join(
                segment_path,
                "langbind_output.pt"
            )

            if os.path.isfile(
                embedding_path
            ):

                existing_segments.add(
                    segment_name
                )

    print(
        f"Existing clips  : "
        f"{len(existing_segments)}"
    )

    # --------------------------------------------------------
    # Process segments
    # --------------------------------------------------------

    processed = 0
    skipped = 0
    failed = 0

    total_person_detections = 0
    total_human_frames = 0
    total_frames = 0

    start = 0.0

    while start < duration:

        end = min(
            start + SEGMENT_DURATION,
            duration
        )

        # Example:
        #
        # 00000-00002
        # 00002-00004
        # 00004-00006
        #
        segment_folder = (
            f"{int(start):05d}-"
            f"{int(math.ceil(end)):05d}"
        )

        segment_output_dir = os.path.join(
            output_dir,
            segment_folder
        )

        embedding_path = os.path.join(
            segment_output_dir,
            "langbind_output.pt"
        )

        print()
        print(
            f"[Segment {processed + skipped + failed + 1}/"
            f"{expected_segments}] "
            f"{start:.2f}s -> {end:.2f}s"
        )

        # ----------------------------------------------------
        # Skip existing valid segment
        # ----------------------------------------------------

        if (
            os.path.isfile(
                embedding_path
            )
            and not overwrite
        ):

            print(
                "  Already exists -> skip"
            )

            skipped += 1

            start += SEGMENT_DURATION

            continue

        # ----------------------------------------------------
        # Remove incomplete segment
        # ----------------------------------------------------

        if os.path.isdir(
            segment_output_dir
        ):

            try:
                shutil.rmtree(
                    segment_output_dir
                )
            except Exception:
                pass

        try:

            result = process_segment(
                video_path=video_path,
                segment_output_dir=segment_output_dir,
                start_time=start,
                end_time=end,
                detector=detector,
                detector_transform=detector_transform,
                languagebind_model=languagebind_model,
                modality_transform=modality_transform,
                device=device
            )

            metadata = result[
                "human_detection"
            ]

            total_person_detections += (
                metadata.get(
                    "total_person_detections",
                    0
                )
            )

            total_human_frames += (
                metadata.get(
                    "frames_with_humans",
                    0
                )
            )

            total_frames += (
                metadata.get(
                    "total_frames",
                    0
                )
            )

            print(
                "  Embedding      : OK"
            )

            print(
                "  Shape           :",
                tuple(
                    result[
                        "embedding"
                    ].shape
                )
            )

            print(
                "  Human frames    :",
                metadata.get(
                    "frames_with_humans",
                    0
                ),
                "/",
                metadata.get(
                    "total_frames",
                    0
                )
            )

            print(
                "  Persons         :",
                metadata.get(
                    "total_person_detections",
                    0
                )
            )

            processed += 1

        except Exception as e:

            failed += 1

            print()
            print(
                "  ERROR processing segment:"
            )

            print(
                f"  {type(e).__name__}: {e}"
            )

        start += SEGMENT_DURATION

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    elapsed = (
        time.time() -
        start_time_wall
    )

    final_embedding_count = 0

    if os.path.isdir(output_dir):

        for root, dirs, files in os.walk(
            output_dir
        ):

            if (
                "langbind_output.pt"
                in files
            ):
                final_embedding_count += 1

    if total_frames > 0:

        human_frame_ratio = (
            total_human_frames /
            total_frames
        )

    else:

        human_frame_ratio = 0.0

    print()
    print("=" * 80)
    print("VIDEO COMPLETE")
    print("=" * 80)

    print(
        "Video           :",
        os.path.basename(video_path)
    )

    print(
        "Expected clips  :",
        expected_segments
    )

    print(
        "Processed       :",
        processed
    )

    print(
        "Skipped         :",
        skipped
    )

    print(
        "Failed          :",
        failed
    )

    print(
        "Final embeddings:",
        final_embedding_count
    )

    print(
        "Human frame %   :",
        f"{human_frame_ratio * 100:.2f}%"
    )

    print(
        "Person detections:",
        total_person_detections
    )

    print(
        "Time            :",
        format_seconds(elapsed)
    )

    print("=" * 80)

    return {
        "video_path": video_path,
        "expected_segments": expected_segments,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "final_embeddings": final_embedding_count,
        "human_frame_ratio": human_frame_ratio,
        "total_person_detections": total_person_detections,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    video_path_file,
    output_root,
    device,
    overwrite=False
):
    """
    Process all videos listed in the JSON file.
    """

    print()
    print("=" * 80)
    print("HUMAN-AWARE LANGUAGBIND 2-SECOND PREPROCESSING")
    print("=" * 80)

    print(
        "JSON file:",
        video_path_file
    )

    print(
        "Output root:",
        output_root
    )

    print(
        "Device:",
        device
    )

    print(
        "LanguageBind:",
        VIDEO_MODEL_NAME
    )

    print(
        "Segment duration:",
        SEGMENT_DURATION,
        "seconds"
    )

    print(
        "Detection frames/clip:",
        DETECTION_SAMPLE_FRAMES
    )

    print(
        "Detection threshold:",
        DETECTION_SCORE_THRESHOLD
    )

    print(
        "Human alpha:",
        HUMAN_ALPHA
    )

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    data = load_json(
        video_path_file
    )

    video_paths = data.get(
        "file_paths",
        []
    )

    print()
    print(
        "Total videos:",
        len(video_paths)
    )

    if len(video_paths) == 0:

        print(
            "WARNING: JSON contains no videos."
        )

        return

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if device == "cuda":

        if not torch.cuda.is_available():

            print(
                "CUDA requested but unavailable."
            )

            print(
                "Falling back to CPU."
            )

            device = "cpu"

    device_obj = torch.device(
        device
    )

    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------

    detector, detector_transform = (
        load_detector(
            device_obj
        )
    )

    # --------------------------------------------------------
    # Load LanguageBind
    # --------------------------------------------------------

    languagebind_model, modality_transform = (
        load_languagebind(
            device_obj
        )
    )

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    total_videos = len(video_paths)

    successful_videos = 0
    failed_videos = 0

    total_expected_segments = 0
    total_final_embeddings = 0

    dataset_start = time.time()

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for video_index, item in enumerate(
        tqdm(
            video_paths,
            desc="Videos"
        )
    ):

        print()
        print()
        print("#" * 80)

        print(
            f"VIDEO {video_index + 1}/{total_videos}"
        )

        print(
            "#" * 80
        )

        # ----------------------------------------------------
        # JSON format:
        #
        # [
        #     video_path,
        #     output_path
        # ]
        #
        # We intentionally use item[1] directly.
        # This is important for retry JSON files.
        # ----------------------------------------------------

        if not isinstance(
            item,
            (list, tuple)
        ):

            print(
                "Invalid JSON item:",
                item
            )

            failed_videos += 1

            continue

        if len(item) < 2:

            print(
                "Invalid path tuple:",
                item
            )

            failed_videos += 1

            continue

        video_path = item[0]
        output_dir = item[1]

        # ----------------------------------------------------
        # Make sure output is absolute when possible
        # ----------------------------------------------------

        if not os.path.isabs(
            output_dir
        ):

            output_dir = os.path.abspath(
                output_dir
            )

        # ----------------------------------------------------
        # Check video
        # ----------------------------------------------------

        if not os.path.isfile(
            video_path
        ):

            print(
                "Video does not exist:"
            )

            print(
                video_path
            )

            failed_videos += 1

            continue

        try:

            result = process_video(
                video_path=video_path,
                output_dir=output_dir,
                detector=detector,
                detector_transform=detector_transform,
                languagebind_model=languagebind_model,
                modality_transform=modality_transform,
                device=device_obj,
                overwrite=overwrite
            )

            total_expected_segments += (
                result[
                    "expected_segments"
                ]
            )

            total_final_embeddings += (
                result[
                    "final_embeddings"
                ]
            )

            if (
                result["failed"] == 0
                and
                result["final_embeddings"]
                ==
                result["expected_segments"]
            ):

                successful_videos += 1

            else:

                failed_videos += 1

        except KeyboardInterrupt:

            print()
            print(
                "Interrupted by user."
            )

            raise

        except Exception as e:

            print()
            print("=" * 80)
            print("VIDEO FAILED")
            print("=" * 80)

            print(
                "Video:",
                video_path
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            print("=" * 80)

            failed_videos += 1

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    dataset_elapsed = (
        time.time() -
        dataset_start
    )

    print()
    print()
    print("=" * 80)
    print("DATASET PREPROCESSING COMPLETE")
    print("=" * 80)

    print(
        "Total videos             :",
        total_videos
    )

    print(
        "Successful videos        :",
        successful_videos
    )

    print(
        "Failed/incomplete videos :",
        failed_videos
    )

    print(
        "Expected embeddings      :",
        total_expected_segments
    )

    print(
        "Generated embeddings     :",
        total_final_embeddings
    )

    print(
        "Missing embeddings       :",
        max(
            0,
            total_expected_segments -
            total_final_embeddings
        )
    )

    print(
        "Total time               :",
        format_seconds(
            dataset_elapsed
        )
    )

    print(
        "Output root              :",
        output_root
    )

    print("=" * 80)


# ============================================================
# CREATE COLLISION-SAFE JSON
# ============================================================

def generate_collision_safe_json(
    input_json,
    output_json,
    output_root
):
    """
    Utility function to convert an existing path JSON into
    collision-safe output paths.

    This is useful when the original JSON contains paths such as:

        video.MOV
        video.MP4

    """

    data = load_json(
        input_json
    )

    new_items = []

    for item in data.get(
        "file_paths",
        []
    ):

        if len(item) < 1:
            continue

        video_path = item[0]

        output_dir = build_output_dir(
            output_root,
            video_path
        )

        new_items.append(
            [
                video_path,
                output_dir
            ]
        )

    with open(
        output_json,
        "w"
    ) as f:

        json.dump(
            {
                "file_paths": new_items
            },
            f,
            indent=2
        )

    print(
        "Created:",
        output_json
    )

    print(
        "Videos:",
        len(new_items)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Human-aware 2-second "
            "LanguageBind preprocessing"
        )
    )

    # --------------------------------------------------------
    # Input JSON
    # --------------------------------------------------------

    parser.add_argument(
        "--video_path_file",
        type=str,
        default=DEFAULT_VIDEO_PATH_FILE,
        help=(
            "JSON containing "
            "[video_path, output_path] tuples"
        )
    )

    # --------------------------------------------------------
    # Output root
    # --------------------------------------------------------

    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for human-aware "
            "LanguageBind embeddings"
        )
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=[
            "cuda",
            "cpu"
        ],
        help="Inference device"
    )

    # --------------------------------------------------------
    # Overwrite
    # --------------------------------------------------------

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Reprocess segments even if "
            "langbind_output.pt already exists"
        )
    )

    # --------------------------------------------------------
    # Detection settings
    # --------------------------------------------------------

    parser.add_argument(
        "--detection_frames",
        type=int,
        default=DETECTION_SAMPLE_FRAMES,
        help=(
            "Number of frames sampled per "
            "2-second clip for human detection"
        )
    )

    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=DETECTION_SCORE_THRESHOLD,
        help=(
            "Faster R-CNN person confidence threshold"
        )
    )

    parser.add_argument(
        "--human_alpha",
        type=float,
        default=HUMAN_ALPHA,
        help=(
            "Strength of human-region enhancement"
        )
    )

    parser.add_argument(
        "--mask_blur_kernel",
        type=int,
        default=MASK_BLUR_KERNEL,
        help=(
            "Gaussian blur kernel size "
            "for human mask"
        )
    )

    # --------------------------------------------------------
    # Utility: generate collision-safe JSON
    # --------------------------------------------------------

    parser.add_argument(
        "--make_collision_safe_json",
        type=str,
        default=None,
        help=(
            "Instead of processing videos, "
            "create a collision-safe JSON file "
            "at the specified path."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Update global settings from arguments
    # --------------------------------------------------------

    global DETECTION_SAMPLE_FRAMES
    global DETECTION_SCORE_THRESHOLD
    global HUMAN_ALPHA
    global MASK_BLUR_KERNEL

    DETECTION_SAMPLE_FRAMES = max(
        1,
        int(
            args.detection_frames
        )
    )

    DETECTION_SCORE_THRESHOLD = float(
        args.detection_threshold
    )

    HUMAN_ALPHA = float(
        args.human_alpha
    )

    MASK_BLUR_KERNEL = int(
        args.mask_blur_kernel
    )

    if MASK_BLUR_KERNEL < 1:
        MASK_BLUR_KERNEL = 1

    # Gaussian kernel should be odd.
    if MASK_BLUR_KERNEL % 2 == 0:
        MASK_BLUR_KERNEL += 1

    # --------------------------------------------------------
    # Utility mode
    # --------------------------------------------------------

    if args.make_collision_safe_json:

        generate_collision_safe_json(
            input_json=args.video_path_file,
            output_json=args.make_collision_safe_json,
            output_root=args.output_root
        )

        return

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)

    print(
        "Input JSON              :",
        args.video_path_file
    )

    print(
        "Output root             :",
        args.output_root
    )

    print(
        "Device                  :",
        args.device
    )

    print(
        "Segment duration        :",
        SEGMENT_DURATION
    )

    print(
        "Detection frames        :",
        DETECTION_SAMPLE_FRAMES
    )

    print(
        "Detection threshold     :",
        DETECTION_SCORE_THRESHOLD
    )

    print(
        "Human alpha             :",
        HUMAN_ALPHA
    )

    print(
        "Mask blur kernel        :",
        MASK_BLUR_KERNEL
    )

    print(
        "Overwrite               :",
        args.overwrite
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Start dataset processing
    # --------------------------------------------------------

    process_dataset(
        video_path_file=args.video_path_file,
        output_root=args.output_root,
        device=args.device,
        overwrite=args.overwrite
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
