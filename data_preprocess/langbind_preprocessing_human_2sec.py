#!/usr/bin/env python3

# ============================================================
# Optimized Human-Aware LanguageBind 2-Second Preprocessing
# ============================================================
#
# Pipeline:
#
# Original Video
#       |
#       +----> 2-second segment
#                    |
#                    +----> Sample 8 frames
#                    |          |
#                    |          +----> Faster R-CNN
#                    |                    |
#                    |                    +----> Person boxes
#                    |
#                    +----> Nearest-frame propagation
#                    |
#                    +----> Human-region enhancement
#                    |
#                    +----> Temporary human-aware MP4
#                                      |
#                                      +----> LanguageBind
#                                                   |
#                                                   +----> 1024-D embedding
#
# Important optimization:
# - No temporary ORIGINAL MP4 is created.
# - Original video is read directly.
# - Random frame seeking has sequential fallback.
# - Faster R-CNN processes detection frames in small batches.
# - Existing valid embeddings are skipped.
#
# ============================================================


# ============================================================
# ENVIRONMENT / THREAD LIMITS
# ============================================================

import os

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "4"
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "4"
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "4"
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "4"
)

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false"
)


# ============================================================
# IMPORTS
# ============================================================

import sys
import math
import json
import cv2
import torch
import argparse
import tempfile
import numpy as np

from tqdm import tqdm
from PIL import Image


# ============================================================
# THREAD CONTROL
# ============================================================

TORCH_NUM_THREADS = 4
TORCH_INTEROP_THREADS = 1


def configure_threads():

    try:

        torch.set_num_threads(
            TORCH_NUM_THREADS
        )

    except Exception:
        pass

    try:

        torch.set_num_interop_threads(
            TORCH_INTEROP_THREADS
        )

    except Exception:
        pass

    try:

        cv2.setNumThreads(1)

    except Exception:
        pass


# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.realpath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

LV_MAE_2SEC = os.path.join(
    os.path.dirname(PROJECT_ROOT),
    "lv-mae-2sec"
)

if LV_MAE_2SEC not in sys.path:

    sys.path.insert(
        0,
        LV_MAE_2SEC
    )


# ============================================================
# LANGUAGEBIND
# ============================================================

from languagebind_2sec import (
    LanguageBind,
    to_device,
    transform_dict
)


# ============================================================
# FASTER R-CNN
# ============================================================

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of frames sampled from every 2-second segment
DETECTION_FRAMES_PER_CLIP = 8

# Small batch because another process is using the V100
DETECTION_BATCH_SIZE = 2

# Faster R-CNN confidence threshold
PERSON_CONFIDENCE = 0.60

# Human enhancement:
#
# I' = I * (1 + alpha * M)
#
HUMAN_ALPHA = 0.50

# Ignore extremely small person boxes
MIN_BOX_AREA_RATIO = 0.005

# Gaussian smoothing of human mask
MASK_BLUR_KERNEL = 31


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Optimized Human-Aware "
            "2-Second LanguageBind Preprocessing"
        )
    )

    parser.add_argument(
        "--video_path_file",
        type=str,
        required=True,
        help=(
            "JSON file containing "
            "[video_path, output_directory] pairs"
        )
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=(
            "Root directory containing "
            "all output embeddings"
        )
    )

    return parser.parse_args()


# ============================================================
# LOAD FASTER R-CNN
# ============================================================

def load_detector(
    device
):

    print(
        "\nLoading Faster R-CNN..."
    )

    weights = (
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    detector = (
        fasterrcnn_resnet50_fpn(
            weights=weights
        )
    )

    detector = detector.to(
        device
    )

    detector.eval()

    detector_transform = (
        weights.transforms()
    )

    print(
        "Faster R-CNN loaded."
    )

    return (
        detector,
        detector_transform
    )


# ============================================================
# LOAD LANGUAGEBIND
# ============================================================

def load_languagebind(
    device
):

    print(
        "\nLoading LanguageBind..."
    )

    video_model_name = (
        "LanguageBind_Video_Huge_V1.5_FT"
    )

    clip_type = {
        "video": video_model_name
    }

    model = LanguageBind(
        clip_type=clip_type,
        cache_dir="./cache_dir"
    )

    model = model.to(
        device
    )

    model.eval()

    modality_transform = {
        c: transform_dict[c](
            model.modality_config[c]
        )
        for c in clip_type.keys()
    }

    print(
        "LanguageBind loaded."
    )

    return (
        model,
        modality_transform
    )


# ============================================================
# FASTER R-CNN PERSON DETECTION
# ============================================================

@torch.no_grad()
def detect_people_batch(
    detector,
    detector_transform,
    frames,
    device
):
    """
    Detect people in a batch of RGB frames.

    Input:
        frames = list of RGB NumPy arrays

    Output:
        list of person-box lists

    Each box:
        (x1, y1, x2, y2)
    """

    if not frames:

        return []


    # --------------------------------------------------------
    # Convert frames to tensors
    # --------------------------------------------------------

    images = []

    for frame in frames:

        if isinstance(
            frame,
            np.ndarray
        ):

            pil_image = Image.fromarray(
                frame
            )

        elif isinstance(
            frame,
            Image.Image
        ):

            pil_image = frame

        else:

            raise TypeError(
                "Expected NumPy array "
                "or PIL Image"
            )

        image_tensor = (
            detector_transform(
                pil_image
            )
        )

        images.append(
            image_tensor.to(
                device
            )
        )


    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    outputs = detector(
        images
    )


    # --------------------------------------------------------
    # Extract person boxes
    # --------------------------------------------------------

    results = []

    for output in outputs:

        boxes = output[
            "boxes"
        ]

        labels = output[
            "labels"
        ]

        scores = output[
            "scores"
        ]

        person_boxes = []

        for (
            box,
            label,
            score
        ) in zip(
            boxes,
            labels,
            scores
        ):

            # COCO class 1 = person
            if int(label) != 1:

                continue

            if (
                float(score)
                <
                PERSON_CONFIDENCE
            ):

                continue

            (
                x1,
                y1,
                x2,
                y2
            ) = (
                box
                .detach()
                .cpu()
                .numpy()
            )

            person_boxes.append(
                (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                )
            )

        results.append(
            person_boxes
        )

    return results


# ============================================================
# HUMAN MASK
# ============================================================

def create_human_mask(
    frame_shape,
    boxes
):
    """
    Create a soft human-region mask from
    Faster R-CNN person bounding boxes.
    """

    height = frame_shape[0]
    width = frame_shape[1]

    mask = np.zeros(
        (
            height,
            width
        ),
        dtype=np.float32
    )

    image_area = float(
        height * width
    )


    # --------------------------------------------------------
    # Add person boxes
    # --------------------------------------------------------

    for box in boxes:

        (
            x1,
            y1,
            x2,
            y2
        ) = box

        x1 = max(
            0,
            min(
                width - 1,
                int(x1)
            )
        )

        x2 = max(
            0,
            min(
                width,
                int(x2)
            )
        )

        y1 = max(
            0,
            min(
                height - 1,
                int(y1)
            )
        )

        y2 = max(
            0,
            min(
                height,
                int(y2)
            )
        )

        if (
            x2 <= x1
            or
            y2 <= y1
        ):

            continue


        box_area = float(
            (x2 - x1)
            *
            (y2 - y1)
        )

        area_ratio = (
            box_area
            /
            image_area
        )

        if (
            area_ratio
            <
            MIN_BOX_AREA_RATIO
        ):

            continue

        mask[
            y1:y2,
            x1:x2
        ] = 1.0


    # --------------------------------------------------------
    # Smooth mask boundaries
    # --------------------------------------------------------

    if (
        MASK_BLUR_KERNEL
        >
        1
    ):

        kernel = (
            MASK_BLUR_KERNEL
            if MASK_BLUR_KERNEL % 2 == 1
            else MASK_BLUR_KERNEL + 1
        )

        mask = cv2.GaussianBlur(
            mask,
            (
                kernel,
                kernel
            ),
            0
        )

    return mask


# ============================================================
# HUMAN REGION ENHANCEMENT
# ============================================================

def enhance_human_regions(
    frame,
    mask
):
    """
    Softly enhance detected human regions.

    I' = I * (1 + alpha * M)
    """

    frame_float = (
        frame.astype(
            np.float32
        )
        /
        255.0
    )

    mask_3d = np.expand_dims(
        mask,
        axis=2
    )

    enhanced = (
        frame_float
        *
        (
            1.0
            +
            HUMAN_ALPHA
            *
            mask_3d
        )
    )

    enhanced = np.clip(
        enhanced,
        0.0,
        1.0
    )

    enhanced = (
        enhanced
        *
        255.0
    ).astype(
        np.uint8
    )

    return enhanced


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()


    # --------------------------------------------------------
    # Fallback FPS
    # --------------------------------------------------------

    if fps <= 0:

        fps = 25.0


    duration = (
        frame_count
        /
        fps
    )

    return (
        fps,
        frame_count,
        width,
        height,
        duration
    )


# ============================================================
# ROBUST VIDEO OPENING AT FRAME
# ============================================================

def open_video_at_frame(
    video_path,
    start_frame
):
    """
    Try random frame seeking first.

    If OpenCV cannot read after the random seek,
    reopen the video and sequentially decode until
    start_frame.

    This fixes videos/codecs where:

        cap.set(CAP_PROP_POS_FRAMES, X)

    reports a position but the next read fails.
    """

    # ========================================================
    # ATTEMPT 1
    # Random seek
    # ========================================================

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot open video: "
            f"{video_path}"
        )

    if start_frame <= 0:

        return cap


    # --------------------------------------------------------
    # Try seeking
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )


    # --------------------------------------------------------
    # Test read
    # --------------------------------------------------------

    test_ret, _ = cap.read()

    if test_ret:

        # Seek again to make sure the caller
        # starts exactly from the requested region.
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        verify_ret, _ = cap.read()

        if verify_ret:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_frame
            )

            print(
                "    Random seek successful "
                f"at frame {start_frame}"
            )

            return cap


    # ========================================================
    # ATTEMPT 2
    # Sequential fallback
    # ========================================================

    cap.release()

    print(
        "    Random seek failed at frame "
        f"{start_frame}; "
        "using sequential fallback..."
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot reopen video: "
            f"{video_path}"
        )

    current_frame = 0

    while (
        current_frame
        <
        start_frame
    ):

        ret, _ = cap.read()

        if not ret:

            cap.release()

            raise RuntimeError(
                "Cannot sequentially reach "
                f"frame {start_frame}. "
                f"Stopped at frame "
                f"{current_frame}: "
                f"{video_path}"
            )

        current_frame += 1

    print(
        "    Sequential fallback reached "
        f"frame {current_frame}"
    )

    return cap


# ============================================================
# HUMAN-AWARE CLIP CREATION
# ============================================================

def create_human_aware_clip(
    input_video,
    output_clip,
    start_time,
    duration,
    detector,
    detector_transform,
    device
):
    """
    Create one human-aware segment directly
    from the original video.

    No intermediate original temporary video.
    """

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    (
        fps,
        source_frame_count,
        width,
        height,
        source_duration
    ) = get_video_info(
        input_video
    )


    # ========================================================
    # FRAME RANGE
    # ========================================================

    start_frame = int(
        round(
            start_time
            *
            fps
        )
    )

    max_frames = max(
        1,
        int(
            math.ceil(
                duration
                *
                fps
            )
        )
    )

    end_frame_exclusive = min(
        start_frame
        +
        max_frames,
        source_frame_count
    )

    if (
        start_frame
        >=
        source_frame_count
    ):

        raise RuntimeError(
            "Segment starts beyond video: "
            f"{input_video}"
        )

    segment_frame_count = (
        end_frame_exclusive
        -
        start_frame
    )


    # ========================================================
    # SELECT DETECTION FRAMES
    # ========================================================

    num_detection_frames = min(
        DETECTION_FRAMES_PER_CLIP,
        segment_frame_count
    )

    detection_relative_indices = (
        np.linspace(
            0,
            segment_frame_count - 1,
            num=num_detection_frames,
            dtype=int
        )
    )

    detection_relative_indices = sorted(
        set(
            detection_relative_indices.tolist()
        )
    )

    detection_set = set(
        detection_relative_indices
    )


    # ========================================================
    # FIRST PASS
    # READ SAMPLED FRAMES
    # ========================================================

    cap = open_video_at_frame(
        input_video,
        start_frame
    )

    sampled_frames = []
    sampled_indices = []

    relative_index = 0

    while (
        relative_index
        <
        segment_frame_count
    ):

        ret, frame = cap.read()

        if not ret:

            break

        if (
            relative_index
            in
            detection_set
        ):

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            sampled_frames.append(
                rgb_frame
            )

            sampled_indices.append(
                relative_index
            )

        relative_index += 1

    cap.release()


    # ========================================================
    # VALIDATION
    # ========================================================

    if not sampled_frames:

        raise RuntimeError(
            "No frames could be read from "
            f"segment: {input_video}"
        )


    # ========================================================
    # FASTER R-CNN DETECTION
    # ========================================================

    sampled_detections = {}

    total_persons = 0

    for batch_start in range(
        0,
        len(sampled_frames),
        DETECTION_BATCH_SIZE
    ):

        batch_frames = (
            sampled_frames[
                batch_start:
                batch_start
                +
                DETECTION_BATCH_SIZE
            ]
        )

        batch_indices = (
            sampled_indices[
                batch_start:
                batch_start
                +
                DETECTION_BATCH_SIZE
            ]
        )


        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        batch_boxes = (
            detect_people_batch(
                detector=detector,
                detector_transform=(
                    detector_transform
                ),
                frames=batch_frames,
                device=device
            )
        )


        # ----------------------------------------------------
        # Store detections
        # ----------------------------------------------------

        for (
            frame_idx,
            boxes
        ) in zip(
            batch_indices,
            batch_boxes
        ):

            sampled_detections[
                frame_idx
            ] = boxes

            total_persons += len(
                boxes
            )


    # ========================================================
    # SECOND PASS
    # CREATE HUMAN-AWARE VIDEO
    # ========================================================

    cap = open_video_at_frame(
        input_video,
        start_frame
    )


    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_clip,
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Cannot create output clip: "
            f"{output_clip}"
        )


    # ========================================================
    # DETECTION INDEX
    # ========================================================

    detection_keys = sorted(
        sampled_detections.keys()
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    total_frames = 0

    frames_with_humans = 0


    # ========================================================
    # ENHANCEMENT PASS
    # ========================================================

    while (
        total_frames
        <
        segment_frame_count
    ):

        ret, frame = cap.read()

        if not ret:

            break


        # ----------------------------------------------------
        # Find nearest sampled detection
        # ----------------------------------------------------

        if detection_keys:

            nearest_key = min(
                detection_keys,
                key=lambda x:
                    abs(
                        x
                        -
                        total_frames
                    )
            )

            boxes = (
                sampled_detections[
                    nearest_key
                ]
            )

        else:

            boxes = []


        # ----------------------------------------------------
        # Human-frame statistics
        # ----------------------------------------------------

        if len(boxes) > 0:

            frames_with_humans += 1


        # ----------------------------------------------------
        # Human mask
        # ----------------------------------------------------

        mask = create_human_mask(
            frame.shape,
            boxes
        )


        # ----------------------------------------------------
        # Enhancement
        # ----------------------------------------------------

        enhanced_frame = (
            enhance_human_regions(
                frame,
                mask
            )
        )


        # ----------------------------------------------------
        # Write frame
        # ----------------------------------------------------

        writer.write(
            enhanced_frame
        )

        total_frames += 1


    # ========================================================
    # CLOSE VIDEO
    # ========================================================

    cap.release()
    writer.release()


    # ========================================================
    # VALIDATION
    # ========================================================

    if total_frames == 0:

        raise RuntimeError(
            "No frames written to: "
            f"{output_clip}"
        )


    if not os.path.isfile(
        output_clip
    ):

        raise RuntimeError(
            "Output clip was not created: "
            f"{output_clip}"
        )


    output_size = os.path.getsize(
        output_clip
    )

    if output_size <= 0:

        raise RuntimeError(
            "Output clip is empty: "
            f"{output_clip}"
        )


    return {

        "total_frames":
            total_frames,

        "frames_with_humans":
            frames_with_humans,

        "total_persons":
            total_persons,

        "sampled_detection_frames":
            len(
                detection_keys
            )
    }


# ============================================================
# LANGUAGEBIND EMBEDDING
# ============================================================

@torch.no_grad()
def get_languagebind_embedding(
    model,
    modality_transform,
    clip_path,
    device
):
    """
    Generate a 1024-D LanguageBind video embedding.
    """

    inputs = {

        "video":
            to_device(
                modality_transform[
                    "video"
                ](
                    [clip_path]
                ),
                device
            )
    }


    output = model(
        inputs
    )


    embedding = (
        output[
            "video"
        ][0]
        .cpu()
        .float()
        .flatten()
    )


    # ========================================================
    # DIMENSION CHECK
    # ========================================================

    if embedding.numel() != 1024:

        raise RuntimeError(
            "Unexpected LanguageBind "
            f"embedding dimension: "
            f"{embedding.numel()}. "
            "Expected 1024."
        )


    return embedding


# ============================================================
# TEMP FILE CREATION
# ============================================================

def create_temp_mp4():

    """
    Create a unique temporary filename.

    The file itself is removed immediately so that
    OpenCV can create it cleanly.
    """

    fd, path = tempfile.mkstemp(
        suffix=".mp4"
    )

    os.close(fd)

    try:

        os.remove(
            path
        )

    except FileNotFoundError:

        pass

    return path


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    output_dir,
    detector,
    detector_transform,
    languagebind,
    modality_transform,
    device
):

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    (
        fps,
        frame_count,
        width,
        height,
        duration
    ) = get_video_info(
        video_path
    )


    # ========================================================
    # NUMBER OF 2-SECOND SEGMENTS
    # ========================================================

    num_segments = int(
        math.ceil(
            duration
            /
            2.0
        )
    )


    # ========================================================
    # VIDEO NAME
    # ========================================================

    video_name = os.path.splitext(
        os.path.basename(
            video_path
        )
    )[0]


    # ========================================================
    # STATISTICS
    # ========================================================

    stats = {

        "video_name":
            video_name,

        "segments":
            0,

        "total_frames":
            0,

        "frames_with_humans":
            0,

        "total_persons":
            0
    }


    # ========================================================
    # SEGMENTS
    # ========================================================

    for segment_idx in range(
        num_segments
    ):

        start = (
            segment_idx
            *
            2.0
        )

        end = min(
            start + 2.0,
            duration
        )


        # ----------------------------------------------------
        # Folder name
        # ----------------------------------------------------

        segment_folder = os.path.join(
            output_dir,
            (
                f"{int(start):05d}-"
                f"{int(math.ceil(end)):05d}"
            )
        )

        os.makedirs(
            segment_folder,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Embedding path
        # ----------------------------------------------------

        embedding_path = os.path.join(
            segment_folder,
            "langbind_output.pt"
        )


        # ====================================================
        # CHECK EXISTING EMBEDDING
        # ====================================================

        if os.path.isfile(
            embedding_path
        ):

            try:

                existing = torch.load(
                    embedding_path,
                    map_location="cpu"
                )


                if (
                    isinstance(
                        existing,
                        dict
                    )
                    and
                    "embedding"
                    in
                    existing
                ):

                    existing_embedding = (
                        existing[
                            "embedding"
                        ]
                    )

                    if (
                        torch.is_tensor(
                            existing_embedding
                        )
                        and
                        existing_embedding.numel()
                        ==
                        1024
                    ):

                        print(
                            f"\n    SKIP existing: "
                            f"{segment_folder}"
                        )

                        stats[
                            "segments"
                        ] += 1

                        continue

            except Exception as e:

                print(
                    "\n    Existing embedding "
                    "could not be validated:"
                )

                print(
                    f"    {embedding_path}"
                )

                print(
                    f"    Reprocessing. "
                    f"Reason: {e}"
                )


        # ====================================================
        # TEMPORARY HUMAN-AWARE VIDEO
        # ====================================================

        temp_human = (
            create_temp_mp4()
        )


        try:

            print(
                f"\n    Segment "
                f"{segment_idx + 1}/"
                f"{num_segments}: "
                f"{start:.2f}s - "
                f"{end:.2f}s"
            )


            # ------------------------------------------------
            # Human detection + enhancement
            # ------------------------------------------------

            human_stats = (
                create_human_aware_clip(
                    input_video=video_path,
                    output_clip=temp_human,
                    start_time=start,
                    duration=end - start,
                    detector=detector,
                    detector_transform=(
                        detector_transform
                    ),
                    device=device
                )
            )


            # ------------------------------------------------
            # LanguageBind
            # ------------------------------------------------

            print(
                "    Generating "
                "LanguageBind embedding..."
            )

            embedding = (
                get_languagebind_embedding(
                    model=languagebind,
                    modality_transform=(
                        modality_transform
                    ),
                    clip_path=temp_human,
                    device=device
                )
            )


            # =================================================
            # HUMAN STATISTICS
            # =================================================

            segment_total_frames = (
                human_stats[
                    "total_frames"
                ]
            )

            segment_human_frames = (
                human_stats[
                    "frames_with_humans"
                ]
            )

            human_ratio = (
                segment_human_frames
                /
                max(
                    segment_total_frames,
                    1
                )
            )


            # =================================================
            # METADATA
            # =================================================

            human_detection = {

                "total_frames":
                    segment_total_frames,

                "frames_with_humans":
                    segment_human_frames,

                "human_frame_ratio":
                    human_ratio,

                "total_detected_persons":
                    human_stats[
                        "total_persons"
                    ],

                "sampled_detection_frames":
                    human_stats[
                        "sampled_detection_frames"
                    ],

                "person_confidence":
                    PERSON_CONFIDENCE,

                "human_alpha":
                    HUMAN_ALPHA,

                "min_box_area_ratio":
                    MIN_BOX_AREA_RATIO,

                "detection_method":
                    "Faster R-CNN",

                "detection_batch_size":
                    DETECTION_BATCH_SIZE,

                "propagation_method":
                    "nearest sampled frame",

                "embedding_model":
                    "LanguageBind_Video_Huge_V1.5_FT",

                "preprocessing_version":
                    "human-aware-2sec-optimized"
            }


            # =================================================
            # SAVE EMBEDDING
            # =================================================

            torch.save(
                {

                    "embedding":
                        embedding,

                    "video_name":
                        os.path.basename(
                            video_path
                        ),

                    "start":
                        start,

                    "end":
                        end,

                    "human_detection":
                        human_detection

                },
                embedding_path
            )


            # =================================================
            # UPDATE STATISTICS
            # =================================================

            stats[
                "segments"
            ] += 1

            stats[
                "total_frames"
            ] += segment_total_frames

            stats[
                "frames_with_humans"
            ] += segment_human_frames

            stats[
                "total_persons"
            ] += (
                human_stats[
                    "total_persons"
                ]
            )


            print(
                "    Saved:"
                f" {embedding_path}"
            )

            print(
                "    Embedding shape:"
                f" {tuple(embedding.shape)}"
            )

            print(
                "    Human-frame ratio:"
                f" {human_ratio * 100:.2f}%"
            )


        finally:

            # ------------------------------------------------
            # Delete temporary human video
            # ------------------------------------------------

            if os.path.exists(
                temp_human
            ):

                try:

                    os.remove(
                        temp_human
                    )

                except Exception:
                    pass


    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # THREAD CONFIGURATION
    # ========================================================

    configure_threads()


    # ========================================================
    # ARGUMENTS
    # ========================================================

    args = parse_args()


    # ========================================================
    # HEADER
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "OPTIMIZED HUMAN-AWARE "
        "LANGUAGEBIND PREPROCESSING"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )


    # ========================================================
    # CONFIGURATION DISPLAY
    # ========================================================

    print(
        f"Person confidence: "
        f"{PERSON_CONFIDENCE}"
    )

    print(
        f"Human alpha: "
        f"{HUMAN_ALPHA}"
    )

    print(
        f"Minimum box area ratio: "
        f"{MIN_BOX_AREA_RATIO}"
    )

    print(
        f"Detection frames per clip: "
        f"{DETECTION_FRAMES_PER_CLIP}"
    )

    print(
        f"Detection batch size: "
        f"{DETECTION_BATCH_SIZE}"
    )

    print(
        f"Torch CPU threads: "
        f"{TORCH_NUM_THREADS}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # LOAD JSON
    # ========================================================

    with open(
        args.video_path_file,
        "r"
    ) as f:

        data = json.load(
            f
        )


    if (
        not isinstance(
            data,
            dict
        )
        or
        "file_paths"
        not in
        data
    ):

        raise ValueError(
            "JSON must contain "
            "'file_paths'"
        )


    video_paths = data[
        "file_paths"
    ]


    print(
        f"\nVideos to process: "
        f"{len(video_paths)}"
    )

    print(
        f"Output root: "
        f"{args.output_root}"
    )


    # ========================================================
    # OUTPUT ROOT
    # ========================================================

    output_root_abs = os.path.abspath(
        args.output_root
    )


    # ========================================================
    # LOAD MODELS
    # ========================================================

    detector, detector_transform = (
        load_detector(
            device
        )
    )

    languagebind, modality_transform = (
        load_languagebind(
            device
        )
    )


    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    total_videos = 0

    total_segments = 0

    total_frames = 0

    total_human_frames = 0

    total_persons = 0


    # ========================================================
    # PROCESS VIDEOS
    # ========================================================

    for item in tqdm(
        video_paths,
        desc="Videos"
    ):

        # ----------------------------------------------------
        # Validate JSON item
        # ----------------------------------------------------

        if not isinstance(
            item,
            (list, tuple)
        ):

            print(
                "\nInvalid JSON item:",
                item
            )

            continue


        if len(item) < 2:

            print(
                "\nIncomplete JSON item:",
                item
            )

            continue


        # ----------------------------------------------------
        # JSON format:
        #
        # [
        #     input_video,
        #     output_directory
        # ]
        # ----------------------------------------------------

        video_path = item[
            0
        ]

        output_dir = item[
            1
        ]


        # ====================================================
        # INPUT CHECK
        # ====================================================

        if not os.path.isfile(
            video_path
        ):

            print(
                "\nWARNING: Video not found:"
            )

            print(
                video_path
            )

            continue


        # ====================================================
        # OUTPUT ROOT SAFETY CHECK
        # ====================================================

        output_dir_abs = os.path.abspath(
            output_dir
        )


        if not (
            output_dir_abs
            ==
            output_root_abs
            or
            output_dir_abs.startswith(
                output_root_abs
                +
                os.sep
            )
        ):

            raise RuntimeError(
                "\nOutput directory is "
                "outside the specified "
                "output root.\n"
                f"Root: {output_root_abs}\n"
                f"Output: {output_dir_abs}"
            )


        # ====================================================
        # PROCESS VIDEO
        # ====================================================

        try:

            video_stats = (
                process_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    detector=detector,
                    detector_transform=(
                        detector_transform
                    ),
                    languagebind=languagebind,
                    modality_transform=(
                        modality_transform
                    ),
                    device=device
                )
            )


            total_videos += 1


            total_segments += (
                video_stats[
                    "segments"
                ]
            )


            total_frames += (
                video_stats[
                    "total_frames"
                ]
            )


            total_human_frames += (
                video_stats[
                    "frames_with_humans"
                ]
            )


            total_persons += (
                video_stats[
                    "total_persons"
                ]
            )


        except Exception as e:

            print(
                "\nERROR processing:"
            )

            print(
                video_path
            )

            print(
                f"Error: {e}"
            )

            continue


    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    human_ratio = (
        total_human_frames
        /
        max(
            total_frames,
            1
        )
    )


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "HUMAN-AWARE PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"Videos processed: "
        f"{total_videos}"
    )

    print(
        f"Segments processed: "
        f"{total_segments}"
    )

    print(
        f"Total frames: "
        f"{total_frames}"
    )

    print(
        f"Frames containing humans: "
        f"{total_human_frames}"
    )

    print(
        f"Human-frame ratio: "
        f"{human_ratio * 100:.2f}%"
    )

    print(
        f"Total detected persons: "
        f"{total_persons}"
    )

    print(
        f"Output: "
        f"{args.output_root}"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
