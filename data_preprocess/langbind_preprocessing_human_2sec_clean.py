import os

# ---------------------------------------------------------------------
# Thread limits -- must be set before NumPy/PyTorch/OpenCV import
# ---------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import math
import json
import time
import shutil
import tempfile
import subprocess
import argparse
import traceback
import gc

import numpy as np
import torch
import cv2
import psutil

PROCESS = psutil.Process(os.getpid())

def print_memory(tag):
    rss_gb = PROCESS.memory_info().rss / (1024 ** 3)
    print(f"[RAM] {tag}: {rss_gb:.2f} GB", flush=True)

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------
# Runtime thread limits
# ---------------------------------------------------------------------
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)

from languagebind_2sec import LanguageBind, to_device, transform_dict

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"

DATASET_ROOT = (
    "/home/y_rajeshwar/final_year_project_29/dataset_new_split_clean"
)

OUTPUT_ROOT = (
    "/home/y_rajeshwar/final_year_project_29/"
    "langbind_features_human_clean"
)

LANGUAGEBIND_MODEL = "LanguageBind_Video_Huge_V1.5_FT"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERSON_CLASS_ID = 1
CONFIDENCE_THRESHOLD = 0.60
MIN_BOX_AREA_RATIO = 0.005

NUM_TEMPORAL_SAMPLES = 8
BLUR_KERNEL = 31
ALPHA = 0.50

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def run_command(cmd):
    """
    Run a normal text-producing command.
    Binary FFmpeg output must NOT use this helper.
    """
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(str(x) for x in cmd)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr
        )

    return result


def get_video_duration(video_path):
    """
    Get duration using FFprobe, not OpenCV.
    """
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    result = run_command(cmd)

    try:
        duration = float(result.stdout.strip())
    except Exception:
        raise RuntimeError(
            f"Could not parse FFprobe duration for {video_path}\n"
            f"FFprobe output: {result.stdout}"
        )

    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            f"Invalid duration {duration} for {video_path}"
        )

    return duration


def get_video_dimensions(video_path):
    """
    Get source video dimensions using FFprobe.
    """
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        video_path,
    ]

    result = run_command(cmd)

    value = result.stdout.strip()

    try:
        width, height = value.split("x")
        return int(width), int(height)
    except Exception:
        raise RuntimeError(
            f"Could not parse video dimensions for {video_path}: "
            f"{value}"
        )


def extract_segment_frames(video_path, start_time, end_time):
    """
    Decode one segment using FFmpeg and read raw RGB frames
    incrementally from stdout.

    Frames are resized by FFmpeg before being transferred to Python,
    preventing subprocess.run() from accumulating the entire video
    in RAM.
    """
    duration = end_time - start_time

    target_width = 640

    source_width, source_height = get_video_dimensions(video_path)

    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(
            f"Invalid source dimensions: "
            f"{source_width}x{source_height}"
        )

    target_height = int(
        round(source_height * target_width / source_width)
    )

    if target_height % 2 != 0:
        target_height -= 1

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_time),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"scale={target_width}:{target_height}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    frame_size = target_width * target_height * 3

    frames = []

    try:
        while True:
            raw_frame = process.stdout.read(frame_size)

            if not raw_frame:
                break

            if len(raw_frame) != frame_size:
                raise RuntimeError(
                    f"Incomplete frame received from FFmpeg: "
                    f"{len(raw_frame)} bytes, expected {frame_size}"
                )

            frame = np.frombuffer(
                raw_frame,
                dtype=np.uint8,
            ).reshape(
                target_height,
                target_width,
                3,
            ).copy()

            frames.append(frame)

        stderr = process.stderr.read()
        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                "FFmpeg frame extraction failed:\n"
                + " ".join(str(x) for x in cmd)
                + "\n\nSTDERR:\n"
                + stderr.decode(
                    "utf-8",
                    errors="replace",
                )
            )

    finally:
        if process.stdout is not None:
            process.stdout.close()

        if process.stderr is not None:
            process.stderr.close()

    if not frames:
        raise RuntimeError(
            f"FFmpeg returned zero frames for "
            f"{video_path}, {start_time}-{end_time}"
        )

    return frames

def sample_indices(number_of_frames, number_of_samples):
    """
    Select approximately evenly distributed frame indices.
    """
    if number_of_frames <= 0:
        return []

    if number_of_frames <= number_of_samples:
        return list(range(number_of_frames))

    values = np.linspace(
        0,
        number_of_frames - 1,
        number_of_samples,
    )

    return sorted(set(int(round(x)) for x in values))


# ---------------------------------------------------------------------
# Faster R-CNN
# ---------------------------------------------------------------------
def build_detector():
    """
    Load COCO-pretrained Faster R-CNN ResNet50 FPN.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    # Keep detector on CPU initially.
    # It will be moved to GPU only during human detection.
    model = model.to("cpu")
    model.eval()

    detector_transform = weights.transforms()

    return model, detector_transform


def detect_human_masks(
    frames,
    detector,
    detector_transform,
):
    """
    Run Faster R-CNN on approximately 8 sampled frames.

    Returns:
        sampled_indices
        sampled_masks
        metadata
    """
    n_frames = len(frames)

    indices = sample_indices(
        n_frames,
        NUM_TEMPORAL_SAMPLES,
    )

    sampled_masks = []

    total_detected_persons = 0
    total_human_frames = 0
    all_confidences = []

    # Move detector to GPU only for the detection phase.
    detector = detector.to(DEVICE)
    detector.eval()

    height, width = frames[0].shape[:2]
    frame_area = float(width * height)

    for idx in indices:
        frame = frames[idx]

        image = Image.fromarray(frame)

        tensor = detector_transform(image)

        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            output = detector([tensor])[0]

        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()

        mask = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        frame_person_count = 0

        for box, label, score in zip(
            boxes,
            labels,
            scores,
        ):
            if int(label) != PERSON_CLASS_ID:
                continue

            score = float(score)

            if score < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = box

            x1 = max(0, min(width, int(round(x1))))
            x2 = max(0, min(width, int(round(x2))))
            y1 = max(0, min(height, int(round(y1))))
            y2 = max(0, min(height, int(round(y2))))

            box_width = max(0, x2 - x1)
            box_height = max(0, y2 - y1)

            box_area = float(box_width * box_height)
            area_ratio = box_area / frame_area

            if area_ratio < MIN_BOX_AREA_RATIO:
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            mask[y1:y2, x1:x2] = 1.0

            frame_person_count += 1
            total_detected_persons += 1
            all_confidences.append(score)

        if frame_person_count > 0:
            total_human_frames += 1

        # Required Gaussian blur kernel = 31
        mask = cv2.GaussianBlur(
            mask,
            (BLUR_KERNEL, BLUR_KERNEL),
            0,
        )

        # Keep mask in [0, 1]
        mask = np.clip(mask, 0.0, 1.0)

        sampled_masks.append(mask)

    metadata = {
        "sampled_frame_count": len(indices),
        "sampled_indices": indices,
        "human_frames": total_human_frames,
        "detected_persons": total_detected_persons,
        "mean_confidence": (
            float(np.mean(all_confidences))
            if all_confidences
            else 0.0
        ),
        "max_confidence": (
            float(np.max(all_confidences))
            if all_confidences
            else 0.0
        ),
    }

    # Release detector GPU memory before LanguageBind inference.
    if DEVICE == "cuda":
        detector = detector.to("cpu")
        torch.cuda.empty_cache()

    return indices, sampled_masks, metadata


# ---------------------------------------------------------------------
# Human-aware enhancement
# ---------------------------------------------------------------------
def nearest_mask_for_frame(
    frame_index,
    sampled_indices,
    sampled_masks,
):
    """
    Temporal nearest-neighbor propagation of sampled human masks.
    """
    distances = [
        abs(frame_index - sample_index)
        for sample_index in sampled_indices
    ]

    nearest = int(np.argmin(distances))

    return sampled_masks[nearest]


def enhance_frames(
    frames,
    sampled_indices,
    sampled_masks,
):
    """
    Apply:
        I' = I * (1 + alpha * M)

    Background is retained. Humans are softly enhanced.
    """
    enhanced = []

    for frame_index, frame in enumerate(frames):

        mask = nearest_mask_for_frame(
            frame_index,
            sampled_indices,
            sampled_masks,
        )

        frame_float = frame.astype(
            np.float32
        )

        multiplier = (
            1.0 + ALPHA * mask[..., None]
        )

        output = frame_float * multiplier

        output = np.clip(
            output,
            0,
            255,
        ).astype(np.uint8)

        enhanced.append(output)

    return enhanced


# ---------------------------------------------------------------------
# Temporary H.264 reconstruction
# ---------------------------------------------------------------------
def write_h264_video(
    frames,
    output_path,
    source_video,
    start_time,
    end_time,
):
    """
    Reconstruct enhanced RGB frames as a temporary H.264 video.

    FFmpeg receives raw RGB bytes through stdin.  stdin is explicitly
    closed before waiting for FFmpeg so communicate() is not used with
    an already-closed stdin pipe.
    """
    cmd_fps = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source_video,
    ]

    result = run_command(cmd_fps)
    fps_text = result.stdout.strip()

    try:
        if "/" in fps_text:
            numerator, denominator = fps_text.split("/")
            fps = float(numerator) / float(denominator)
        else:
            fps = float(fps_text)
    except Exception:
        fps = 30.0

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    height, width = frames[0].shape[:2]

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        output_path,
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        for frame in frames:
            process.stdin.write(
                np.asarray(frame, dtype=np.uint8).tobytes()
            )

        process.stdin.close()
        process.stdin = None

        stdout, stderr = process.communicate()

    except Exception as exc:
        try:
            if process.stdin is not None:
                process.stdin.close()
        except Exception:
            pass

        try:
            process.kill()
        except Exception:
            pass

        stdout, stderr = process.communicate()

        raise RuntimeError(
            "Failed while writing temporary H.264 video.\n"
            + str(exc)
            + "\nFFmpeg stderr:\n"
            + stderr.decode(errors="replace")
        )

    if process.returncode != 0:
        raise RuntimeError(
            "FFmpeg H.264 reconstruction failed.\n"
            + stderr.decode(errors="replace")
        )

    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"Temporary video was not created: {output_path}"
        )

    if os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"Temporary video is empty: {output_path}"
        )


# ---------------------------------------------------------------------
# Validate temporary video
# ---------------------------------------------------------------------
def validate_video_with_ffmpeg(video_path):
    """
    Confirm the reconstructed H.264 video is readable.
    """
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-i",
        video_path,
        "-f",
        "null",
        "-",
    ]

    run_command(cmd)


# ---------------------------------------------------------------------
# LanguageBind
# ---------------------------------------------------------------------
def build_languagebind():
    """
    Reuse the existing working LanguageBind implementation.
    """
    clip_type = {
        "video": LANGUAGEBIND_MODEL
    }

    model = LanguageBind(
        clip_type=clip_type,
        cache_dir="./cache_dir",
    )

    model = model.to(DEVICE)
    model.eval()

    modality_transform = {
        c: transform_dict[c](
            model.modality_config[c]
        )
        for c in clip_type.keys()
    }

    return model, modality_transform


def get_languagebind_embedding(
    model,
    modality_transform,
    video_path,
):
    """
    Generate one 1024-D LanguageBind video embedding.
    """
    inputs = {
        "video": to_device(
            modality_transform["video"](
                [video_path]
            ),
            DEVICE,
        )
    }

    with torch.inference_mode():
        outputs = model(inputs)

    embedding = outputs["video"][0].detach().cpu()

    if embedding.numel() != 1024:
        del outputs
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

        raise RuntimeError(
            f"Expected 1024-D embedding, got "
            f"shape={tuple(embedding.shape)}"
        )

    if not torch.isfinite(embedding).all():
        del outputs
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

        raise RuntimeError(
            "LanguageBind embedding contains NaN or Inf"
        )

    embedding = embedding.reshape(1024)

    # Release LanguageBind intermediate tensors.
    del outputs
    del inputs

    gc.collect()
    torch.cuda.empty_cache()

    return embedding


# ---------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------
def output_video_name(video_path):
    """
    Preserve extension information to avoid collisions.

    Example:
        abc.MOV -> abc_MOV
        abc.MP4 -> abc_MP4
    """
    filename = os.path.basename(video_path)

    stem, extension = os.path.splitext(filename)

    if extension:
        return (
            stem
            + "_"
            + extension.lstrip(".").upper()
        )

    return stem


def segment_name(start_time, end_time):
    return (
        f"{int(start_time):05d}-"
        f"{int(math.ceil(end_time)):05d}"
    )


# ---------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------
def process_segment(
    video_path,
    output_video_root,
    start_time,
    end_time,
    detector,
    detector_transform,
    languagebind,
    modality_transform,
):
    segment_folder = os.path.join(
        output_video_root,
        segment_name(start_time, end_time),
    )

    os.makedirs(
        segment_folder,
        exist_ok=True,
    )

    save_path = os.path.join(
        segment_folder,
        "langbind_output.pt",
    )

    # Resume only from the NEW clean output directory.
    if os.path.isfile(save_path):
        try:
            existing = torch.load(
                save_path,
                map_location="cpu",
            )

            embedding = existing["embedding"]

            if (
                torch.is_tensor(embedding)
                and embedding.numel() == 1024
                and torch.isfinite(embedding).all()
            ):
                print(
                    f"SKIP VALID EXISTING: {save_path}"
                )
                return {
                    "success": True,
                    "skipped": True,
                }

        except Exception:
            pass

    temp_video = None

    try:
        # -------------------------------------------------------------
        # 1. FFmpeg decode
        # -------------------------------------------------------------
        frames = extract_segment_frames(
            video_path,
            start_time,
            end_time,
        )
        print_memory("after extract_segment_frames")

        # -------------------------------------------------------------
        # 2. Human detection
        # -------------------------------------------------------------
        (
            sampled_indices,
            sampled_masks,
            detection_metadata,
        ) = detect_human_masks(
            frames,
            detector,
            detector_transform,
        )
        print_memory("after detect_human_masks")

        # -------------------------------------------------------------
        # 3. Human-aware enhancement
        # -------------------------------------------------------------
        enhanced_frames = enhance_frames(
            frames,
            sampled_indices,
            sampled_masks,
        )
        print_memory("after enhance_frames")

        # -------------------------------------------------------------
        # 4. Temporary H.264 video
        # -------------------------------------------------------------
        fd, temp_video = tempfile.mkstemp(
            suffix=".mp4"
        )
        os.close(fd)

        write_h264_video(
            enhanced_frames,
            temp_video,
            video_path,
            start_time,
            end_time,
        )
        print_memory("after write_h264_video")

        # -------------------------------------------------------------
        # 5. Validate reconstructed video
        # -------------------------------------------------------------
        validate_video_with_ffmpeg(
            temp_video
        )

        # -------------------------------------------------------------
        # 6. LanguageBind
        # -------------------------------------------------------------
        embedding = get_languagebind_embedding(
            languagebind,
            modality_transform,
            temp_video,
        )
        print_memory("after LanguageBind")

        # -------------------------------------------------------------
        # 7. Metadata
        # -------------------------------------------------------------
        metadata = {
            "video_name": os.path.basename(video_path),
            "video_path": video_path,
            "segment_start": float(start_time),
            "segment_end": float(end_time),
            "segment_duration": float(
                end_time - start_time
            ),
            "total_frames": len(frames),
            "human_aware": True,
            "detector": (
                "Faster R-CNN ResNet50 FPN "
                "COCO pretrained"
            ),
            "person_class_id": PERSON_CLASS_ID,
            "confidence_threshold": (
                CONFIDENCE_THRESHOLD
            ),
            "minimum_box_area_ratio": (
                MIN_BOX_AREA_RATIO
            ),
            "temporal_samples": (
                NUM_TEMPORAL_SAMPLES
            ),
            "actual_sampled_frames": len(
                sampled_indices
            ),
            "blur_kernel": BLUR_KERNEL,
            "alpha": ALPHA,
            "mask_propagation": (
                "nearest sampled frame"
            ),
            "enhancement": (
                "I_prime = I * "
                "(1 + alpha * M)"
            ),
            "languagebind_model": (
                LANGUAGEBIND_MODEL
            ),
            "embedding_dimension": 1024,
            **detection_metadata,
        }

        result = {
            "embedding": embedding,
            "video_name": os.path.basename(video_path),
            "start": float(start_time),
            "end": float(end_time),
            "end": float(end_time),
            "metadata": metadata,
        }

        # Remove duplicate key caused by explicit fields above.
        result["end"] = float(end_time)

        torch.save(
            result,
            save_path,
        )

        # -------------------------------------------------------------
        # 8. Final saved-file verification
        # -------------------------------------------------------------
        check = torch.load(
            save_path,
            map_location="cpu",
        )

        saved_embedding = check["embedding"]

        if (
            not torch.is_tensor(saved_embedding)
            or saved_embedding.numel() != 1024
            or not torch.isfinite(
                saved_embedding
            ).all()
        ):
            raise RuntimeError(
                f"Saved embedding failed validation: "
                f"{save_path}"
            )

        print(
            f"SUCCESS: {video_path} | "
            f"{start_time:.3f}-{end_time:.3f} | "
            f"embedding={tuple(saved_embedding.shape)}"
        )

        return {
            "success": True,
            "skipped": False,
        }

    finally:
        # -------------------------------------------------------------
        # 9. Temporary cleanup
        # -------------------------------------------------------------
        if temp_video is not None:
            try:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
            except Exception as cleanup_error:
                print(
                    f"WARNING: could not remove temporary "
                    f"file {temp_video}: "
                    f"{cleanup_error}"
                )


# ---------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------
def process_video(
    video_path,
    output_root,
    detector,
    detector_transform,
    languagebind,
    modality_transform,
):
    duration = get_video_duration(
        video_path
    )

    number_of_segments = int(
        math.ceil(duration / 2.0)
    )

    video_name = output_video_name(
        video_path
    )

    class_name = os.path.basename(
        os.path.dirname(video_path)
    )

    split_name = os.path.basename(
        os.path.dirname(
            os.path.dirname(video_path)
        )
    )

    # More robust split detection:
    if split_name not in ("train", "test"):
        parts = os.path.normpath(
            video_path
        ).split(os.sep)

        if "train" in parts:
            split_name = "train"
        elif "test" in parts:
            split_name = "test"
        else:
            raise RuntimeError(
                f"Could not determine train/test split: "
                f"{video_path}"
            )

    output_video_root = os.path.join(
        output_root,
        split_name,
        class_name,
        video_name,
    )

    os.makedirs(
        output_video_root,
        exist_ok=True,
    )

    print(
        f"\nVIDEO: {video_path}"
    )
    print(
        f"Duration: {duration:.6f} sec"
    )
    print(
        f"Expected segments: {number_of_segments}"
    )
    print(
        f"Output: {output_video_root}"
    )

    successful_segments = 0
    failed_segments = 0

    for segment_index in range(
        number_of_segments
    ):
        start_time = (
            segment_index * 2.0
        )

        end_time = min(
            start_time + 2.0,
            duration,
        )

        try:
            result = process_segment(
                video_path,
                output_video_root,
                start_time,
                end_time,
                detector,
                detector_transform,
                languagebind,
                modality_transform,
            )

            if result["success"]:
                successful_segments += 1

        except Exception as error:
            failed_segments += 1

            print(
                "\nFAILED SEGMENT:"
            )
            print(
                f"Video: {video_path}"
            )
            print(
                f"Segment: "
                f"{start_time:.6f}-{end_time:.6f}"
            )
            print(
                f"ERROR: {repr(error)}"
            )
            traceback.print_exc()

    return {
        "video": video_path,
        "duration": duration,
        "expected_segments": number_of_segments,
        "successful_segments": successful_segments,
        "failed_segments": failed_segments,
        "split": split_name,
        "class": class_name,
        "output_video_root": output_video_root,
    }


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------
def discover_videos(split):
    split_root = os.path.join(
        DATASET_ROOT,
        split,
    )

    if not os.path.isdir(split_root):
        raise RuntimeError(
            f"Missing dataset split: {split_root}"
        )

    videos = []

    video_extensions = {
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".m4v",
    }

    for class_name in sorted(
        os.listdir(split_root)
    ):
        class_root = os.path.join(
            split_root,
            class_name,
        )

        if not os.path.isdir(class_root):
            continue

        for filename in sorted(
            os.listdir(class_root)
        ):
            path = os.path.join(
                class_root,
                filename,
            )

            if not os.path.isfile(path):
                continue

            if (
                os.path.splitext(
                    filename
                )[1].lower()
                in video_extensions
            ):
                videos.append(path)

    return videos


# ---------------------------------------------------------------------
# Failure logging
# ---------------------------------------------------------------------
def failure_log_path(split):
    return os.path.join(
        os.getcwd(),
        f"human_aware_clean_{split}_failures.txt",
    )


def log_failure(split, video_path, error):
    path = failure_log_path(split)

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n"
            + "=" * 80
            + "\n"
        )
        f.write(
            f"FAILED:{video_path}\n"
        )
        f.write(
            f"ERROR:{repr(error)}\n"
        )
        f.write(
            traceback.format_exc()
        )
        f.write("\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Human-aware 2-second "
            "LanguageBind preprocessing"
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "test",
        ],
        default=None,
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Process one video only. "
            "Used for first validation."
        ),
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=OUTPUT_ROOT,
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "HUMAN-AWARE 2-SECOND LANGUAGEBIND PREPROCESSING"
    )
    print("=" * 80)

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        f"Output: {args.output_root}"
    )

    # -------------------------------------------------------------
    # Initialize detector
    # -------------------------------------------------------------
    print(
        "\nLoading Faster R-CNN..."
    )

    detector, detector_transform = (
        build_detector()
    )

    print(
        "Faster R-CNN loaded."
    )

    # -------------------------------------------------------------
    # Initialize LanguageBind
    # -------------------------------------------------------------
    print(
        "\nLoading LanguageBind..."
    )

    languagebind, modality_transform = (
        build_languagebind()
    )

    print(
        "LanguageBind loaded."
    )

    # -------------------------------------------------------------
    # Single-video mode
    # -------------------------------------------------------------
    if args.video is not None:

        if not os.path.isfile(args.video):
            raise FileNotFoundError(
                args.video
            )

        split = None

        normalized = os.path.normpath(
            args.video
        ).split(os.sep)

        if "train" in normalized:
            split = "train"
        elif "test" in normalized:
            split = "test"

        if split is None:
            raise RuntimeError(
                "Could not determine train/test "
                f"from video path: {args.video}"
            )

        result = process_video(
            args.video,
            args.output_root,
            detector,
            detector_transform,
            languagebind,
            modality_transform,
        )

        print("\n" + "=" * 80)
        print("ONE-VIDEO TEST REPORT")
        print("=" * 80)

        print(
            f"Video: {result['video']}"
        )
        print(
            f"Duration: {result['duration']:.6f}"
        )
        print(
            f"Expected segments: "
            f"{result['expected_segments']}"
        )
        print(
            f"Successful segments: "
            f"{result['successful_segments']}"
        )
        print(
            f"Failed segments: "
            f"{result['failed_segments']}"
        )

        if (
            result["successful_segments"]
            != result["expected_segments"]
        ):
            raise RuntimeError(
                "ONE-VIDEO TEST FAILED: "
                "not all expected segments "
                "were generated."
            )

        print(
            "\nONE-VIDEO TEST PASSED."
        )

        return

    # -------------------------------------------------------------
    # Full split mode
    # -------------------------------------------------------------
    splits = (
        [args.split]
        if args.split is not None
        else ["train", "test"]
    )

    overall = []

    for split in splits:

        videos = discover_videos(
            split
        )

        print(
            f"\n{split.upper()} VIDEOS: "
            f"{len(videos)}"
        )

        total_expected = 0
        total_successful = 0
        total_failed_segments = 0
        successful_videos = 0
        failed_videos = 0

        for video_index, video_path in enumerate(
            videos,
            start=1,
        ):

            print(
                f"\n[{video_index}/{len(videos)}]"
            )

            try:
                result = process_video(
                    video_path,
                    args.output_root,
                    detector,
                    detector_transform,
                    languagebind,
                    modality_transform,
                )

                total_expected += (
                    result["expected_segments"]
                )

                total_successful += (
                    result["successful_segments"]
                )

                total_failed_segments += (
                    result["failed_segments"]
                )

                if (
                    result["successful_segments"]
                    == result["expected_segments"]
                ):
                    successful_videos += 1
                else:
                    failed_videos += 1

            except Exception as error:

                failed_videos += 1

                print(
                    f"\nFAILED:{video_path}"
                )
                print(
                    f"ERROR:{repr(error)}"
                )
                traceback.print_exc()

                log_failure(
                    split,
                    video_path,
                    error,
                )

        coverage = (
            100.0 * total_successful
            / total_expected
            if total_expected > 0
            else 0.0
        )

        print(
            "\n" + "=" * 80
        )
        print(
            f"{split.upper()} FINAL REPORT"
        )
        print(
            "=" * 80
        )
        print(
            f"Total videos: {len(videos)}"
        )
        print(
            f"Successful videos: "
            f"{successful_videos}"
        )
        print(
            f"Failed videos: {failed_videos}"
        )
        print(
            f"Expected segments: "
            f"{total_expected}"
        )
        print(
            f"Generated segments: "
            f"{total_successful}"
        )
        print(
            f"Failed segments: "
            f"{total_failed_segments}"
        )
        print(
            f"Segment coverage: "
            f"{coverage:.4f}%"
        )

        overall.append(
            {
                "split": split,
                "videos": len(videos),
                "successful_videos": successful_videos,
                "failed_videos": failed_videos,
                "expected_segments": total_expected,
                "generated_segments": total_successful,
                "failed_segments": total_failed_segments,
                "coverage_percent": coverage,
            }
        )

    print(
        "\n" + "=" * 80
    )
    print("OVERALL REPORT")
    print("=" * 80)

    for row in overall:
        print(row)


if __name__ == "__main__":
    main()
