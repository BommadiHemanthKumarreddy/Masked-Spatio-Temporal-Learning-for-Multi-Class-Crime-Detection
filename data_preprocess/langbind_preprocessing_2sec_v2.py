import sys
import math
import os
import cv2
# Get the directory of the current script
current_file_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_file_directory)
SRC = os.path.join(parent_directory, 'data_preprocess')

# Add the constructed path to sys.path
sys.path.append(SRC)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from languagebind_2sec import LanguageBind, to_device, transform_dict

from tqdm import tqdm
import torch
import argparse
import subprocess
import tempfile
import shutil
from more_itertools import chunked

from data_preprocess.helper_functions import load_json

DEFAULT_VIDEO_PATH_FILE = os.path.join(SRC, 'test_assets/prepro_path_tuples.json') # use your saved path_tuples
DEFAULT_END_JOB_FILE = '~/tmp/tmp_end_job_file.txt'
def extract_clip(input_video, output_video, start_time, duration=2):
    """
    Extract a temporary clip from a video using FFmpeg.
    """

    cmd = [
        "/usr/bin/ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", input_video,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "copy",
        output_video,
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

def main():
    # Load and initialize LanguageBind model
    parser = argparse.ArgumentParser(description="Run preprocessing jobs on a dataset.")
    parser.add_argument('--video_path_file', type=str, default=DEFAULT_VIDEO_PATH_FILE)  # default values are only for testing.
    parser.add_argument('--end_job_file', type=str, default='~/tmp/tmp_end_job_file.txt')  # default values are only for testing.
    args = parser.parse_args()

    # Model initialization
    batch_size = 2
    video_model_name = 'LanguageBind_Video_Huge_V1.5_FT'
    device = 'cuda'
    clip_type = {'video': video_model_name}  # use only video type

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    # Iterate over batches
    video_paths = load_json(args.video_path_file)["file_paths"]

    for in_path, out_path in tqdm(video_paths):

        cap = cv2.VideoCapture(in_path)

        if not cap.isOpened():
            print(f"Cannot open: {in_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        cap.release()

        if fps <= 0:
            print(f"Invalid FPS: {in_path}")
            continue

        duration = frame_count / fps

        start = 0

        while start < duration:

            end = min(start + 2, duration)
            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as tmp:

                temp_video = tmp.name

            extract_clip(
                input_video=in_path,
                output_video=temp_video,
                start_time=start,
                duration=end - start,
            )

            inputs = {
                "video": to_device(
                    modality_transform["video"]([temp_video]),
                    device
                )
            }

            with torch.no_grad():
                embedding = model(inputs)["video"][0].cpu()

            segment_folder = os.path.join(
                out_path,
                f"{int(start):05d}-{int(math.ceil(end)):05d}"
            )

            os.makedirs(segment_folder, exist_ok=True)

            torch.save(
                {
                    "embedding": embedding,
                    "video_name": os.path.basename(in_path),
                    "start": start,
                    "end": end,
                },
                os.path.join(segment_folder, "langbind_output.pt")
            )

            os.remove(temp_video)
            start += 2
if __name__ == '__main__':
    main()
