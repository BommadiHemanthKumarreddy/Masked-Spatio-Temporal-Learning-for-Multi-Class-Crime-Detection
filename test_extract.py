from data_preprocess.langbind_preprocessing_2sec_v2 import extract_clip

input_video = "/home/y_rajeshwar/final_year_project_29/dataset/train/abuse/Screen_Recording_2025-12-14_000108.mp4"

output_video = "/tmp/test_clip.mp4"

extract_clip(
    input_video=input_video,
    output_video=output_video,
    start_time=0,
    duration=2
)

print("Done!")
