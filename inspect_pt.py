import os
import torch

root = "/home/y_rajeshwar/final_year_project_29/dataset/langbind_features/train/abuse"

video = os.listdir(root)[0]

path = os.path.join(root, video, "langbind_output.pt")

print("Loading:", path)

data = torch.load(path)

print("Type:", type(data))

if isinstance(data, dict):
    print("Keys:", data.keys())
    for k, v in data.items():
        print(f"{k}: {type(v)}")
        if torch.is_tensor(v):
            print("Shape:", v.shape)
