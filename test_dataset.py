from dataset_finetune import LangBindDataset

train = LangBindDataset(
    "/home/y_rajeshwar/final_year_project_29/dataset/langbind_features/train"
)

print(len(train))

x, y = train[0]

print(x.shape)
print(y)
