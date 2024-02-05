


mymodel = Cov_Att()
mymodel = mymodel.float()

import numpy as np

# Create a 28*28*512 image
image = np.random.rand(28, 28, 512)

# Create a 1*1*512 text encoding
text_encoding = np.random.rand(1, 1, 512)

# 将 NumPy 数组转换为 PyTorch 张量
image_tensor = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2).float()  # 假设 image 的形状是 (28, 28, 512)
text_encoding_tensor = torch.tensor(text_encoding).float()  # 假设 text_encoding 的形状是 (1, 1, 512)

print(image_tensor.shape)
print(text_encoding_tensor.shape)

# 将张量传递给模型
x = mymodel(image_tensor, text_encoding_tensor)

print(x.shape)



transform_args = dict(
    resize=256,
    crop_size=224,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

root_dir = ".\MIMIC-CXR"

dataset = MyDataset(root_dir, transform_args,label_map)

# 构建DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
for img, texts ,labels in dataloader:
    labels = list(labels)
    numeric_labels = [label_map[label] for label in labels]
    path = dataset.sample_random_images_by_labels(numeric_labels).to(torch.device('cuda'))
    print(path.device)
