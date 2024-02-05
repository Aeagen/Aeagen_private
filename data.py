import torch
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as T
from labels_num import label_map

class MyDataset(Dataset):

    def __init__(self, root_dir, transform_args, label_map):

        self.img_paths = []
        self.labels = []
        self.transform_args = transform_args
        self.label_paths_dict = {}
        self.label_map = label_map

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for sub_subdir in os.listdir(subdir_path):
                    sub_subdir_path = os.path.join(subdir_path, sub_subdir)
                    if os.path.isdir(sub_subdir_path):
                        # 注意这里需要再遍历一级子目录
                        for sub_sub_root, _, files in os.walk(sub_subdir_path):
                            for f in files:
                                if f.endswith('.png'):
                                    img_path = os.path.join(sub_sub_root, f)
                                    label = sub_sub_root.split(os.path.sep)[-3:]
                                    label_num = self.label_map[label[2]]
                                    self.img_paths.append(img_path)
                                    self.labels.append(label)

                                    # Add image path to corresponding label set
                                    if label_num not in self.label_paths_dict:
                                        self.label_paths_dict[label_num] = set()
                                    self.label_paths_dict[label_num].add(img_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = torchvision.io.read_image(img_path)

        # 将 torch.Tensor 转换为 PIL 图像
        img = torchvision.transforms.ToPILImage()(img)

        # 定义transforms
        transforms = []
        if self.transform_args['resize']:
            transforms.append(T.Resize(self.transform_args['resize']))
        if self.transform_args['crop_size']:
            transforms.append(T.CenterCrop(self.transform_args['crop_size']))

        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=self.transform_args['mean'],
                        std=self.transform_args['std'])
        ])

        transform = T.Compose(transforms)
        img = transform(img)

        label = "This medical image is a " + self.labels[idx][0] + \
                " scan, with imaging information originating from the " + self.labels[idx][1] + \
                " of the patient, primarily depicting " + self.labels[idx][2]

        return img, label ,self.labels[idx][2]

    def sample_random_images_by_labels(self, label_list):
        sampled_paths = []
        sampled_images=[]
        for label in label_list:
            if label in self.label_paths_dict:
                sampled_paths.append(random.choice(list(self.label_paths_dict[label])))
        for label_path in sampled_paths:
            img = torchvision.io.read_image(label_path)
            img = torchvision.transforms.ToPILImage()(img)
            transforms = [
                T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=self.transform_args['mean'],
                            std=self.transform_args['std'])
            ]
            transform = T.Compose(transforms)
            img = transform(img)
            img = img.permute(1,2,0)
            sampled_images.append(img)
        return torch.stack(sampled_images)


