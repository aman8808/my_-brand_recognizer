import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # размер изображения для модели
    transforms.ToTensor(),          # преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация (для CLIP и других моделей)
])

dataset = CustomDataset(csv_file='csv_file.csv',
                        root_dir='brands_images',
                        transform=transform)

# загрузчик данных
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for data in dataloader:
    images, labels = data['image'], data['label']
    print(images.shape)
    print(labels)
    break
