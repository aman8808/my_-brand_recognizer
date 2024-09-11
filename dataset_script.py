import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset


def main():
    csv_file = 'csv_file.csv'
    root_dir = 'brands_images'

    df = pd.read_csv(csv_file)
    missing_files = [img for img in df['image_path'] if not os.path.isfile(os.path.join(root_dir, img))]
    if missing_files:
        print(f"Warning: The following files are missing: {missing_files}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    for images, labels in train_loader:
        print(images.size(), labels)


if __name__ == '__main__':
    main()
