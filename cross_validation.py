import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os


class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.fc1 = nn.Linear(3 * 128 * 128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.data['label'] = self.label_encoder.fit_transform(self.data['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def cross_validation(csv_file, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(csv_file=csv_file, transform=data_transforms)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{num_folds}")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        model = YourModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer)

        
        accuracy = evaluate_model(model, test_loader)
        fold_results.append(accuracy)
        print(f"Accuracy for fold {fold + 1}: {accuracy:.4f}")

    print(f"Mean accuracy: {sum(fold_results) / len(fold_results):.4f}")


if __name__ == "__main__":
    csv_file = 'csv_file.csv'
    cross_validation(csv_file, num_folds=5)
