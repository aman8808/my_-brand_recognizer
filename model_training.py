import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Открываем изображение
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(model, train_loader, optimizer, device, num_epochs=5):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}')

def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

def main():
    # Параметры
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-4
    num_folds = 5

    df = pd.read_csv('csv_file.csv')
    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()

    label_to_id = {label: idx for idx, label in enumerate(set(labels))}
    labels = [label_to_id[label] for label in labels]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f'Fold {fold + 1}/{num_folds}')

        train_dataset = CustomDataset([image_paths[i] for i in train_idx], [labels[i] for i in train_idx], transform=transform)
        val_dataset = CustomDataset([image_paths[i] for i in val_idx], [labels[i] for i in val_idx], transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model_name = "google/vit-base-patch16-224-in21k"  # Замените на нужную модель
        model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=len(label_to_id))

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_model(model, train_loader, optimizer, device, num_epochs)

        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device)
        fold_results.append((accuracy, precision, recall, f1))

        print(f'Fold {fold + 1} results:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

    avg_results = np.mean(fold_results, axis=0)
    print(f'\nAverage results across all folds:')
    print(f'Accuracy: {avg_results[0]:.4f}')
    print(f'Precision: {avg_results[1]:.4f}')
    print(f'Recall: {avg_results[2]:.4f}')
    print(f'F1 Score: {avg_results[3]:.4f}')

if __name__ == "__main__":
    main()