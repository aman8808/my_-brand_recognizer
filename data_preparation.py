import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def prepare_data(df, image_size=(224, 224)):
    images = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(df['label'].unique())}  # Создаем маппинг меток
    for _, row in df.iterrows():
        image = Image.open(row['image_path'])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(image_size)
        image = np.array(image)
        images.append(image)
        labels.append(label_map[row['label']])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def main():
    df = pd.read_csv('csv_file.csv')

    images, labels = prepare_data(df)

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.15,
                                                                          random_state=42)

    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_labels)
    np.save('val_images.npy', val_images)
    np.save('val_labels.npy', val_labels)


if __name__ == "__main__":
    main()
