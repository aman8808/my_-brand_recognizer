import os
from PIL import Image
import albumentations as A
import numpy as np


class ImageAugmentor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Определение преобразований
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Resize(height=256, width=256, p=1.0)
        ])

    def augment_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        augmented_images = []
        for i in range(5):
            augmented = self.transform(image=image_np)
            augmented_image = Image.fromarray(augmented['image'])
            augmented_images.append(augmented_image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for i, aug_image in enumerate(augmented_images):
            aug_image_path = os.path.join(self.output_folder, f"{base_name}_aug_{i}.png")
            aug_image.save(aug_image_path)

    def augment_all_images(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.input_folder, filename)
                self.augment_image(image_path)


if __name__ == '__main__':
    input_folder = 'brands_images'
    output_folder = 'augmented_images'
    augmentor = ImageAugmentor(input_folder, output_folder)
    augmentor.augment_all_images()
    print("Аугментация завершена.")
