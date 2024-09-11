import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model_path = "model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def check_brand(image, brand_name):
    brand_name_with_suffix = f"{brand_name}_logo"

    inputs = processor(text=[brand_name_with_suffix], images=image, return_tensors="pt", padding=True)
    image_tensor = inputs['pixel_values'].to(device)
    text_tensor = inputs['input_ids'].to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=image_tensor)
        text_features = model.get_text_features(input_ids=text_tensor)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(image_features, text_features.T).squeeze().cpu().numpy()

        print("Image Features Shape:", image_features.shape)
        print("Text Features Shape:", text_features.shape)
        print("Similarity Shape:", similarity.shape)
        print("Similarity Values:", similarity)

        if similarity.ndim == 0:
            similarity = np.array([similarity])

        threshold = 0.5  
        return similarity[0] > threshold if similarity.size > 0 else False

st.title("Определение соответствия бренда")

uploaded_file = st.file_uploader("Выберите изображение товара", type=["jpg", "jpeg", "png"])
brand_name = st.text_input("Введите наименование бренда")

if uploaded_file and brand_name:
    image = Image.open(uploaded_file).convert("RGB")
    result = check_brand(image, brand_name)

    st.image(image, caption="Загруженное изображение", use_column_width=True)
    st.write(f"Соответствие бренда: {'Да (True)' if result else 'Нет (False)'}")
