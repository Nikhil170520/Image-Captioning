import streamlit as st
from PIL import Image
from captioning_model import ImageCaptioningModel
from segmentation_model import ImageSegmentationModel
from utils import convert_pil_to_np
import numpy as np
import matplotlib.pyplot as plt

# Load models
caption_model = ImageCaptioningModel()
segment_model = ImageSegmentationModel()

st.set_page_config(page_title="Image Captioning & Segmentation", layout="centered")
st.title("🖼️ Image Captioning & Segmentation")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    st.subheader("📌 Generated Caption")
    caption = caption_model.generate_caption(image_pil)
    st.success(caption)

    st.subheader("🎯 Segmentation Result")
    masks, labels, scores = segment_model.segment_image(image_pil)

    if len(masks) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(image_pil)
        for i in range(min(5, len(masks))):  # Limit for clarity
            mask = masks[i]
            ax.imshow(mask, alpha=0.5, cmap='jet')
        st.pyplot(fig)
    else:
        st.warning("No objects detected.")
