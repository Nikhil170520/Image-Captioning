## 🧠 Project Title: Image Captioning & Segmentation Web App (Streamlit)
# 📌 Description
This project is a web-based application built using Streamlit that performs two major computer vision tasks:

Image Captioning – Automatically generates a descriptive sentence based on the contents of an uploaded image.

Image Segmentation – Identifies and highlights different objects in the image by overlaying segmentation masks using a pre-trained deep learning model.

The app allows users to upload images and see both the generated caption and a visual mask-based segmentation of detected objects.

🚀 Features
🔤 Image Captioning (simulated with randomly selected realistic captions)

🧠 Image Segmentation using Mask R-CNN pretrained on the COCO dataset

📷 Upload any .jpg, .jpeg, or .png image

🖥️ Real-time visualization with Streamlit

🎯 Lightweight and easy to deploy

🛠️ Technologies Used
Component	Library
Web Framework	Streamlit
Image Processing	Pillow, OpenCV
Segmentation Model	PyTorch + torchvision (maskrcnn_resnet50_fpn)
Visualization	matplotlib, NumPy
Dummy Captioning	Python (random caption selection)

📂 Folder Structure

Image_Captioning_Segmentation/
├── app.py                     # Streamlit app
├── captioning_model.py       # Simulated caption generator
├── segmentation_model.py     # Mask R-CNN model loader
├── utils.py                  # Utility functions
├── requirements.txt          # Project dependencies
└── sample.jpg                # Example image (optional)
