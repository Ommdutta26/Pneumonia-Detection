import streamlit as st
from PIL import Image
from utils import predict_image, apply_gradcam

st.title("Pneumonia Detection using Chest X-Ray")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-ray", width="stretch")

    # prediction
    result, prob = predict_image(image)

    st.subheader("Prediction:")
    st.write(result)

    st.write("Confidence:", round(prob * 100, 2), "%")

    # GradCAM
    st.subheader("Grad-CAM Explanation")

    heatmap = apply_gradcam(image)

    st.image(heatmap, caption="Model Attention (Grad-CAM)", width="stretch")