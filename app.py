import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Medical Diagnosis AI 🩺")

model = tf.keras.models.load_model("xray_model_small.h5")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150,150))
    st.image(uploaded_file, caption="Uploaded Image")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    result = model.predict(img_array)
    score = result[0][0]

    if score > 0.5:
        st.error(f"PNEUMONIA ({round(score*100,2)}%)")
    else:
        st.success(f"NORMAL ({round((1-score)*100,2)}%)")