import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

MODEL_PATH = "Custom_Lite_100x100.h5"

st.set_page_config(page_title="X-Ray Pneumonia Classifier", layout="wide", initial_sidebar_state="expanded")
selected_page = st.sidebar.radio("Navigation", ["Upload Image", "About"])

@st.cache_resource()
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

if selected_page == "About":
    st.title("Welcome to the Image Classifier")
    st.write("Use the sidebar to navigate.")

elif selected_page == "Upload Image":
    st.title("Pneumonia X-Ray classifier")
    st.subheader("Upload an Image")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # print(uploaded_image)
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", width=500)

        if st.button("Classify"):
            # Preprocess the image for the model
            image = tf.keras.preprocessing.image.load_img(uploaded_image, target_size=(100, 100))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])
            input_arr = input_arr.astype('float32') / 255.

            # Make predictions using the model
            prediction = float(model.predict(input_arr,verbose = 0)[0])
            # print("\n", prediction)

            if prediction>0.5 :
                defect = 'Pneumonia: Positive'
            else: 
                defect = 'Pneumonia: Negative'

            st.subheader(defect)
            st.write(f"Confidence: {prediction:.2%}")