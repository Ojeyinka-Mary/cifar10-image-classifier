import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load Model
model = load_model("cifar10_model.h5")

# CIFAR-10 Classes
classes = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

st.title("CIFAR-10 Image Classification System")

st.write("Upload an image and let the AI predict its class.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")