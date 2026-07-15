import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="cifar10_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

st.title("CIFAR-10 Image Classification System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((32, 32))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(
        input_details[0]['index'],
        img
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    predicted_class = classes[np.argmax(prediction)]

    st.success(
        f"Prediction: {predicted_class}"
    )
