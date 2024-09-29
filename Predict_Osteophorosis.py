import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Streamlit app title
st.title('Osteoporosis Predictor')

# Function to predict osteoporosis from an image using a CNN model
def predict_osteoporosis(image, model):
    """
    Predicts whether an X-ray image shows osteoporosis.

    Args:
        image: The uploaded X-ray image as a PIL Image object.
        model: The trained CNN model.

    Returns:
        A prediction probability (0 for normal, 1 for osteoporosis).
    """
    try:
        # Convert the image to grayscale
        image = image.convert("L")
        
        # Resize the image to match the model input size
        image = image.resize((224, 224))  # Update this to match your model's input size
        
        # Normalize the image and add necessary dimensions for batch and channels
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = np.expand_dims(image_array, axis=3)  # Add channel dimension (grayscale)

        # Make prediction
        prediction = model.predict(image_array)

        return prediction[0][0]  # Return the prediction probability

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Load your trained model (make sure 'Osteophorosis predictor.h5' is in your working directory)
model = load_model('Osteophorosis predictor.h5')

# Upload an image file
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=False, width=200)

        # Predict the result
        prediction = predict_osteoporosis(image, model)

        # Display prediction in the second column with emphasis
        with col2:
            if prediction is not None:
                if prediction < 0.5:
                    st.markdown("<h2 style='color: green;'>Prediction: Normal (No Osteoporosis)</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color: red;'>Prediction: Osteoporosis Detected</h2>", unsafe_allow_html=True)
                st.markdown(f"<h4>Probability of Osteoporosis: {prediction:.4f}</h4>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error uploading or processing the image: {e}")
