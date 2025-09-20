import streamlit as st
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog
import os

# Constants
IMG_SIZE = 128
CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash',
               'white-glass']


# Function to compute color histogram
def compute_color_histogram(img, bins=32):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b])
    return hist / (hist.sum() + 1e-6)


# Function to extract features from the image
def extract_features(img):
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_gray = img_resized.convert('L')
    img_array = np.array(img_gray) / 255.0
    hog_features = hog(img_array, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), orientations=9,
                       feature_vector=True)
    color_features = compute_color_histogram(img_resized)
    features = np.concatenate([hog_features, color_features])
    return features.reshape(1, -1)


# Function to load the selected model
def load_model(model_name):
    filename_map = {
        "SVM": "svm_garbage_classifier.pkl",
        "Random Forest": "random_forest_garbage_classifier.pkl",
        "k-NN": "k-nn_garbage_classifier.pkl",
        "Improved SVM": "svm_garbage_classifier_tuned.pkl",
        "Improved Random Forest": "random_forest_garbage_classifier_tuned.pkl",
        "Improved k-NN": "k-nn_garbage_classifier_tuned.pkl"
    }
    model_path = filename_map.get(model_name)
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return joblib.load(model_path)


# Streamlit UI
st.title("♻️ Garbage Classifier")
st.write("Upload an image of garbage and select a model to classify its "
         "type.")

# Model selector with improved models
model_choice = st.selectbox("Choose a model", [
    "SVM", "Random Forest", "k-NN",
    "Improved SVM", "Improved Random Forest", "Improved k-NN"
])

# Image uploader
uploaded_file = st.file_uploader("Upload an image",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features from the image
    features = extract_features(image)

    # Load the selected model
    model = load_model(model_choice)
    if model:
        prediction = model.predict(features)[0]
        predicted_class = CLASS_NAMES[prediction]
        st.success(f"Predicted Class: **{predicted_class}**")
