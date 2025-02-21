import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load the trained models
audio_model = tf.keras.models.load_model('resnet_audio_model.h5')  # For audio classification
image_model = tf.keras.models.load_model('resnet_finetuned_model.h5')  # For image classification


# Define the audio class labels (your list)
class_labels_audio = ['bewickii', 'polyglottos', 'migratorius', 'melodia', 'cardinalis']

# Define the image class labels (based on folder names)
# image_folder_path = "../../data/processed/augmented_images"  # Adjust this path to your local folder path
# class_labels_image = sorted(os.listdir(image_folder_path))
class_labels_image =  ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '004.Groove_billed_Ani', '005.Crested_Auklet', '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet', '009.Brewer_Blackbird', '010.Red_winged_Blackbird']
# print("Number of image classes:", len(class_labels_image))  # This should be 10
# print("Class labels from the folder names:", class_labels_image)

# print("Number of classes predicted by the model:", image_model.output_shape[1])  

# Preprocess the uploaded audio file to generate a spectrogram
def preprocess_audio(file_path, target_shape=(1026, 257)):
    y, sr = librosa.load(file_path, sr=None)
    spectrogram = np.abs(librosa.stft(y, n_fft=512, hop_length=64))
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    spectrogram_resized = np.resize(spectrogram_db, target_shape)  # Resize to (1026, 257)
    spectrogram_resized = spectrogram_resized / 80 + 1  # Normalize
    return spectrogram_resized

# Predict bird species from the uploaded audio file
def predict_audio_class(file_path):
    # Preprocess the audio and create a spectrogram
    spectrogram = preprocess_audio(file_path)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension for the model

    # Make prediction
    predictions = audio_model.predict(spectrogram)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    predicted_label = class_labels_audio[predicted_class]
    confidence = predictions[0][predicted_class]

    return predicted_label, confidence

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to 224x224 for the model
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Predict bird species from the uploaded image
def predict_image_class(img):
    # Preprocess the image and make predictions
    img_array = preprocess_image(img)
    predictions = image_model.predict(img_array)
    
    # Extract the predicted class index
    predicted_class = np.argmax(predictions, axis=-1)[0]

    # Ensure the predicted class index is within the range of available labels
    if predicted_class < len(class_labels_image):
        predicted_label = class_labels_image[predicted_class]
        confidence = predictions[0][predicted_class]
        return predicted_label, confidence
    else:
        raise ValueError(f"Predicted class index {predicted_class} is out of bounds for class_labels_image")


# Streamlit app layout
st.title('Bird Species Classifier (Audio & Image)')
st.write('Upload an audio file or an image to predict the bird species.')

# File uploader for audio files
uploaded_audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_audio_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_audio_file.getbuffer())
    
    # Display audio waveform (optional)
    y, sr = librosa.load("temp_audio.wav", sr=None)
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')
    st.pyplot(plt)

    # Predict the class of the bird species from the uploaded audio
    predicted_species, confidence = predict_audio_class("temp_audio.wav")

    # Display the results
    st.write(f"**Predicted Bird Species from Audio:** {predicted_species}")
    st.write(f"**Confidence Level:** {confidence * 100:.2f}%")

# File uploader for images
uploaded_image_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_image_file is not None:
    # Open the image file
    img = Image.open(uploaded_image_file)

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Get the class labels from the folder names (assuming a local folder structure)
    image_folder_path = "../../data/processed/images"  # Replace this with the path to your image data folder
    class_labels = sorted(os.listdir(image_folder_path))  # Get class labels from folder names

    # Predict the class of the bird species from the uploaded image
    predicted_species, confidence = predict_image_class(img)

    # Display the results
    st.write(f"**Predicted Bird Species from Image:** {predicted_species}")
    st.write(f"**Confidence Level:** {confidence * 100:.2f}%")
