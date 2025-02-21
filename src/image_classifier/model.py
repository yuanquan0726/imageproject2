import os
import shutil
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# def split_data(spectrogram_dir, train_dir, val_dir, val_split=0.2):
#     """
#     Split the augmented images into training and validation sets.
    
#     :param spectrogram_dir: Directory containing all spectrograms (augmented images).
#     :param train_dir: Directory to save training data.
#     :param val_dir: Directory to save validation data.
#     :param val_split: Fraction of data to be used for validation.
#     """
#     # Create train and val directories for each species
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)

#     # List all species subdirectories
#     species_folders = [f for f in os.listdir(spectrogram_dir) if os.path.isdir(os.path.join(spectrogram_dir, f))]

#     for species in species_folders:
#         species_folder_path = os.path.join(spectrogram_dir, species)
#         # Get all spectrogram files in the species folder
#         image_files = [f for f in os.listdir(species_folder_path) if f.endswith('.jpeg')]

#         # Shuffle files and split into train and val
#         random.shuffle(image_files)
#         val_count = int(len(image_files) * val_split)
#         val_files = image_files[:val_count]
#         train_files = image_files[val_count:]

#         # Move train images
#         species_train_dir = os.path.join(train_dir, species)
#         species_val_dir = os.path.join(val_dir, species)
#         os.makedirs(species_train_dir, exist_ok=True)
#         os.makedirs(species_val_dir, exist_ok=True)

#         for file in train_files:
#             shutil.move(os.path.join(species_folder_path, file), os.path.join(species_train_dir, file))

#         for file in val_files:
#             shutil.move(os.path.join(species_folder_path, file), os.path.join(species_val_dir, file))

# spectrogram_dir = '../../data/processed/augmented_images'  # Path where your images are located
# train_dir = '../../data/processed/images/train_images'  # Path to save training data
# val_dir = '../../data/processed/images/val_images'  # Path to save validation data

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Function to build the ResNet model
def build_resnet_image_model(input_shape=(224, 224, 3), num_classes=10, learning_rate=0.0001):
    """
    Builds a ResNet50-based image classification model.
    
    :param input_shape: The shape of input images (224x224x3).
    :param num_classes: The number of classes in the dataset.
    :param learning_rate: The learning rate for the optimizer.
    :return: The compiled Keras model.
    """
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model layers (use transfer learning)

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2048, activation='relu', kernel_regularizer=l2(0.01)),  # L2 regularization added
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    
    # Use Adam optimizer with a specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Fine-tune the ResNet50 model
def fine_tune_model(model, train_generator, val_generator, learning_rate=0.0001, epochs=5):
    """
    Fine-tune the model by unfreezing some layers and retraining.
    
    :param model: Pre-trained model to fine-tune.
    :param train_generator: Data generator for training data.
    :param val_generator: Data generator for validation data.
    :param learning_rate: Learning rate for the optimizer.
    :param epochs: Number of epochs to fine-tune.
    :return: Fine-tuned model and history of training.
    """
    # Unfreeze the last 20 layers of ResNet50
    base_model = model.layers[0]  # Accessing the ResNet50 base model
    for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
        layer.trainable = True

    # Recompile the model with a smaller learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Retrain the model with fine-tuned layers
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
    
    return model, history

# Prepare Data Generators (training and validation)
def prepare_data_generators(train_dir, val_dir, batch_size=32):
    """
    Prepares data generators for training and validation with augmentation.
    
    :param train_dir: Directory for training data.
    :param val_dir: Directory for validation data.
    :param batch_size: Batch size for the generator.
    :return: Train and validation generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize the images to [0, 1]
        rotation_range=40,  # Increased rotation range
        width_shift_range=0.3,  # Increased shift range
        height_shift_range=0.3,  # Increased shift range
        shear_range=0.3,  # Increased shear range
        zoom_range=0.3,  # Increased zoom range
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest'  # Fill the pixels after transformations
    )

    val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale validation images

    # Train data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Resize images to 224x224
        batch_size=batch_size,
        class_mode='categorical'  # Multi-class classification
    )

    # Validation data generator
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),  # Resize images to 224x224
        batch_size=batch_size,
        class_mode='categorical'  # Multi-class classification
    )
    
    return train_generator, val_generator

# Plotting the training history
def plot_training_history(history, plot_filename='training_history.png'):
    """
    Plot training and validation accuracy and loss, and save the plot.
    
    :param history: The history object from model training.
    :param plot_filename: Filename to save the plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()


train_dir = '../../data/processed/images/train_images'  # Path to save training data
val_dir = '../../data/processed/images/val_images'  # Path to save validation data
# Prepare data generators
train_generator, val_generator = prepare_data_generators(train_dir, val_dir)

# Build the model from scratch (no loading)
model = build_resnet_image_model(input_shape=(224, 224, 3), num_classes=10, learning_rate=0.0001)

# Fine-tune the model
fine_tuned_model, fine_tuned_history = fine_tune_model(model, train_generator, val_generator, learning_rate=0.0001, epochs=20)

# Save the fine-tuned model
fine_tuned_model.save('resnet_finetuned_model.h5')

# Plot and save the training history
plot_training_history(fine_tuned_history, 'fine_tuned_training_history.png')

