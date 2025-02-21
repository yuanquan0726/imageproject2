import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil


def resize_image(image_path, target_size=(224, 224)):
    """
    Resize an image to the target size (224x224).
    
    :param image_path: Path to the image file.
    :param target_size: The target size to resize the image to.
    :return: Resized image in numpy array format.
    """
    with Image.open(image_path) as img:
        img = img.resize(target_size)
    return np.array(img)

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Preprocess and resize all images in the input directory.
    For this, we assume that the input directory contains subfolders (each for a species).
    
    :param input_dir: Directory containing the image files, organized by species.
    :param output_dir: Directory to save the resized images, organized by species.
    :param target_size: Target size for image resizing (default 224x224).
    """
    os.makedirs(output_dir, exist_ok=True)
    for species_folder in os.listdir(input_dir):
        species_folder_path = os.path.join(input_dir, species_folder)
        if os.path.isdir(species_folder_path):  # Check if it's a folder (species)
            output_species_folder = os.path.join(output_dir, species_folder)
            os.makedirs(output_species_folder, exist_ok=True)

            for image_file in os.listdir(species_folder_path):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
                    image_path = os.path.join(species_folder_path, image_file)
                    resized_image = resize_image(image_path, target_size)
                    output_image_path = os.path.join(output_species_folder, image_file)
                    Image.fromarray(resized_image).save(output_image_path)



import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_images(input_dir, output_dir, augmentations=10, target_size=(224, 224)):
    """
    Apply augmentations (e.g., rotation, flip) to images and save them.
    
    :param input_dir: Directory containing the image files (organized by species).
    :param output_dir: Directory to save augmented images.
    :param augmentations: Number of augmented images to generate for each input image.
    :param target_size: Target size to resize the augmented images to (default 224x224).
    """
    os.makedirs(output_dir, exist_ok=True)
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize image pixels to [0, 1] range
        rotation_range=15,  # Random rotations up to 15 degrees
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2,  # Randomly shift images vertically
        shear_range=0.2,  # Randomly shear the image
        zoom_range=0.2,  # Random zoom
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest',  # Fill the pixels after transformations
    )

    for species_folder in os.listdir(input_dir):
        species_folder_path = os.path.join(input_dir, species_folder)
        if os.path.isdir(species_folder_path):  # Check if it's a folder (species)
            output_species_folder = os.path.join(output_dir, species_folder)
            os.makedirs(output_species_folder, exist_ok=True)

            for image_file in os.listdir(species_folder_path):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
                    image_path = os.path.join(species_folder_path, image_file)
                    img = Image.open(image_path)
                    img = np.array(img)

                    # Ensure the image has 3 dimensions (height, width, channels)
                    if len(img.shape) == 2:  # Grayscale image (1 channel)
                        img = np.stack([img] * 3, axis=-1)  # Convert to 3 channels (RGB)

                    # Add an extra batch dimension (1, height, width, channels)
                    img = img.reshape((1,) + img.shape)

                    # Manually resize the image to the target size
                    img_resized = np.array([Image.fromarray(img[0]).resize(target_size)])

                    # Generate augmented images
                    i = 0
                    for batch in datagen.flow(img_resized, batch_size=1, save_to_dir=output_species_folder, save_prefix='aug_', save_format='jpeg'):
                        i += 1
                        if i >= augmentations:
                            break



input_image_dir = '../../data/raw/bird_images'
output_resized_dir = '../../data/processed/resized_images'

# Preprocess the images by resizing them
preprocess_images(input_image_dir, output_resized_dir)

# Augment the images and save them in the augmented directory
augment_images('../../data/processed/resized_images', '../../data/processed/augmented_images')

