import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Step 1: Creating labels from the metadata CSV file
metadata_path = "../../data/metadata/bird_songs_metadata.csv"
bird_labels = pd.read_csv(metadata_path, usecols=['species'])
bird_labels = bird_labels.values

# Convert species names to numerical labels
bird_labels[bird_labels == ['bewickii']] = 0
bird_labels[bird_labels == ['polyglottos']] = 1
bird_labels[bird_labels == ['migratorius']] = 2
bird_labels[bird_labels == ['melodia']] = 3
bird_labels[bird_labels == ['cardinalis']] = 4
bird_labels = np.squeeze(bird_labels)

# Step 2: Creating a list of file paths from metadata
file_names = pd.read_csv(metadata_path, usecols=['filename'])
file_names = np.squeeze(file_names.values)

# Base directory for the raw audio files
base_audio_path = "../../data/raw/bird_songs"

# Combine file paths to form the full file path
bird_filepaths = np.array([])
for file in file_names:
    bird_filepaths = np.append(bird_filepaths, os.path.join(base_audio_path, file))

# Step 3: Splitting data into training and validation sets
bird_filepaths_train, bird_filepaths_val, bird_labels_train, bird_labels_val = train_test_split(
    bird_filepaths, bird_labels, test_size=0.10, random_state=2419)

# Step 4: Function that reads audio files from file paths using librosa
def read_file(path):
    y, _ = librosa.load(path)
    return y

# Step 5: Convert amplitude to decibel (dB) scale
def spec_to_db(y):
    y_db = librosa.amplitude_to_db(y, ref=100)
    return y_db

# Step 6: Map function that processes the audio files and converts them into spectrograms
def map_function(path_tensor, label):
    # Read the audio file
    y = tf.numpy_function(read_file, inp=[path_tensor], Tout=tf.float32)
    # Generate the spectrogram using STFT (Short-Time Fourier Transform)
    spectrogram = tf.abs(tf.signal.stft(y, frame_length=512, frame_step=64))
    # Convert the spectrogram to dB scale
    spectrogram_db = tf.numpy_function(spec_to_db, inp=[spectrogram], Tout=tf.float32)
    spectrogram_db = spectrogram_db / 80 + 1  # Normalize the spectrogram
    return spectrogram_db, label

# Step 7: Function to create a TensorFlow Dataset and save it to TFRecord format
def create_tfrecord(dataset, filename):
    """
    Convert the TensorFlow dataset into a TFRecord file.
    """
    writer = tf.io.TFRecordWriter(filename)
    
    for spectrogram, label in dataset:
        # Remove the batch dimension (we are encoding one spectrogram at a time)
        spectrogram = spectrogram[0]  # Assuming batch size is 1
        
        # Convert spectrogram to uint8 in the range [0, 255]
        spectrogram_uint8 = tf.cast(spectrogram * 255.0, tf.uint8)  # Scale float to uint8
        
        # Ensure the spectrogram has only one channel (1D)
        spectrogram_uint8 = tf.expand_dims(spectrogram_uint8, axis=-1)  # Add the channel dimension
        
        # Encode the spectrogram as JPEG (expecting a 3D tensor with shape [height, width, channels])
        encoded_spectrogram = tf.io.encode_jpeg(spectrogram_uint8)
        
        # Ensure the label is a scalar integer
        # Handle case where label might be a tensor with batch size > 1
        if isinstance(label, tf.Tensor):
            label = label.numpy().item() if label.shape == () else label[0].numpy().item()
        else:
            label = int(label)   # If label is not a tensor, convert directly to integer
        
        # Create the feature dictionary
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_spectrogram.numpy()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # Ensure label is scalar
        }

        # Create an Example and write it to the TFRecord file
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    
    writer.close()
    print(f"TFRecord file saved to: {filename}")



# Step 8: Function to create and save the training and validation datasets to TFRecord format
def make_and_save_dataset(bird_labels, bird_filepaths, shuffle=True, output_path="train.tfrecord"):
    bird_labels = tf.convert_to_tensor(bird_labels, dtype=tf.int32)
    bird_filepaths = tf.convert_to_tensor(bird_filepaths, dtype=tf.string)

    # Create TensorFlow datasets from the file paths and labels
    bird_labels_dataset = tf.data.Dataset.from_tensor_slices(bird_labels)
    bird_filepaths_dataset = tf.data.Dataset.from_tensor_slices(bird_filepaths)
    dataset = tf.data.Dataset.zip((bird_filepaths_dataset, bird_labels_dataset))

    # Shuffle the dataset if shuffle is True
    if shuffle:
        dataset = dataset.shuffle(buffer_size=dataset.cardinality(), reshuffle_each_iteration=True)
    
    # Apply the map function to process the audio files and generate spectrograms
    dataset = dataset.map(map_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size=32, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
    
    # Save the dataset to TFRecord format
    create_tfrecord(dataset, output_path)


# Step 9: Create and save training and validation datasets to TFRecord format
train_output_path = "../../data/processed/audios/train.tfrecord"
val_output_path = "../../data/processed/audios/val.tfrecord"

make_and_save_dataset(bird_labels_train, bird_filepaths_train, shuffle=True, output_path=train_output_path)
make_and_save_dataset(bird_labels_val, bird_filepaths_val, shuffle=False, output_path=val_output_path)
