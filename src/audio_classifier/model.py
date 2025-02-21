import tensorflow as tf
from tensorflow.keras import layers, models

# Step 1: Create a function to parse TFRecord file
def parse_tfrecord_fn(example):
    """
    Parse the TFRecord data from the file.
    """
    # Define your `tfrecord` fields
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # The spectrograms (as JPEG)
        'label': tf.io.FixedLenFeature([1], tf.int64),  # The labels
    }

    # Parse the input tf.train.Example proto using the feature description
    parsed_features = tf.io.parse_single_example(example, feature_description)

    # Decode the image
    image = tf.io.decode_jpeg(parsed_features['image'])
    
    # Normalize the image
    image = tf.cast(image, tf.float32) / 255.0

    return image, parsed_features['label']

# Step 2: Create a function to load TFRecord data
def load_tfrecord_dataset(tfrecord_path, batch_size=32):
    # Create the dataset from the TFRecord files
    dataset = tf.data.TFRecordDataset(filenames=[tfrecord_path])
    
    # Parse the dataset
    dataset = dataset.map(parse_tfrecord_fn)
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    
    return dataset

# Step 3: Build the model (ResNet50) with modified input for single-channel
def build_resnet_image_model(input_shape=(1026, 257, 1), num_classes=5, learning_rate=0.0001):
    # Modify the input layer to accept single-channel input (grayscale)
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Fine-tune the model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
    ])
    
    # Use Adam optimizer with a specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the model and plot accuracy/loss
def train_resnet_model(train_tfrecord_path, val_tfrecord_path, epochs=50, batch_size=32):
    # Load the datasets
    train_dataset = load_tfrecord_dataset(train_tfrecord_path, batch_size)
    val_dataset = load_tfrecord_dataset(val_tfrecord_path, batch_size)

    # Build the model
    model = build_resnet_image_model(input_shape=(1026, 257, 1), num_classes=5, learning_rate=0.0001)
    
    # Train the model
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    
    # Save the model
    model.save('resnet_audio_model.h5')
    
    # Step 5: Plot the accuracy and loss graphs
    import matplotlib.pyplot as plt
    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, history

train_tfrecord_path = '../../data/processed/audios/train.tfrecord'
val_tfrecord_path = '../../data/processed/audios/val.tfrecord'

train_resnet_model(train_tfrecord_path, val_tfrecord_path, epochs=5, batch_size=32)
