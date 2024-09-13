import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the CSV file (with gender, age, and country)
metadata = pd.read_csv('emotions.csv')

# Function to encode categorical variables (e.g., gender, country)
def preprocess_metadata(metadata):
    # Encode 'gender' and 'country' into numerical form
    label_encoder_gender = LabelEncoder()
    metadata['gender_encoded'] = label_encoder_gender.fit_transform(metadata['gender'])
    
    label_encoder_country = LabelEncoder()
    metadata['country_encoded'] = label_encoder_country.fit_transform(metadata['country'])
    
    # Standardize age
    scaler = StandardScaler()
    metadata['age_scaled'] = scaler.fit_transform(metadata[['age']])
    
    # Return the processed metadata and the encoders for future use
    return metadata[['gender_encoded', 'age_scaled', 'country_encoded']]

# Preprocess the metadata
metadata_features = preprocess_metadata(metadata)

# Function to load images based on set_id from metadata
def load_images_from_folders(metadata, image_folder):
    images = []
    labels = []
    for index, row in metadata.iterrows():
        folder_id = row['set_id']
        folder_path = os.path.join(image_folder, str(folder_id))
        for image_file in os.listdir(folder_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (64, 64))  # Resize to 64x64
                images.append(image)
                label = image_file.split('_')[0]  # Emotion label from the file name
                labels.append(label)
    return np.array(images), np.array(labels)

# Path to the folder containing the images (set this to your image folder path)
image_folder = 'images'

# Load image data and corresponding labels (emotions)
X_images, y = load_images_from_folders(metadata, image_folder)

# Normalize image data
X_images = X_images / 255.0

# Encode emotion labels into numerical form
label_encoder_emotions = LabelEncoder()
y_encoded = label_encoder_emotions.fit_transform(y)

# Combine image data and metadata features for model input
X_combined = [X_images, metadata_features.to_numpy()]

# Split data into training and test sets
X_train_images, X_test_images, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
    X_images, metadata_features, y_encoded, test_size=0.2, random_state=42)

# Building the model (multi-input CNN + metadata)
def build_model():
    # CNN for image data
    image_input = layers.Input(shape=(64, 64, 3), name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    
    # Dense layers for metadata input
    metadata_input = layers.Input(shape=(3,), name='metadata_input')
    meta_x = layers.Dense(64, activation='relu')(metadata_input)
    
    # Concatenate image and metadata features
    concatenated = layers.concatenate([x, meta_x])
    
    # Final Dense layers
    output = layers.Dense(64, activation='relu')(concatenated)
    output = layers.Dense(len(label_encoder_emotions.classes_), activation='softmax')(output)
    
    # Define the model
    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate and compile the model
model = build_model()

# Train the model
history = model.fit(
    {'image_input': X_train_images, 'metadata_input': X_train_meta},
    y_train,
    epochs=10,
    validation_data=({'image_input': X_test_images, 'metadata_input': X_test_meta}, y_test)
)

# Evaluate the model
test_loss, test_acc = model.evaluate({'image_input': X_test_images, 'metadata_input': X_test_meta}, y_test)
print(f"Test Accuracy: {test_acc}")

# Predictions and classification report
y_pred = np.argmax(model.predict({'image_input': X_test_images, 'metadata_input': X_test_meta}), axis=1)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder_emotions.classes_))
