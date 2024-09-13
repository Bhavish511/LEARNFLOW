import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score

# Load CSV file
df = pd.read_csv('emotions.csv')

# Define parameters
image_size = (48, 48)  # Resize all images to 48x48 for consistency
data_path = '/images'  # Folder path where emotion folders are located

# Create empty lists for data and labels
X = []
y = []

# Load images and labels
for index, row in df.iterrows():
    folder_id = str(row['ID'])
    folder_path = os.path.join(data_path, folder_id)
    
    if os.path.exists(folder_path):
        for emotion in os.listdir(folder_path):
            emotion_folder_path = os.path.join(folder_path, emotion)
            if os.path.isdir(emotion_folder_path):
                for image_name in os.listdir(emotion_folder_path):
                    image_path = os.path.join(emotion_folder_path, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                    if image is not None:
                        image = cv2.resize(image, image_size)  # Resize image
                        X.append(image)
                        y.append(emotion)  # Emotion as the label

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Normalize image data
X = X / 255.0  # Normalize pixel values between 0 and 1

# Reshape the data for CNN input
X = X.reshape(-1, image_size[0], image_size[1], 1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Regularization

model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, callbacks=[early_stopping])

# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

# Print accuracy score
accuracy = accuracy_score(y_true, y_pred_classes)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# Save the model
model.save('emotion_recognition_model.h5')
