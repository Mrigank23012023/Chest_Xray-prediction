import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow import keras

# Data Augmentation Layer
data_augmentation = keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Load training set and validation set without data augmentation
training_set = keras.utils.image_dataset_from_directory(
    directory='/content/drive/MyDrive/Lung_Disease_Dataset/2gb data/Lung Disease Dataset/train',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True
)

val_set = keras.utils.image_dataset_from_directory(
    directory='/content/drive/MyDrive/Lung_Disease_Dataset/2gb data/Lung Disease Dataset/val',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True
)

# Checking shape of one batch
for x, y in training_set:
    print(x.shape, y.shape)
    break

# Transfer Learning with VGG16 as base model
base_model = tf.keras.applications.VGG16(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model layers

# Define the model using Functional API
inputs = tf.keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)  # Apply data augmentation
x = base_model(x)  # Pass the augmented images to the base model
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Global Average Pooling
x = tf.keras.layers.Dense(1500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)  # Increased Dropout for regularization
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)  # Output layer

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Class weights for handling class imbalance (example weights, adjust based on your dataset)
class_weights = {0: 1., 1: 2., 2: 1.5, 3: 2., 4: 1.}

# Learning Rate Scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20))

# Early Stopping Callback with adjusted patience
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,  # Stop after 5 epochs with no improvement
    restore_best_weights=True
)

# Train the model with learning rate scheduler, early stopping, and class weights
history = model.fit(
    training_set,
    validation_data=val_set,
    epochs=25,  # Train for 25 epochs
    class_weight=class_weights,
    callbacks=[lr_schedule, early_stopping]
)

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Compile again with a smaller learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model for a few more epochs
history_fine = model.fit(
    training_set,
    validation_data=val_set,
    epochs=25,  # Fine-tuning for 25 additional epochs
    class_weight=class_weights,
    callbacks=[early_stopping]  # Include early stopping in fine-tuning
)

# Save the model
model.save('/content/drive/MyDrive/Lung_Disease_Dataset/chest_xray_vgg16_finetuned.h5')

# Plotting accuracy and loss
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'])
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'])
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()