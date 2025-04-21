import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set paths - ensure this matches your directory structure
# base_path = os.path.join("C:\\", "Users", "ramiz", "Downloads", "BreaKHis_v1")
dataset_path = os.path.join("D:\\People work\\Shipra2\\Breast_Cancer_Detection_System_using_DL\\BreaKHis_v1", "histology_slides", "breast")

# Verify directory structure
print("Verifying directory structure...")
benign_path = os.path.join(dataset_path, "benign")
malignant_path = os.path.join(dataset_path, "malignant")
print(f"Benign samples: {len(os.listdir(benign_path)) if os.path.exists(benign_path) else 0}")
print(f"Malignant samples: {len(os.listdir(malignant_path)) if os.path.exists(malignant_path) else 0}")

# Image parameters
img_size = (224, 224)
batch_size = 32

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation generator
val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Evaluate class distribution
def analyze_class_distribution(generator, name):
    class_counts = np.bincount(generator.classes)
    total_samples = sum(class_counts)
    print(f"\n{name} Class Distribution:")
    print(f"Benign: {class_counts[0]} ({class_counts[0]/total_samples:.2%})")
    print(f"Malignant: {class_counts[1]} ({class_counts[1]/total_samples:.2%})")
    return {0: total_samples/(2*class_counts[0]), 1: total_samples/(2*class_counts[1])}

class_weights = analyze_class_distribution(train_generator, "Training")
analyze_class_distribution(val_generator, "Validation")

# Visualize sample images
def visualize_samples(generator):
    x, y = next(generator)
    plt.figure(figsize=(12, 8))
    for i in range(min(9, batch_size)):
        plt.subplot(3, 3, i+1)
        plt.imshow(x[i])
        plt.title(f"Class: {'Benign' if y[i] == 0 else 'Malignant'}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("\nVisualizing training samples...")
visualize_samples(train_generator)

# Build transfer learning model
def build_transfer_model(input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base model layers

    model = Sequential([
        base_model,
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    return model

model = build_transfer_model()
model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights
)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)
print("✅ Model trained and saved as 'best_model.h5'")