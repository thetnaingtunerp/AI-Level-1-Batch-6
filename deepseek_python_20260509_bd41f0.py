# ============================================
# CELL 1: Import libraries and check TensorFlow version
# ============================================
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# ============================================
# CELL 2: Setup data generators
# ============================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths (UPDATE THESE PATHS!)
train_dir = 'path/to/your/train/directory'  # CHANGE THIS
validation_dir = 'path/to/your/validation/directory'  # CHANGE THIS
test_dir = 'path/to/your/test/directory'  # CHANGE THIS

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get class labels
class_labels = list(train_generator.class_indices.keys())
print(f"Class labels: {class_labels}")

# ============================================
# CELL 3: Print length of train_generator
# ============================================
print(f"Length of train_generator: {len(train_generator)}")
print(f"Number of training samples: {train_generator.samples}")
print(f"Number of classes: {train_generator.num_classes}")

# ============================================
# CELL 4: Create Extract Features Model (Transfer Learning)
# ============================================
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Load pre-trained base model (without top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create extract features model
extract_feat_model = Model(inputs=base_model.input, outputs=predictions)

# ============================================
# CELL 5: Print model summary
# ============================================
print("Extract Features Model Summary:")
extract_feat_model.summary()

# ============================================
# CELL 6: Compile the extract features model
# ============================================
extract_feat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# CELL 7: Train the extract features model
# ============================================
history_extract = extract_feat_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1
)

# ============================================
# CELL 8: Create Fine-Tuned Model
# ============================================
# Start from the trained extract features model
fine_tuned_model = extract_feat_model

# Unfreeze some layers for fine-tuning
base_model.trainable = True

# Freeze all layers except the last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
fine_tuned_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# CELL 9: Train the fine-tuned model
# ============================================
history_finetune = fine_tuned_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1
)

# ============================================
# CELL 10: Plot accuracy curves for extract features model
# ============================================
import matplotlib.pyplot as plt

def plot_accuracy(history, model_name="Model"):
    """Plot training and validation accuracy"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_accuracy(history_extract, "Extract Features Model")

# ============================================
# CELL 11: Plot loss curves for fine-tuned model
# ============================================
def plot_loss(history, model_name="Model"):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history_finetune, "Fine-Tuned Model")

# ============================================
# CELL 12: Plot accuracy curves for fine-tuned model
# ============================================
plot_accuracy(history_finetune, "Fine-Tuned Model")

# ============================================
# CELL 13: Plot test image using Extract Features Model (index_to_plot = 1)
# ============================================
import numpy as np

def plot_test_image_with_prediction(generator, model, index_to_plot, class_labels, model_name=""):
    """Plot a test image with model prediction"""
    # Reset generator
    generator.reset()
    
    # Get the batch containing the image
    batch_idx = index_to_plot // generator.batch_size
    img_index = index_to_plot % generator.batch_size
    
    images, true_labels = generator[batch_idx]
    
    image = images[img_index]
    true_label_idx = np.argmax(true_labels[img_index])
    
    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
    pred_label_idx = np.argmax(prediction[0])
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'{model_name}\nTrue: {class_labels[true_label_idx]}\nPredicted: {class_labels[pred_label_idx]}')
    plt.axis('off')
    plt.show()
    
    return true_label_idx, pred_label_idx

# Plot test image index 1
plot_test_image_with_prediction(
    test_generator, 
    extract_feat_model, 
    index_to_plot=1, 
    class_labels=class_labels,
    model_name="Extract Features Model"
)

# ============================================
# CELL 14: Plot test image using Fine-Tuned Model (index_to_plot = 1)
# ============================================
plot_test_image_with_prediction(
    test_generator, 
    fine_tuned_model, 
    index_to_plot=1, 
    class_labels=class_labels,
    model_name="Fine-Tuned Model"
)

# ============================================
# CELL 15: Evaluate both models on test set (optional)
# ============================================
print("\n=== Extract Features Model Test Evaluation ===")
extract_loss, extract_acc = extract_feat_model.evaluate(test_generator, verbose=0)
print(f"Test Loss: {extract_loss:.4f}")
print(f"Test Accuracy: {extract_acc:.4f}")

print("\n=== Fine-Tuned Model Test Evaluation ===")
fine_loss, fine_acc = fine_tuned_model.evaluate(test_generator, verbose=0)
print(f"Test Loss: {fine_loss:.4f}")
print(f"Test Accuracy: {fine_acc:.4f}")