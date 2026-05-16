# ============================================
# QUESTION 1: Final Assignment Jupyter Notebook
# ============================================

# 1.1: Print the version of TensorFlow
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# 1.2: Create a test_generator using the test_datagen object
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have test_datagen defined similar to this:
test_datagen = ImageDataGenerator(rescale=1./255)

# Create test_generator (adjust parameters as needed)
test_generator = test_datagen.flow_from_directory(
    directory='path/to/test/directory',  # REPLACE with your test path
    target_size=(224, 224),  # REPLACE with your image size
    batch_size=32,
    class_mode='categorical',  # or 'binary' depending on your problem
    shuffle=False
)

# 1.3: Print the length of the train_generator
# Assuming train_generator already exists in your notebook
print(f"Length of train_generator: {len(train_generator)}")

# 1.4: Print the summary of the model
# Assuming your model is named 'model' or 'extract_feat_model'
# For extract_feat_model:
print("Extract Features Model Summary:")
extract_feat_model.summary()

# If you have a fine_tuned_model:
print("\nFine-Tuned Model Summary:")
fine_tuned_model.summary()

# 1.5: Compile the model
# Replace with your actual model compilation
extract_feat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',  # or 'binary_crossentropy'
    metrics=['accuracy']
)

# 1.6: Plot accuracy curves for training and validation sets (extract_feat_model)
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

# Assuming you have history_extract from training extract_feat_model
plot_accuracy(history_extract, "Extract Features Model")

# 1.7: Plot loss curves for training and validation sets (fine tune model)
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

# Assuming you have history_finetune from training fine_tuned_model
plot_loss(history_finetune, "Fine-Tuned Model")

# 1.8: Plot accuracy curves for training and validation sets (fine tune model)
plot_accuracy(history_finetune, "Fine-Tuned Model")

# 1.9: Plot a test image using Extract Features Model (index_to_plot = 1)
import numpy as np

def plot_test_image_with_prediction(generator, model, index_to_plot, class_labels, model_name=""):
    """Plot a test image with model prediction"""
    # Get the batch containing the image
    generator.reset()  # Reset generator to ensure consistent indexing
    images, true_labels = generator[index_to_plot // generator.batch_size]
    img_index = index_to_plot % generator.batch_size
    
    image = images[img_index]
    true_label = np.argmax(true_labels[img_index]) if len(true_labels.shape) > 1 else true_labels[img_index]
    
    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    pred_label = np.argmax(prediction[0]) if len(prediction.shape) > 1 else (1 if prediction[0][0] > 0.5 else 0)
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'{model_name}\nTrue: {class_labels[true_label]}\nPredicted: {class_labels[pred_label]}')
    plt.axis('off')
    plt.show()
    
    return true_label, pred_label

# Define your class labels (REPLACE with your actual class names)
class_labels = ['class_0', 'class_1']  # Example - update this!

# Plot test image index 1 using extract features model
plot_test_image_with_prediction(
    test_generator, 
    extract_feat_model, 
    index_to_plot=1, 
    class_labels=class_labels,
    model_name="Extract Features Model"
)

# 1.10: Plot a test image using Fine-Tuned Model (index_to_plot = 1)
plot_test_image_with_prediction(
    test_generator, 
    fine_tuned_model, 
    index_to_plot=1, 
    class_labels=class_labels,
    model_name="Fine-Tuned Model"
)