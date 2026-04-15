import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32
VAL_DIR = "dataset_binary/val"

# Load model
model = tf.keras.models.load_model("cnn_scratch_binary.keras")

val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Predictions
preds = model.predict(val_gen)
y_pred = (preds > 0.5).astype(int).flatten()
y_true = val_gen.classes

# Report
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=["non_cancer", "cancer"]
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["non_cancer", "cancer"],
            yticklabels=["non_cancer", "cancer"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Binary Confusion Matrix")
plt.show()