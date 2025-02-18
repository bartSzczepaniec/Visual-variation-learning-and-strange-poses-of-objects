import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras.src.optimizers import SGD
from keras.src.applications.resnet import ResNet50
from keras.src.layers import GlobalAveragePooling2D, Dense
from tensorboard import program
from tensorflow.python.data import AUTOTUNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

from parameter_config import class_names


def generate_confusion_matrix(model, dataset, batch_size=32, ds_y_type="variations", model_y_type="variations"):
    if batch_size != 32:
        dataset = dataset.unbatch().batch(10)

    y_true_np = []
    y_pred_np = []

    for x_batch, y_batch in dataset:
        y_true = y_batch["y_class"].numpy() if ds_y_type == "variations" else y_batch.numpy()
        y_true_np.extend(y_true)

        y_pred = model.predict(x_batch, verbose=0)
        if model_y_type == "variations":
            y_pred = y_pred[0]

        y_pred_np.extend(np.argmax(y_pred, axis=1))

    y_true_np = np.array(y_true_np)
    y_pred_np = np.array(y_pred_np)
    return confusion_matrix(y_true_np, y_pred_np, labels=range(len(class_names)))

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(10, 10))
    sn.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    # Add labels, title
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
