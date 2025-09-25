from datetime import datetime


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

from confusion_matrix import generate_confusion_matrix, plot_confusion_matrix
from grad_cam import reverse_resnet50_preprocessing, show_grad_cam_examples
from load_ax_ds import load_ax_ds, load_ax_ds_splited, load_ax_chosen_objects_paths_ds, load_ax_ds_chosen_objects
from model import get_resnet50_var_classification
from parameter_config import *
from load_ds import load_ds, load_ds_with_variations, decode_img, decode_img_png

def remap_class_to_imagenet(x, y):
    y_class = y["y_class"]
    new_class = tf.where(y_class == 2, 817,
                 tf.where(y_class == 4, 751,
                 tf.where(y_class == 9, 895,
                 tf.where(y_class == 11, 847,
                 tf.where(y_class == 12, 466, -1)))))
    return x, new_class

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Model setup

input_tensor = keras.Input(shape=input_shape)

model = ResNet50(weights="imagenet", include_top=True, input_tensor=input_tensor,
                      input_shape=input_shape)
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

ax_dir = "../paper_code_strike_with_a_pose_for_win/custom_dataset_v2"
chosen_objects_ds = load_ax_ds_chosen_objects(ax_dir, 2, visualize=True, training=False)
chosen_objects_ds = chosen_objects_ds.map(remap_class_to_imagenet)


for batch in chosen_objects_ds.take(1):
    print(f"Batch type: {type(batch)}")
    print(f"Batch contents: {batch}")
model.evaluate(chosen_objects_ds, batch_size=batch_size)
