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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

#Model setup
model = get_resnet50_var_classification()
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy', 'y_var': 'sparse_categorical_crossentropy'},
              loss_weights={'y_class': 1.0 - alpha, 'y_var': alpha},
              metrics={'y_class': 'accuracy', 'y_var': 'accuracy'})
#model.load_weights(saved_weights_path)
outputs = model.output
layers = [layer for layer in model.layers if layer.name != "y_var"]
model = Model(inputs=model.input, outputs=layers[-1].output)
model.load_weights("trainings/training_objects/best_epoch_from_imagenet_v3.weights.h5")
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy'},
              metrics={'y_class': 'accuracy'})
model.summary()

# Load test dataset
#train_ds = load_ds_with_variations(test_dir, visualize=True)

# ax_ds = load_ax_ds(ax_dir, visualize=True, training=False)
# #ax_ds = ax_ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False, seed=0)
# model.evaluate(ax_ds)

# test_ds = load_ax_ds_splited(ax_dir, visualize=True, training=False)
#
# model.evaluate(test_ds, batch_size=batch_size)

evaluate_model_accuracy = True
evaluate_confusion_matrix = False
visualize_grad_cam = False

chosen_objects_ds = load_ax_ds_chosen_objects(ax_dir, 2, visualize=True, training=False)
if evaluate_model_accuracy:
    model.evaluate(chosen_objects_ds, batch_size=batch_size)
if evaluate_confusion_matrix:
    confusion_matrix = generate_confusion_matrix(model, chosen_objects_ds, ds_y_type="variations", model_y_type="no_vars")
    plot_confusion_matrix(confusion_matrix)
if visualize_grad_cam:
    chosen_objects_ds = chosen_objects_ds.shuffle(buffer_size=shuffle_buffer_size)
    show_grad_cam_examples(model, chosen_objects_ds, ds_y_type="variations", model_y_type="no_vars")


# img = tf.io.read_file("ax_examples/pose_2.png")
# #img = tf.io.read_file("ax_examples/bus-i1224-b0061-c04-r04-l0-f2.jpg")
# #img = decode_img(img)
# img = decode_img_png(img)
# img = tf.image.central_crop(img, central_fraction=0.875)
# img = tf.expand_dims(img, axis=0)
#
# pred=model.predict(img)
# print(pred[0])
# pr_cl = np.argmax(pred[0], axis=1)
# print(class_names[pr_cl[0]])
# # results = model.predict(test_ds)
