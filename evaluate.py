from datetime import datetime


import keras
import tensorflow as tf
from keras import Model

from keras.src.applications.resnet import ResNet50
from keras.src.layers import GlobalAveragePooling2D, Dense
from tensorboard import program

from model import get_resnet50_var_classification
from parameter_config import *
from load_ds import load_ds, load_ds_with_variations

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Model setup
model = get_resnet50_var_classification()

model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy', 'y_var': 'sparse_categorical_crossentropy'},
              loss_weights={'y_class': 1.0 - alpha, 'y_var': alpha},
              metrics={'y_class': 'accuracy', 'y_var': 'accuracy'})
model.load_weights(saved_weights_path)
model.summary()
# Load test dataset
test_ds = load_ds(test_dir, visualize=True, training=False)

results = model.evaluate(test_ds)
