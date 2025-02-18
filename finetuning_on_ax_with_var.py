from datetime import datetime


import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras.src.optimizers import SGD
from keras.src.layers import GlobalAveragePooling2D, Dense

from tensorboard import program
from tensorflow.python.keras import backend

from load_ax_ds import load_ax_ds_chosen_objects, load_ax_ds_splited
from load_ax_ds_with_var import load_ax_ds_chosen_objects_with_var
from model import get_resnet50_var_classification
from parameter_config import *
from load_ds import load_ds_with_variations
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.platform import tf_logging as logging

from reduce_lr_backtrack import ReduceLRBacktrack

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Tensorboard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "logs"])
url = tb.launch()
print(f"Tensorflow listening on {url}")

train_ds = load_ax_ds_chosen_objects_with_var(ax_dir, 1, visualize=True, training=True)
val_ds = load_ax_ds_chosen_objects_with_var(ax_dir, 2, visualize=True, training=False)

model = get_resnet50_var_classification()
# Comment line below to learn from imagenet weights
#model.load_weights("./trainings/training_whole_s/cp-0001.weights.h5")
outputs = model.output
layers = [layer for layer in model.layers if layer.name != "y_var"]
model = Model(inputs=model.input, outputs=layers[-1].output)

train_whole = True
if train_whole:
    for layer in model.layers:
        layer.trainable = True

last_fc_layer = model.get_layer("fc").output
yaw_output = Dense(1, name="y_yaw")(last_fc_layer)
pitch_output = Dense(1, name="y_pitch")(last_fc_layer)
roll_output = Dense(1, name="y_roll")(last_fc_layer)

# Updating the model to include the new regression output layer
model = Model(inputs=model.input, outputs=[layers[-1].output, yaw_output, pitch_output, roll_output])
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy',
                    'y_yaw': 'mean_squared_error',
                    'y_pitch': 'mean_squared_error',
                    'y_roll': 'mean_squared_error'},
              loss_weights={'y_class': 1.0,
                            'y_yaw': 0.33,
                            'y_pitch': 0.33,
                            'y_roll': 0.33},
              metrics={'y_class': 'accuracy'})
model.summary()

ax_checkpoint_path = "trainings/training_objects_var/best_epoch.weights.h5"
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=ax_checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_y_class_accuracy',
    save_freq='epoch',)
reduce_lr_callback = ReduceLRBacktrack(
    best_path=ax_checkpoint_path,
    factor=0.1,
    patience=10,
    verbose=1,
    min_lr=1e-8)
# Tensorboard callback
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)
epochs = 60
history = model.fit(train_ds,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=val_ds,
          callbacks=[cp_callback, tensorboard_callback, reduce_lr_callback],
          verbose=1)

model.load_weights(ax_checkpoint_path)
model.evaluate(val_ds)
