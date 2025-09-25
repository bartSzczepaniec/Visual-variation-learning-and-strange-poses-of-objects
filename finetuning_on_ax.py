from datetime import datetime


import keras
import tensorflow as tf
from keras import Model
from keras.src.optimizers import SGD, Adam

from tensorboard import program

from load_ax_ds import load_ax_ds_chosen_objects, load_ax_ds_splited
from model import get_resnet50_var_classification
from parameter_config import *
from load_ds import load_ds_with_variations

from reduce_lr_backtrack import ReduceLRBacktrack

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

FINE_TUNING_TYPE = "OBJECTS" # "SPLIT" OR "OBJECTS"

# Tensorboard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "logs"])
url = tb.launch()
print(f"Tensorflow listening on {url}")

# Set the dataset location
ax_smaller_ds_dir = "../paper_code_strike_with_a_pose_for_win/custom_dataset_v2"
# Loading datasets
train_ds = None
if FINE_TUNING_TYPE == "OBJECTS":
    train_ds = load_ax_ds_chosen_objects(ax_dir, 1, visualize=True, training=True)
if FINE_TUNING_TYPE == "SPLITS":
    train_ds = load_ax_ds_splited(ax_dir, visualize=True, training=True)
print("Training dataset loaded")
val_ds = None
if FINE_TUNING_TYPE == "OBJECTS":
    val_ds = load_ax_ds_chosen_objects(ax_dir, 2, visualize=True, training=False)
if FINE_TUNING_TYPE == "SPLITS":
    val_ds = load_ax_ds_splited(ax_dir, visualize=True, training=False)
print("Validation dataset loaded")

model = get_resnet50_var_classification()
load_custom_weights=False
if load_custom_weights:
    model.load_weights("./trainings/training_whole_s/cp-0001.weights.h5")
outputs = model.output
layers = [layer for layer in model.layers if layer.name != "y_var"]
train_whole_model = True
if train_whole_model:
    for layer in model.layers:
        layer.trainable = True
model = Model(inputs=model.input, outputs=layers[-1].output)
model.compile(optimizer=Adam(learning_rate=0.0001), #SGD(learning_rate=0.001, momentum=0.9), # Adam(learning_rate=0.0001)
              loss={'y_class': 'sparse_categorical_crossentropy'},
              metrics={'y_class': 'accuracy'})
model.summary()

# Training setup
ax_checkpoint_path = ""
if FINE_TUNING_TYPE == "OBJECTS":
    ax_checkpoint_path = "trainings/training_objects/best_epoch.weights.h5"
if FINE_TUNING_TYPE == "SPLITS":
    ax_checkpoint_path = "trainings/training_on_split/best_epoch.weights.h5"
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=ax_checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy',
    save_freq='epoch',)
reduce_lr_callback = ReduceLRBacktrack(
    best_path=ax_checkpoint_path,
    factor=0.1,
    patience=10,
    verbose=1,
    min_lr=1e-9)
# Tensorboard callback
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)
epochs = 40
history = model.fit(train_ds,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=val_ds,
          callbacks=[cp_callback, tensorboard_callback, reduce_lr_callback],
          verbose=1)

print(history.history.keys())
print("CLASS accuracy:")
print(history.history["val_accuracy"])
print("CLASS loss:")
print(history.history["val_loss"])

model.load_weights(ax_checkpoint_path)
model.evaluate(val_ds)

