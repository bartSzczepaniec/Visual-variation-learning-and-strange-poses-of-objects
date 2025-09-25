from datetime import datetime


import keras
import tensorflow as tf
from keras.src.optimizers import SGD

from tensorboard import program

from model import get_resnet50_var_classification
from parameter_config import *
from load_ds import load_ds_with_variations

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Tensorboard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "logs"])
url = tb.launch()
print(f"Tensorflow listening on {url}")

# Loading datasets
train_ds = load_ds_with_variations(train_dir, visualize=True, training=True)
print("Training dataset loaded")
val_ds = load_ds_with_variations(val_dir, visualize=True)
print("Validation dataset loaded")

# Model setup
TRAIN_WHOLE_MODEL = False # if False - only the head of the model is trained
model = get_resnet50_var_classification()
if TRAIN_WHOLE_MODEL:
    for layer in model.layers:
        layer.trainable = True
    model.load_weights(saved_weights_for_future_train_path)
# for layer in model.layers:
#     layer.trainable = True
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy', 'y_var': 'sparse_categorical_crossentropy'},
              loss_weights={'y_class': 1.0 - alpha, 'y_var': alpha},
              metrics={'y_class': 'accuracy', 'y_var': 'accuracy'})
model.summary()

# Training setup
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    monitor='val_y_class_accuracy',
    save_freq='epoch',)
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    factor=0.1,
    patience=5,
    verbose=1,
    min_lr=1e-8)
# Tensorboard callback
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)

history = model.fit(train_ds,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=val_ds,
          callbacks=[cp_callback, tensorboard_callback, reduce_lr_callback],
          verbose=1)

print(history.history.keys())
print("CLASS accuracy:")
print(history.history["val_y_class_accuracy"])
print("CLASS loss:")
print(history.history["val_y_class_loss"])
print("VARIATION accuracy:")
print(history.history["val_y_var_accuracy"])

model.save("./saved_model/after_whole_training_model.keras")
