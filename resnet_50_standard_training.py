import keras
import tensorflow as tf

from keras.src.applications.resnet import ResNet50
from parameter_config import *
from load_ds import load_ds

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Loading datasets
train_ds = load_ds(train_dir, visualize=True)
print("Training dataset loaded")
val_ds = load_ds(val_dir, visualize=True)
print("Validation dataset loaded")

# Model setup
input_shape = (img_width, img_height, 3)
input_tensor = keras.Input(shape=input_shape)

model = ResNet50(weights=None, classes=classes, include_top=True, input_tensor=input_tensor, input_shape=input_shape)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['categorical_accuracy'])

# Training setup
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    monitor='val_categorical_accuracy',
    save_best_only=True,
    mode='auto',
    save_freq='epoch',)

model.fit(train_ds,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=val_ds,
          callbacks=[cp_callback])
model.summary()

