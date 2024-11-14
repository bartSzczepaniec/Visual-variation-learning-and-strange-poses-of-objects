from datetime import datetime
from tabnanny import verbose

import keras
import tensorflow as tf
from keras import Model

from keras.src.applications.resnet import ResNet50
from keras.src.layers import GlobalAveragePooling2D, Dense
from tensorboard import program

from parameter_config import *
from load_ds import load_ds, load_ds_with_variations

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
input_tensor = keras.Input(shape=input_shape)

base_model = ResNet50(weights="imagenet", classes=classes, include_top=False, input_tensor=input_tensor, input_shape=input_shape)
# for layer in base_model.layers:
#     layer.trainable = False
base_model_ouput = base_model.output


outputs = GlobalAveragePooling2D()(base_model_ouput)
# Adding fully connected layer
outputs = Dense(1024, activation='relu', name="fc")(outputs)

outputs_standard = Dense(classes, activation='softmax', name='y_class')(outputs)
outputs_var = Dense(variations, activation='softmax', name='y_var')(outputs)

model = Model(inputs=base_model.input, outputs=[outputs_standard, outputs_var])
model.compile(optimizer=optimizer,
              loss={'y_class': 'sparse_categorical_crossentropy', 'y_var': 'sparse_categorical_crossentropy'},
              loss_weights={'y_class': 1.0 - alpha, 'y_var': alpha},
              metrics={'y_class': 'accuracy', 'y_var': 'accuracy'})
model.summary()

# test_model = ResNet50(weights="imagenet", classes=1000, include_top=True, input_tensor=input_tensor, input_shape=input_shape)
# test_model.summary()

# Training setup
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    monitor='val_y_class_accuracy',
    save_best_only=True,
    mode='auto',
    save_freq='epoch',)

# Tensorboard callback
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)

history = model.fit(train_ds,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=val_ds,
          callbacks=[cp_callback, tensorboard_callback],
          verbose=1)

print(history.history.keys())
print("CLASS accuracy:")
print(history.history["val_y_class_accuracy"])
print("VARIATION accuracy:")
print(history.history["val_y_var_accuracy"])