import keras
import tensorflow as tf
from keras import Model

from keras.src.applications.resnet import ResNet50
from keras.src.layers import GlobalAveragePooling2D, Dense

from parameter_config import *
from load_ds import load_ds

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Loading datasets
train_ds = load_ds(train_dir, visualize=True, training=True)
print("Training dataset loaded")
val_ds = load_ds(val_dir, visualize=True)
print("Validation dataset loaded")

# Model setup
input_tensor = keras.Input(shape=input_shape)

base_model = ResNet50(weights="imagenet", classes=classes, include_top=False, input_tensor=input_tensor, input_shape=input_shape)
# for layer in base_model.layers:
#     layer.trainable = False
base_model_ouput = base_model.output


outputs = GlobalAveragePooling2D()(base_model_ouput)
# Adding fully connected layer
outputs = Dense(512, activation='relu')(outputs)
outputs = Dense(classes, activation='softmax')(outputs)
model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer="sgd",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
model.summary()

# Training setup
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True,
    mode='auto',
    save_freq='epoch',)

model.fit(train_ds,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=val_ds,
          callbacks=[cp_callback])
model.summary()

