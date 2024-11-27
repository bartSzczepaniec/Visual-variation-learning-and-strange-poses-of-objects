import keras
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
)

# Load some data
(x_train, y_train), _ = keras.datasets.cifar10.load_data()
input_shape = x_train.shape[1:]
classes = 10

# Create a tf.data pipeline of augmented images (and their labels)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(64).map(lambda x, y: (data_augmentation(x), y))


# Create a model and train it on the augmented image data
inputs = keras.Input(shape=input_shape)
x = keras.layers.Rescaling(1.0 / 255)(inputs)  # Rescale inputs
outputs = keras.applications.ResNet50(  # Add the rest of the model
    weights=None, input_shape=input_shape, classes=classes
)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=['categorical_accuracy'])
model.fit(train_dataset, epochs=5, validation_data=train_dataset)