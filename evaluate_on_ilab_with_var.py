import tensorflow as tf
from keras import Model
from keras.src.optimizers import SGD
from keras.src.layers import Dense

from tensorboard import program

from load_ax_ds import load_ax_ds_chosen_objects, load_ax_ds_splited
from load_ax_ds_with_var import load_ax_ds_chosen_objects_with_var
from model import get_resnet50_var_classification
from parameter_config import *
from load_ds import load_ds_with_variations, load_ds_with_variations_regr

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Tensorboard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "logs"])
url = tb.launch()
print(f"Tensorflow listening on {url}")

# Loading dataset
test_dir = load_ds_with_variations_regr(test_dir, visualize=True)
print("Test dataset loaded")
# Model setup
model = get_resnet50_var_classification()

outputs = model.output
layers = [layer for layer in model.layers if layer.name != "y_var"]
model = Model(inputs=model.input, outputs=layers[-1].output)

train_whole = True
if train_whole:
    for layer in model.layers:
        layer.trainable = True

last_fc_layer = model.get_layer("fc").output
elevation_output = Dense(1, name="y_roll")(last_fc_layer)
azimuth_output = Dense(1, name="y_yaw")(last_fc_layer)
# Updating the model to include the new regression output layer
model = Model(inputs=model.input, outputs=[layers[-1].output, elevation_output, azimuth_output])
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy',
                    'y_roll': 'mean_squared_error',
                    'y_yaw': 'mean_squared_error'},
              loss_weights={'y_class': 1.0,
                            'y_roll': 0.5,
                            'y_yaw': 0.5},
              metrics={'y_class': 'accuracy'})
model.summary()

ax_checkpoint_path = "trainings/training_ilab_regression/best_epoch.weights.h5"

model.load_weights(ax_checkpoint_path)
model.evaluate(test_dir)
