import tensorflow as tf
from keras import Model
from keras.src.optimizers import SGD
from keras.src.layers import Dense


from confusion_matrix import generate_confusion_matrix, plot_confusion_matrix, plot_reduced_confusion_matrix
from grad_cam import show_grad_cam_examples
from load_ax_ds import load_ax_ds, load_ax_ds_splited, load_ax_chosen_objects_paths_ds, load_ax_ds_chosen_objects
from load_ax_ds_with_var import load_ax_ds_chosen_objects_with_var
from model import get_resnet50_var_classification
from parameter_config import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Model setup
model = get_resnet50_var_classification()
# model.load_weights("./trainings/training_whole_s/cp-0001.weights.h5")
outputs = model.output
layers = [layer for layer in model.layers if layer.name != "y_var"]
model = Model(inputs=model.input, outputs=layers[-1].output)
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
model.load_weights("trainings/training_objects_var/best_epoch_dataset_v3_run2.weights.h5")
model.summary()

# Set the booleans to evaluate the model in a specific way
evaluate_model_accuracy = False
evaluate_confusion_matrix = True
visualize_grad_cam = False
ax_dir = "../paper_code_strike_with_a_pose_for_win/custom_dataset_v3"
chosen_objects_ds = load_ax_ds_chosen_objects_with_var(ax_dir, 2, visualize=True, training=False)
if evaluate_model_accuracy:
    model.evaluate(chosen_objects_ds, batch_size=batch_size)
if evaluate_confusion_matrix:
    confusion_matrix = generate_confusion_matrix(model, chosen_objects_ds, ds_y_type="variations", model_y_type="variations")
    plot_confusion_matrix(confusion_matrix)
    plot_reduced_confusion_matrix(confusion_matrix)
if visualize_grad_cam:
    chosen_objects_ds = chosen_objects_ds.shuffle(buffer_size=100, seed=3)
    for i in range(3):
        show_grad_cam_examples(model, chosen_objects_ds, ds_y_type="variations", model_y_type="variations")


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
