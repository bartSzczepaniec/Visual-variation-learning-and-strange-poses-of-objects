import tensorflow as tf
from keras import Model
from keras.src.optimizers import SGD



from confusion_matrix import generate_confusion_matrix, plot_confusion_matrix, plot_reduced_confusion_matrix
from grad_cam import reverse_resnet50_preprocessing, show_grad_cam_examples
from load_ax_ds import load_ax_ds, load_ax_ds_splited, load_ax_chosen_objects_paths_ds, load_ax_ds_chosen_objects
from model import get_resnet50_var_classification
from parameter_config import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Model setup
model = get_resnet50_var_classification()
model.summary()
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy', 'y_var': 'sparse_categorical_crossentropy'},
              loss_weights={'y_class': 1.0 - alpha, 'y_var': alpha},
              metrics={'y_class': 'accuracy', 'y_var': 'accuracy'})

use_ilab_weights=False
if use_ilab_weights:
    saved_weights_path = "./trainings/training_whole_s/cp-0001.weights.h5"
    model.load_weights(saved_weights_path)
outputs = model.output
layers = [layer for layer in model.layers if layer.name != "y_var"]
model = Model(inputs=model.input, outputs=layers[-1].output)
if not use_ilab_weights:
    model.load_weights("trainings/training_objects/best_epoch_dataset_v3_run2.weights.h5")
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy'},
              metrics={'y_class': 'accuracy'})

model.summary()

# Set the booleans to evaluate the model in a specific way
evaluate_model_accuracy = False
evaluate_confusion_matrix = True
visualize_grad_cam = False
ax_dir = "../paper_code_strike_with_a_pose_for_win/custom_dataset_v3"
chosen_objects_ds = load_ax_ds_chosen_objects(ax_dir, 2, visualize=True, training=False)
if evaluate_model_accuracy:
    model.evaluate(chosen_objects_ds, batch_size=batch_size)
if evaluate_confusion_matrix:
    confusion_matrix = generate_confusion_matrix(model, chosen_objects_ds, ds_y_type="variations", model_y_type="no_vars")
    plot_confusion_matrix(confusion_matrix)
    plot_reduced_confusion_matrix(confusion_matrix)
if visualize_grad_cam:
    chosen_objects_ds = chosen_objects_ds.shuffle(buffer_size=100, seed=5)
    show_grad_cam_examples(model, chosen_objects_ds, ds_y_type="variations", model_y_type="no_vars")
