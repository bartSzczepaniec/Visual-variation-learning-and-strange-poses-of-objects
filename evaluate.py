import tensorflow as tf
from keras.src.optimizers import SGD

from confusion_matrix import generate_confusion_matrix, plot_confusion_matrix
from grad_cam import show_grad_cam_examples
from model import get_resnet50_var_classification
from parameter_config import *
from load_ds import load_ds

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)

# Model setup
model = get_resnet50_var_classification()

model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss={'y_class': 'sparse_categorical_crossentropy', 'y_var': 'sparse_categorical_crossentropy'},
              loss_weights={'y_class': 1.0 - alpha, 'y_var': alpha},
              metrics={'y_class': 'accuracy', 'y_var': 'accuracy'})
# Set the saved weights file's name below to evaluate the model on test data
model.load_weights("./trainings/training_whole_s/cp-0001.weights.h5")
model.summary()
# Load test dataset
test_ds = load_ds(test_dir, visualize=True, training=False)

# Set the booleans to evaluate the model in a specific way
evaluate_model_accuracy = True
evaluate_confusion_matrix = False
visualize_grad_cam = True
if evaluate_model_accuracy:
    results = model.evaluate(test_ds)
if evaluate_confusion_matrix:
    confusion_matrix = generate_confusion_matrix(model, test_ds, ds_y_type="class_only", model_y_type="variations")
    plot_confusion_matrix(confusion_matrix)
if visualize_grad_cam:
    test_ds = test_ds.shuffle(buffer_size=100)
    for i in range(6):
        show_grad_cam_examples(model, test_ds, ds_y_type="class_only", model_y_type="variations")
