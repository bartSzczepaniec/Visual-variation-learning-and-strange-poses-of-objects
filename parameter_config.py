import os

from utils import get_variations_names, get_ax_variations_names

# dataset params
classes = 15
img_width = 256
img_height = 256
class_names = ['boat', 'bus', 'car', 'equip', 'f1car', 'heli', 'mil', 'monster', 'pickup', 'plane', 'semi', 'tank', 'train', 'ufo', 'van']
variations_names = get_variations_names()
variations = len(variations_names) # number of all poses
ax_variations_names = get_ax_variations_names()
ax_variations = len(ax_variations_names) # number of all poses

shuffle_buffer_size = 1000
# dirs
train_dir = "../iLab-2M/home2/toy/iLab2M/train_img"
val_dir = "../iLab-2M/home2/toy/iLab2M/val_img"
test_dir = "../iLab-2M/home2/toy/iLab2M/test_img"
ax_dir = "../paper_code_strike_with_a_pose_for_win/custom_dataset_v3"
# weights path
saved_weights_for_future_train_path = "./trainings/training_whole_f/cp-0014.weights.h5"
saved_weights_path = "./trainings/training_objects/best_epoch.weights.h5"

checkpoint_path = "trainings/training_1/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# training params
batch_size = 32
epochs = 60

# model params
# From the paper: SGD optimizer, lr=0.001, alpha=0.8, momentum=0.9, batch=16
input_shape = (224, 224, 3)
# optimizer = "sgd"
alpha = 0.8
