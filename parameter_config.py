import os

# dataset params
classes = 15
img_width = 256
img_height = 256
class_names = ['boat', 'bus', 'car', 'equip', 'f1car', 'heli', 'mil', 'monster', 'pickup', 'plane', 'semi', 'tank', 'train', 'ufo', 'van']
# dirs
train_dir = "../iLab-2M/home2/toy/iLab2M/train_img"
val_dir = "../iLab-2M/home2/toy/iLab2M/val_img"
test_dir = "../iLab-2M/home2/toy/iLab2M/test_img"

checkpoint_path = "trainings/training_1/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# training params
batch_size = 32
epochs = 10