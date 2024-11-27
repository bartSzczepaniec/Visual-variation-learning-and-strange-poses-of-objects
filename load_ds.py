import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

from tensorflow.python.data import AUTOTUNE
from parameter_config import *


def get_label(file_path):
  path_parts = tf.strings.split(file_path, os.path.sep)
  file_name = path_parts[-1]
  file_name_parts = tf.strings.split(file_name, "-")
  class_name = file_name_parts[0]
  one_hot = class_name == class_names
  # Encoding label in one hot encoding way
  return tf.argmax(one_hot)

def decode_img(img):
  img = tf.io.decode_jpeg(img, channels=3)
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def get_all_class_names(ds_dir):
  all_files = glob.glob(f"{ds_dir}/*")
  class_names = []
  for file in all_files:
    path_parts = file.split("/")
    file_name = path_parts[-1]
    file_name_parts = file_name.split("-")
    class_name = file_name_parts[0]
    if class_name not in class_names:
      class_names.append(class_name)
  return class_names

def configure_for_performance(ds):
  #ds = ds.cache()
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def load_paths_ds(ds_dir):
    file_list = [os.path.join(ds_dir, x) for x in os.listdir(ds_dir)]
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.shuffle(buffer_size=len(file_list), reshuffle_each_iteration=False)
    return ds

def crop_images(img, label, training):
    if training:
        img = tf.image.random_crop(img, size=input_shape)  # Randomly crop to 224x224
    else:
        img = tf.image.central_crop(img, central_fraction=0.875)
    return img, label

def load_ds(ds_dir, visualize: bool = False, training: bool = False):
    ds = load_paths_ds(ds_dir)
    # FOR TESTING - remove this line
    # ds = ds.take(1000)

    # Slower version below
    # ds = tf.data.Dataset.list_files(str(ds_dir + '/*'), shuffle=False)
    # print("dataset files listed")

    ds = ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("dataset loaded")
    ds = ds.map(lambda x, y: crop_images(x, y, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = configure_for_performance(ds)
    # Preprocessing for the ResNet50 model
    ds = ds.map(lambda x, y: (keras.applications.resnet50.preprocess_input(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("dataset preprocessed")

    if visualize:
        image_batch, label_batch = next(iter(ds))
        # exemplary data in batch visualisation
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch[i]
            plt.title(class_names[label])
            plt.axis("off")
        plt.show()
    return ds


def get_labels(file_path):
    path_parts = tf.strings.split(file_path, os.path.sep)
    file_name = path_parts[-1]
    file_name_parts = tf.strings.split(file_name, ".")
    file_name_parts = tf.strings.split(file_name_parts[0], "-")
    class_name = file_name_parts[0]
    var_name = tf.strings.join([file_name_parts[-4], file_name_parts[-3], file_name_parts[-2], file_name_parts[-1]], separator="-")
    one_hot = class_name == class_names
    one_hot_var = var_name == variations_names
    # tf.print(tf.argmax(one_hot_var))
    # Encoding label in one hot encoding way
    return tf.argmax(one_hot), tf.argmax(one_hot_var)

def process_path_with_variations(file_path):
  label, label_var = get_labels(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label, label_var

def crop_images_with_variations(img, label, label_var, training):
    if training:
        img = tf.image.random_crop(img, size=input_shape)  # Randomly crop to 224x224
    else:
        img = tf.image.central_crop(img, central_fraction=0.875)
    return img, label, label_var

def load_ds_with_variations(ds_dir, visualize: bool = False, training: bool = False):
    ds = load_paths_ds(ds_dir)
    # FOR TESTING - remove this line
    # ds = ds.take(1000)


    ds = ds.map(process_path_with_variations, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("dataset with variations loaded")
    ds = ds.map(lambda x, y, y_var: crop_images_with_variations(x, y, y_var, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = configure_for_performance(ds)
    # Preprocessing for the ResNet50 model
    ds = ds.map(lambda x, y, y_var: (keras.applications.resnet50.preprocess_input(x), {'y_class': y, 'y_var': y_var}), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("dataset with variations preprocessed")

    if visualize:
        image_batch, label_batch = next(iter(ds))
        # exemplary data in batch visualisation
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch["y_class"][i]
            label_var = label_batch["y_var"][i]
            plt.title(class_names[label] + "-" + variations_names[label_var])
            plt.axis("off")
        plt.show()
    return ds