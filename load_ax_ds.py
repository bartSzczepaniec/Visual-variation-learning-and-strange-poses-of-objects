import parameter_config
from load_ds import *
from parameter_config import ax_variations_names
def load_ax_paths_ds(ds_dir):
    file_list = [os.path.join(ds_dir, x) for x in os.listdir(ds_dir)]
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.shuffle(buffer_size=len(file_list), reshuffle_each_iteration=False, seed=0)
    return ds

def get_ax_labels(file_path):
    path_parts = tf.strings.split(file_path, os.path.sep)
    file_name = path_parts[-1]
    file_name_parts = tf.strings.split(file_name, ".")
    file_name_parts = tf.strings.split(file_name_parts[0], "_")
    class_name = file_name_parts[0]
    var_name = tf.strings.join([file_name_parts[3], file_name_parts[4], file_name_parts[5]], separator="-")
    one_hot = class_name == class_names
    #one_hot_var = var_name == ax_variations_names

    # tf.print(tf.argmax(one_hot_var))
    # Encoding label in one hot encoding way
    return tf.argmax(one_hot), tf.argmax([1])

def process_ax_path_with_variations(file_path):
  label, label_var = get_ax_labels(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label, label_var

def configure_ax_for_performance(ds, training):
  #ds = ds.cache()
  if training:
      ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def load_ax_ds(ds_dir, visualize: bool = False, training: bool = False):
    ds = load_ax_paths_ds(ds_dir)
    print("dataset loaded")
    ds = ds.map(process_ax_path_with_variations, num_parallel_calls=tf.data.AUTOTUNE)
    print("dataset with variations loaded")
    ds = ds.map(lambda x, y, y_var: crop_images_with_variations(x, y, y_var, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = configure_ax_for_performance(ds, training)
    # Preprocessing for the ResNet50 model
    ds = ds.map(lambda x, y, y_var: (keras.applications.resnet50.preprocess_input(x), {'y_class': y, 'y_var': y_var}), num_parallel_calls=tf.data.AUTOTUNE)
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

def load_ax_ds_splited(ds_dir, visualize: bool = False, training: bool = False):
    ds = load_ax_paths_ds(ds_dir)
    if training:
        ds = ds.take(800)
    else:
        ds = ds.skip(800)
    print("dataset loaded")
    ds = ds.map(process_ax_path_with_variations, num_parallel_calls=tf.data.AUTOTUNE)
    print("dataset with variations loaded")
    ds = ds.map(lambda x, y, y_var: crop_images_with_variations(x, y, y_var, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = configure_ax_for_performance(ds, training)
    # Preprocessing for the ResNet50 model
    ds = ds.map(lambda x, y, y_var: (keras.applications.resnet50.preprocess_input(x), {'y_class': y, 'y_var': y_var}), num_parallel_calls=tf.data.AUTOTUNE)
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

def load_ax_chosen_objects_paths_ds(ds_dir, object_number):
    file_list = [os.path.join(ds_dir, x) for x in os.listdir(ds_dir) if x.split(sep='_')[1] == str(object_number)]
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.shuffle(buffer_size=len(file_list), reshuffle_each_iteration=False, seed=0)
    return ds

def load_ax_ds_chosen_objects(ds_dir, object_number, visualize: bool = False, training: bool = False):
    ds = load_ax_chosen_objects_paths_ds(ds_dir, object_number)
    print("dataset loaded")
    ds = ds.map(process_ax_path_with_variations, num_parallel_calls=tf.data.AUTOTUNE)
    print("dataset with variations loaded")
    ds = ds.map(lambda x, y, y_var: crop_images_with_variations(x, y, y_var, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = configure_ax_for_performance(ds, training)
    # Preprocessing for the ResNet50 model
    ds = ds.map(lambda x, y, y_var: (keras.applications.resnet50.preprocess_input(x), {'y_class': y, 'y_var': y_var}), num_parallel_calls=tf.data.AUTOTUNE)
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