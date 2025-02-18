import parameter_config
from load_ax_ds import configure_ax_for_performance, load_ax_chosen_objects_paths_ds
from load_ds import *
from parameter_config import ax_variations_names

def get_ax_labels(file_path):
    path_parts = tf.strings.split(file_path, os.path.sep)
    file_name = path_parts[-1]
    file_name_parts = tf.strings.split(file_name, ".")
    file_name_parts = tf.strings.split(file_name_parts[0], "_")
    class_name = file_name_parts[0]
    var_name = tf.strings.join([file_name_parts[3], file_name_parts[4], file_name_parts[5]], separator="-")
    
    yaw = tf.strings.to_number(tf.strings.substr(file_name_parts[3], 1, -1), out_type=tf.float32)
    pitch = tf.strings.to_number(tf.strings.substr(file_name_parts[4], 1, -1), out_type=tf.float32)
    roll = tf.strings.to_number(tf.strings.substr(file_name_parts[5], 1, -1), out_type=tf.float32)

    yaw = tf.experimental.numpy.deg2rad(yaw)
    pitch = tf.experimental.numpy.deg2rad(pitch)
    roll = tf.experimental.numpy.deg2rad(roll)
    
    one_hot = class_name == class_names

    return tf.argmax(one_hot), yaw, pitch, roll

def crop_images_with_ax_variations(img, label, label_yaw, label_pitch, label_roll, training):
    if training:
        img = tf.image.random_crop(img, size=input_shape)  # Randomly crop to 224x224
    else:
        img = tf.image.central_crop(img, central_fraction=0.875)
    return img, label, label_yaw, label_pitch, label_roll

def process_ax_path_with_variations(file_path):
  label, label_yaw, label_pitch, label_roll = get_ax_labels(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label, label_yaw, label_pitch, label_roll

def load_ax_ds_chosen_objects_with_var(ds_dir, object_number, visualize: bool = False, training: bool = False):
    ds = load_ax_chosen_objects_paths_ds(ds_dir, object_number)
    print("dataset loaded")
    ds = ds.map(process_ax_path_with_variations, num_parallel_calls=tf.data.AUTOTUNE)
    print("dataset with variations loaded")
    ds = ds.map(lambda x, y, y_yaw, y_pitch, y_roll: crop_images_with_ax_variations(x, y, y_yaw, y_pitch, y_roll, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = configure_ax_for_performance(ds, training)
    # Preprocessing for the ResNet50 model
    ds = ds.map(lambda x, y, y_yaw, y_pitch, y_roll: (keras.applications.resnet50.preprocess_input(x), {'y_class': y, 'y_yaw': y_yaw, 'y_pitch': y_pitch, 'y_roll': y_roll}), num_parallel_calls=tf.data.AUTOTUNE)
    print("dataset with variations preprocessed")

    if visualize:
        image_batch, label_batch = next(iter(ds))
        # exemplary data in batch visualisation
        plt.figure(figsize=(10, 12))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch["y_class"][i]
            label_yaw = label_batch["y_yaw"][i].numpy()
            label_pitch = label_batch["y_pitch"][i].numpy()
            label_roll = label_batch["y_roll"][i].numpy()
            plt.title(class_names[label] + "\n-y:" + str(label_yaw) + "\n-p:" + str(label_pitch) + "\n-r:" + str(label_roll))
            plt.axis("off")
        plt.show()
    return ds