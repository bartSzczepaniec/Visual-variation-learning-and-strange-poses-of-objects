# Grad-CAM algorithm from https://keras.io/examples/vision/grad_cam/
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

from parameter_config import class_names, shuffle_buffer_size


def make_grad_cam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None, model_y_type="variations"):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    print(pred_index)
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        if model_y_type == "variations":
            class_channel = preds[0][:, pred_index]
        else:
            class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_img_with_grad_cam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    return superimposed_img

def reverse_resnet50_preprocessing(x):
    mean = np.array([103.939, 116.779, 123.68])
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]

    x = x[..., ::-1]

    return np.clip(x, 0, 255).astype("uint8")

def show_grad_cam_examples(model, dataset, ds_y_type="variations", model_y_type="variations"):
    image_batch, label_batch = next(iter(dataset))

    og_images = reverse_resnet50_preprocessing(image_batch.numpy())
    y_pred = model.predict(image_batch)
    if model_y_type == "variations":
        y_pred = y_pred[0]
    y_pred = np.argmax(y_pred, axis=1)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        heatmap = make_grad_cam_heatmap(np.expand_dims(image_batch[i], axis=0), model, pred_index=y_pred[i], model_y_type=model_y_type)
        superimposed_img = get_img_with_grad_cam(og_images[i], heatmap)
        plt.imshow(superimposed_img)

        if ds_y_type == "variations":
            label = label_batch["y_class"][i]
        else:
            label = label_batch[i]
        plt.title(class_names[label])
        plt.axis("off")
    plt.show()