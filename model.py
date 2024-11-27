import keras
from keras import Model

from keras.src.applications.resnet import ResNet50
from keras.src.layers import GlobalAveragePooling2D, Dense

from parameter_config import *
def get_resnet50_var_classification():
    input_tensor = keras.Input(shape=input_shape)

    base_model = ResNet50(weights="imagenet", classes=classes, include_top=False, input_tensor=input_tensor,
                          input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    base_model_ouput = base_model.output

    outputs = GlobalAveragePooling2D()(base_model_ouput)
    # Adding fully connected layer
    outputs = Dense(1024, activation='relu', name="fc")(outputs)

    outputs_standard = Dense(classes, activation='softmax', name='y_class')(outputs)
    outputs_var = Dense(variations, activation='softmax', name='y_var')(outputs)

    return Model(inputs=base_model.input, outputs=[outputs_standard, outputs_var])