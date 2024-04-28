import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, ZeroPadding2D
from keras.applications.vgg16 import preprocess_input
import numpy as np
import h5py
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser(description='Density Map for simple dataset')
parser.add_argument('--dataset', type=str, default='iris', help='Dataset to use', choices=['iris', 'diabetes', 'wine', 'breast_cancer'])
args = parser.parse_args()
dataset_name = args.dataset

# load pre-trained VGG16 model with imagenet weights
vgg_model = VGG16(weights='imagenet')
weights = vgg_model.get_weights()

# load image
image_path = '{}.png'.format(dataset_name + '_density')
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)
image_array = preprocess_input(image_array)

# create a new model with the same weights as the pre-trained VGG16 model but for different input shape
def custom_model():
    inputs = Input(shape=(500, 500, 3))
    x = ZeroPadding2D(padding=(100, 100))(inputs)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    layer6_output = x
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    o = x
    model = Model(inputs=inputs, outputs=[o, layer6_output])

    for layer in model.layers:
        layer.trainable = False

    for i, layer in enumerate(model.layers[2:]):
        layer.set_weights(vgg_model.layers[i + 1].get_weights())
    return model

model1 = custom_model()

# extract features from the custom model
layer32_output, layer6_output = model1(image_array)
sio.savemat('{}_l6.mat'.format(dataset_name), {'layer6': layer6_output[0]})
sio.savemat('{}_l32.mat'.format(dataset_name), {'layer32': layer32_output[0]})