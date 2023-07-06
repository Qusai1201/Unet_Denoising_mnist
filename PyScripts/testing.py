import tensorflow as tf
from PIL import Image
from processing import *
import numpy as np
import os

def makeImage(array):
    array = np.reshape(array, (28, 280))
    array *= 255.0
    data = Image.fromarray(array.astype('uint8'))
    return data

def get_concat_v(im1, im2):
    dst = Image.new('L', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

test_data = read_data("test")

test_data = test_data / 255.0

autoencoder = tf.keras.models.load_model('../models/Unet_autoencoder.h5')

noisy_test_data = noise(test_data, 0.5)
predictions = autoencoder.predict(noisy_test_data)

display(predictions , test_data , 1)
display(predictions , noisy_test_data , 1)


#if not os.path.exists('../predictions'):
#    os.makedirs('../predictions')

for i in range(len(predictions)):
    img = get_concat_v(get_concat_v(makeImage(test_data[i]) , makeImage(noisy_test_data[i])),
                       makeImage(predictions[i]))
    img.save(str(i) + '.png')
