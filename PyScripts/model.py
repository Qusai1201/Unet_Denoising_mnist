import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from processing import *
import matplotlib.pyplot as plt
import os

train_data = read_data("Data/training")
valid_data = read_data("Data/validation")

print(train_data.shape)
print(valid_data.shape)


noisy_train_data = noise(train_data , 0.5)
noisy_valid_data = noise(valid_data, 0.5)



def unet_model(input_shape):

    inputs = tf.keras.Input(shape=input_shape)


    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)

    conv1 = layers.ZeroPadding2D(((2, 2), (0, 0)))(conv1)

    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)

    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)


    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)

    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)


    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)



    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)


    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)



    up5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)

    up5 = layers.concatenate([up5, conv3])

    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(up5)

    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)


    up6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)

    up6 = layers.concatenate([up6, conv2])

    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(up6)

    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)


    up7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)

    up7 = layers.concatenate([up7, conv1])

    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(up7)

    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)


    output = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    output = layers.Cropping2D(cropping=((2, 2),  (0 , 0)))(output)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model


input_shape = (28, 280, 1)


model = unet_model(input_shape)


model.summary()


model.compile(optimizer='adam', loss="binary_crossentropy")

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='autoencoder.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

history = model.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=10,
    batch_size=32,
    shuffle=True,
    validation_data=(noisy_valid_data, valid_data),
    callbacks=[model_checkpoint_callback]
)

fig, ax = plt.subplots(figsize=(16,9), dpi=300)
plt.title(label='Model Loss by Epoch', loc='center')

ax.plot(history.history['loss'], label='Training Data', color='black')
ax.plot(history.history['val_loss'], label='Test Data', color='red')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.xticks(ticks=np.arange(len(history.history['loss'])), labels=np.arange(1, len(history.history['loss'])+1))
plt.legend()

plt.show()