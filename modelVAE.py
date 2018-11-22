#This file contains the VAE model on Keras

#Some imports e are going to need

import keras
keras.__version__
from keras import backend as K
K.clear_session()
import keras
from keras import layers
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from dataProcessing import preprocess
from keras.models import Model





def modelCode(input_shape = (10000,)):

    #Encoder

    input_coordinates = keras.Input(shape=input_shape)
    x = layers.Dense(300)(input_coordinates)
    x = layers.BatchNormalization(momentum=0.99,epsilon=0.001)(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    z_mean = layers.Dense(128)(x)
    z_mean = layers.BatchNormalization(momentum=0.99,epsilon=0.001)(z_mean)
    z_mean = keras.layers.LeakyReLU(alpha=0.3)(z_mean)
    z_log_var = layers.Dense(128)(x)
    z_log_var = layers.BatchNormalization(momentum=0.99,epsilon=0.001)(z_log_var)
    z_log_var = keras.layers.LeakyReLU(alpha=0.3)(z_log_var)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 128),
                                  mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = Model(input_coordinates, z)
    #Decoder

    # This is the input where we will feed `z`.
    decoder_input = layers.Input(shape=(128,))

    # Upsample to the correct number of units
    x = layers.Dense(300)(decoder_input)
    x = layers.BatchNormalization(momentum=0.99,epsilon=0.001)(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    # Reshape into an image of the same shape as before our last `Flatten` layer
    x = layers.Dense(10000)(x)
    x = layers.BatchNormalization(momentum=0.99,epsilon=0.001)(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    output = layers.Activation(activation='tanh')(x)

    # We end up with a feature map of the same size as the original input.

    # This is our decoder model.
    decoder = Model(decoder_input, output)

    # We then apply it to `z` to recover the decoded `z`.
    z_decoded = decoder(z)

    class CustomVariationalLayer(keras.layers.Layer):

        def vae_loss(self, x, z_decoded):
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)
            xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
            kl_loss = -5e-4 * K.mean(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            z_decoded = inputs[1]
            loss = self.vae_loss(x, z_decoded)
            self.add_loss(loss, inputs=inputs)
            # We don't use this output.
            return x

    y = CustomVariationalLayer()([input_coordinates, z_decoded])
    vae = Model(input_coordinates,y)
    return vae,decoder,encoder

