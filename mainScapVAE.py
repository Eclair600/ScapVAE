#This is the main function that we want to execute

from dataProcessing import preprocess
from modelVAE import modelCode
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
from keras.engine.input_layer import Input

vae_x, decoder, encoder = modelCode((10000,))
filePathname = 'flattenMeshes_1.csv'
input = preprocess(filePathname)
input_x = input[0:170,0,:]
input_x_valid = input[170:,0,:]

vae_x.compile(optimizer='rmsprop', loss=None)
vae_x.summary()

max_input = np.amax(input_x)
input_x = input_x/max_input
input_x_valid = input_x_valid/max_input

vae_x.fit(x=input_x, y=None,
        shuffle=True,
        epochs=5,
        batch_size=5,
        validation_data=(input_x_valid, None))


# Generation of sample from the decoder
DL_input = Input(vae_x.layers[11].input_shape[1:])
DL_model = DL_input
for layer in vae_x.layers[11:12]:
    DL_model = layer(DL_model)
DL_model = Model(inputs=DL_input, outputs=DL_model)
DL_model.summary()

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 128)
new_s = np.array([s])
s_decoded = DL_model.predict(new_s)


