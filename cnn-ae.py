import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np
from sklearn.model_selection import train_test_split
from helpers import read_data
from helpers import plot_velocities
import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
from tensorflow.keras import models, layers


X, n, m, UV, x, y = read_data()

# subtract average from U and V
UV_norm = np.zeros_like(UV)
UV_norm[:, :, :, 0] = (UV[:, :, :, 0] - np.mean(UV[:, :, :, 0])) / np.std(UV[:, :, :, 0])
UV_norm[:, :, :, 1] = (UV[:, :, :, 1] - np.mean(UV[:, :, :, 1])) / np.std(UV[:, :, :, 1])

# 80 - 20 split
UV_train, UV_val = train_test_split(UV, test_size=0.1, random_state=0)


class CNNAutoencoder(models.Model):  # todo use hierarchical?
    def __init__(self, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.encoder = models.Sequential([
            layers.InputLayer(shape=(128, 64, 2)),
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='tanh', padding='same'),
            layers.MaxPool2D((2, 2), padding='same'),
            layers.Dropout(0.2),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', padding='same'),
            layers.MaxPool2D((2, 2), padding='same'),
            layers.Dropout(0.2),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='tanh', padding='same'),
            layers.MaxPool2D((2, 2), padding='same'),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='tanh'),
            layers.Dense(latent_dim, activation='tanh')
            ])

        self.decoder = models.Sequential([
            layers.InputLayer(shape=(latent_dim,)),
            layers.Dense(128, activation='tanh'),
            layers.Dense(np.prod(self.encoder.layers[-4].output.shape[1:]), activation='tanh'),
            layers.Reshape(self.encoder.layers[-4].output.shape[1:]),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='tanh', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), activation='tanh', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(filters=2, kernel_size=(3, 3), padding='same')
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


latent_dim = 20
cnn_ae = CNNAutoencoder(latent_dim)

cnn_ae.encoder.summary()
cnn_ae.decoder.summary()


cnn_ae.compile(optimizer='Adam', loss='mse')
cnn_ae.fit(UV_train, UV_train, epochs=2000, batch_size=100, validation_data=(UV_val, UV_val))
cnn_ae.save('cnn_ae.keras')

# plot all modes
for i in range(latent_dim):
    mode = np.zeros((1, latent_dim))
    mode[0, i] = 1
    output = cnn_ae.decoder(mode)
    output = output.numpy()
    mode = np.zeros(n)
    mode[:n // 2] = output[0, :, :, 0].ravel()
    mode[n // 2:] = output[0, :, :, 1].ravel()
    plot_velocities(mode, x, y, n)
