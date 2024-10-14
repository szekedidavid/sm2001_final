import os

import keras.src.saving

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np
from sklearn.model_selection import train_test_split
from helpers import read_data
from helpers import plot_velocities
import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
from tensorflow.keras import models, layers


X, n, m, UV, x, y = read_data()

u_min = np.min(UV[:, :, :, 0])
u_max = np.max(UV[:, :, :, 0])
v_min = np.min(UV[:, :, :, 1])
v_max = np.max(UV[:, :, :, 1])

# subtract average from U and V
UV_norm = np.zeros_like(UV)
UV_norm[:, :, :, 0] =UV[:, :, :, 0] - np.mean(UV[:, :, :, 0])
UV_norm[:, :, :, 1] = UV[:, :, :, 1] - np.mean(UV[:, :, :, 1])

# 80 - 20 split
UV_train, UV_val = train_test_split(UV_norm, test_size=0.1, random_state=0)

@keras.src.saving.register_keras_serializable()
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
            layers.Conv2DTranspose(filters=2, kernel_size=(3, 3), actviation='linear', padding='same')
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


latent_dim = 20
cnn_ae = CNNAutoencoder(latent_dim)

cnn_ae.encoder.summary()
cnn_ae.decoder.summary()


cnn_ae.compile(optimizer='Adam', loss='mse')
cnn_ae.fit(UV_train, UV_train, epochs=1000, batch_size=100, validation_data=(UV_val, UV_val))
cnn_ae.save('cnn_ae.keras')

# load model
# cnn_ae = models.load_model('cnn_ae.keras')


# plot all modes
for i in range(latent_dim):
    mode = np.zeros((1, latent_dim))
    mode[0, i] = 1
    output = cnn_ae.decoder(mode).numpy()[0]
    plot_velocities(output, x, y, n)

# print RMS
UV_pred = cnn_ae.predict(UV_val)
UV_pred[:, :, :, 0] = UV_pred[:, :, :, 0] + np.mean(UV[:, :, :, 0])
UV_pred[:, :, :, 1] = UV_pred[:, :, :, 1] + np.mean(UV[:, :, :, 1])
UV_val[:, :, :, 0] = UV_val[:, :, :, 0] + np.mean(UV[:, :, :, 0])
UV_val[:, :, :, 1] = UV_val[:, :, :, 1] + np.mean(UV[:, :, :, 1])
rms = np.sqrt(np.mean((UV_val - UV_pred) ** 2))
print(f'RMS: {rms}')

times = 0, 1, 2, 3, 4, 5
for t in times:
    plot_velocities(UV_val[t], x, y, n, u_min, u_max, v_min, v_max)
    plot_velocities(UV_pred[t], x, y, n, u_min, u_max, v_min, v_max)
