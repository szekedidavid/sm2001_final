import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helpers import read_data
from helpers import plot_velocities
import matplotlib.pyplot as plt
import keras
import gc

# noinspection PyUnresolvedReferences
from tensorflow.keras import models, layers, callbacks, backend

backend.clear_session()
gc.collect()

# read data
X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()

epochs = 1000
batch_size = 100
patience = 50
latent_dim = 12

# 80 - 20 split
UV_train, UV_temp = train_test_split(UV, test_size=0.3, random_state=0)
UV_test, UV_val = train_test_split(UV_temp, test_size=0.33, random_state=0)

# subtract average from U and V
scaler = StandardScaler()
UV_train_norm = scaler.fit_transform(UV_train.reshape(-1, 2)).reshape(UV_train.shape)
UV_val_norm = scaler.transform(UV_val.reshape(-1, 2)).reshape(UV_val.shape)
UV_test_norm = scaler.transform(UV_test.reshape(-1, 2)).reshape(UV_test.shape)

@keras.src.saving.register_keras_serializable()
class CNNAutoencoder(models.Model):
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
            layers.Conv2DTranspose(filters=2, kernel_size=(3, 3), activation='linear', padding='same')
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

cnn_ae = CNNAutoencoder(latent_dim)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

cnn_ae.compile(optimizer='Adam', loss='mse')
cnn_ae.fit(UV_train_norm, UV_train_norm,  validation_data=(UV_val_norm, UV_val_norm), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
cnn_ae.save(f'cnn_ae_{latent_dim}.keras')
cnn_ae.encoder.summary()
cnn_ae.decoder.summary()
# cnn_ae = models.load_model(f'cnn_ae_d{latent_dim}.keras', custom_objects={'CNNAutoencoder': CNNAutoencoder})

# print MSE
UV_pred_norm = cnn_ae.predict(UV_test_norm)
UV_pred = scaler.inverse_transform(UV_pred_norm.reshape(-1, 2)).reshape(UV_pred_norm.shape)
mse = np.mean((UV_test - UV_pred) ** 2)
print(f'MSE: {mse}')
