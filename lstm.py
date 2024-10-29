import numpy as np

from helpers import plot_velocities
from functools import partial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from helpers import read_data
import gc

# noinspection PyUnresolvedReferences
from tensorflow.keras import models, layers, regularizers, callbacks, backend

backend.clear_session()
gc.collect()

X, X_norm, n, m, x, y, UV, u_min, u_max, v_min, v_max = read_data()
plot_vel = partial(plot_velocities, x=x, y=y, n=n)

r = 100
epochs = 2000
batch_size = 32
patience = 100
sequence_length = 10

# get pod modes
svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]
Vt_red = V_red.T

V_temp, V_test = train_test_split(V_red, test_size=0.2, shuffle=False)
Vt_temp = V_temp.T
Vt_test = V_test.T

def preprocess(X, sequence_length):
    data_nr = X.shape[1] - sequence_length
    X_seq = np.zeros((data_nr, sequence_length, X.shape[0]))
    Y_seq = np.zeros((data_nr, X.shape[0]))

    for i in range(data_nr):
        X_seq[i] = X[:, i: i + sequence_length].T
        Y_seq[i] = X[:, i + sequence_length]

    return X_seq, Y_seq


X_temp, Y_temp = preprocess(Vt_temp, sequence_length)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.09, shuffle=True, random_state=0)

@keras.saving.register_keras_serializable()
class LSTM(models.Model):
    def __init__(self, sequence_length, r, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.r = r

        self.model = models.Sequential([
            layers.LSTM(256, return_sequences=True),
            layers.BatchNormalization(),
            layers.LSTM(256, return_sequences=False),
            layers.Dense(r, activation='linear')
        ])

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        return {'sequence_length': self.sequence_length, 'r': self.r}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

lstm = LSTM(sequence_length, r)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
lstm.save(f'lstm_l{sequence_length}_r{r}.keras')
# lstm = models.load_model(f'lstm_models/lstm_l{sequence_length}_r{r}.keras', custom_objects={'LSTM': LSTM})

# get training loss
loss = lstm.evaluate(X_train, Y_train)
print(f'Training loss: {loss}')

# get validation loss
loss = lstm.evaluate(X_val, Y_val)
print(f'Validation loss: {loss}')

# simulate the full dataset
# V_init = Vt_red[:, :sequence_length].T
# V_pred = np.zeros((m, r))
# V_pred[:sequence_length] = V_init
# for i in range(sequence_length, Vt_red.shape[1]):
#     print('Predicting', i)
#     V_pred_norm = V_pred[i - sequence_length: i]
#     V_pred[i] = lstm.predict(V_pred_norm[np.newaxis, :, :])
# X_pred = U_red @ S_red @ V_pred.T
# mse_time = np.mean((X_norm - X_pred) ** 2, axis=0)
# times = np.arange(X_temp.shape[0], X_temp.shape[0] + Vt_test.shape[1])
# plt.plot(mse_time)
# plt.xlim(0, 800)
# plt.xlabel('$t$')
# plt.ylabel('MSE')
# plt.axvline(sequence_length, color='black', linestyle='--')
# plt.show()

# simulate the test data
# V_pred = np.zeros((Vt_test.shape[1], r))
# V_pred[:sequence_length] = Vt_test[:, :sequence_length].T
# for i in range(sequence_length, Vt_test.shape[1]):
#     print('Predicting', i)
#     V_pred[i] = lstm.predict(V_pred[i - sequence_length: i][np.newaxis, :, :])
# X_pred = U_red @ S_red @ V_pred.T
# X_test = X_norm[:, -Vt_test.shape[1]:]
# mse_time = np.mean((X_test - X_pred) ** 2, axis=0)
# times = np.arange(800, 1000)
# plt.plot(times, mse_time)
# plt.xlabel('$t$')
# plt.ylabel('MSE')
# plt.xlim(800, 1000)
# plt.axvline(810, color='black', linestyle='--')
# plt.show()
