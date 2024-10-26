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

# get pod modes
svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

r = 50
epochs = 1000
batch_size = 32
patience = 50
sequence_length = 10
alpha = 0

U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]
Vt_red = V_red.T  # each column is a state

def preprocess(X, sequence_length):
    data_nr = X.shape[1] // (sequence_length + 1) # number of sequences
    X_seq = np.zeros((data_nr, sequence_length, X.shape[0]))
    Y_seq = np.zeros((data_nr, X.shape[0]))
    for i in range(data_nr):
        X_seq[i] = X[:, i * sequence_length: (i + 1) * sequence_length].T
        Y_seq[i] = X[:, (i + 1) * sequence_length].T
    return X_seq, Y_seq

X_seq, Y_seq = preprocess(Vt_red, sequence_length)
X_train, X_temp, Y_train, Y_temp = train_test_split(X_seq, Y_seq, test_size=0.3, shuffle=False)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.33, shuffle=False)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train.reshape(-1, r)).reshape(X_train.shape)
X_val_norm = scaler.transform(X_val.reshape(-1, r)).reshape(X_val.shape)
X_test_norm = scaler.transform(X_test.reshape(-1, r)).reshape(X_test.shape)
Y_train_norm = scaler.transform(Y_train)
Y_val_norm = scaler.transform(Y_val)
Y_test_norm = scaler.transform(Y_test)

@keras.saving.register_keras_serializable()
class LSTM(models.Model):
    def __init__(self, sequence_length, r, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.r = r

        self.model = models.Sequential([
            layers.LSTM(128, kernel_regularizer=regularizers.l2(alpha), return_sequences=True),
            layers.LSTM(64, kernel_regularizer=regularizers.l2(alpha)),
            layers.Dense(r, activation='linear')
        ])

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        return {'sequence_length': self.sequence_length, 'alpha': self.alpha, 'r': self.r}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

lstm = LSTM(sequence_length, r, alpha)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_norm, Y_train_norm, validation_data=(X_val_norm, Y_val_norm), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
lstm.save('lstm.keras')
lstm.model.summary()
# lstm = models.load_model('lstm.keras', custom_objects={'LSTM': LSTM})

V_init = Vt_red[:, :sequence_length].T
V_pred = np.zeros((m, r))
V_pred[:sequence_length] = V_init
for i in range(sequence_length, Vt_red.shape[1] - sequence_length):
    print('Predicting', i)
    V_pred_norm = scaler.transform(V_pred[i - sequence_length: i])
    V_pred[i] = scaler.inverse_transform(lstm.predict(V_pred_norm.reshape(1, sequence_length, r))).reshape(r)
X_pred = U_red @ S_red @ V_pred.T

# plot mse over time
mse_time = np.mean((X_norm - X_pred) ** 2, axis=0)
plt.plot(mse_time)
plt.show()

# print mse
mse = np.mean(mse_time)
print(f'MSE: {mse}')
