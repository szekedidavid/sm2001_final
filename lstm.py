import numpy as np

from helpers import plot_velocities
from functools import partial
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from helpers import read_data

X, n, m, UV, x, y = read_data()

X_norm = np.zeros_like(X)
X_norm[:n // 2] = X[:n // 2] - np.mean(X[:n // 2])
X_norm[n // 2:] = X[n // 2:] - np.mean(X[n // 2:])

# get pod modes
svd = np.linalg.svd(X_norm, full_matrices=False)
U = svd[0]
S = np.diag(svd[1])
V = svd[2].T

r = 20
U_red = U[:, :r]
S_red = S[:r, :r]
V_red = V[:, :r]
Vt_red = V_red.T  # each column is a state

# noinspection PyUnresolvedReferences
from tensorflow.keras import models, layers

sequence_length = 10

def preprocess(X, sequence_length):
    data_nr = X.shape[1] // (sequence_length + 1) # number of sequences
    X_seq = np.zeros((data_nr, sequence_length, X.shape[0]))
    Y_seq = np.zeros((data_nr, X.shape[0]))
    for i in range(data_nr):
        X_seq[i] = X[:, i * sequence_length: (i + 1) * sequence_length].T
        Y_seq[i] = X[:, (i + 1) * sequence_length].T
    return X_seq, Y_seq

X_seq, Y_seq = preprocess(X_norm, sequence_length)
X_train, X_val, Y_train, Y_val = train_test_split(X_seq, Y_seq, test_size=0.1, random_state=0)

class LSTM(models.Model):  # todo mess around with regularizers
    def __init__(self, sequence_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.lstm = layers.LSTM(128)  # todo stack layers?
        self.dropout = layers.Dropout(0.2)
        self.dense = layers.Dense(n, activation='linear')

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dropout(x)+9
        return self.dense(x)

lstm = LSTM(sequence_length)
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=1000)

# save
lstm.save('lstm.keras')

# predict flow from first 10 data points
X_init = X_norm[:, :sequence_length].T
X_pred = np.zeros((X_norm.shape[1], n))
X_pred[:sequence_length] = X_init
for i in range(sequence_length, X_norm.shape[1] - sequence_length):
    print('Predicting', i)
    X_pred[i] = lstm.predict(X_pred[i - sequence_length: i].reshape(1, sequence_length, n))

# plot MSE
plt.plot(np.mean((X_norm - X_pred.T) ** 2, axis=1))
plt.xlabel('Index')
plt.ylabel('MSE')
plt.show()
