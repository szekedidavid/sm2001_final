import numpy as np
import os
from sklearn.model_selection import train_test_split
from helpers import read_data
import tensorflow as tf
from tensorflow.python.keras import layers, models

X, n, m, x, y = read_data()

X_norm = np.zeros_like(X)
X_norm[:n // 2] = (X[:n // 2] - np.mean(X[:n // 2])) / np.std(X[:n // 2])
X_norm[n // 2:] = (X[n // 2:] - np.mean(X[n // 2:])) / np.std(X[n // 2:])

# 70 - 20 - 10 split
X_train, X_test = train_test_split(X_norm.T, test_size=0.3)
X_test, X_val = train_test_split(X_test, test_size=0.33)

class CNN_AE(models.Model):
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model = self.build_model()
