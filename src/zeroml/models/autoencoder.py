import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AEModel:
    def __init__(self, input_dim, latent=32):
        inp = keras.Input(shape=(input_dim,))
        x = layers.Dense(1024, activation="relu")(inp)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        z = layers.Dense(latent, activation=None)(x)
        y = layers.Dense(128, activation="relu")(z)
        y = layers.Dense(512, activation="relu")(y)
        y = layers.Dense(1024, activation="relu")(y)
        out = layers.Dense(input_dim, activation="sigmoid")(y)
        
        self.model = keras.Model(inp, out)
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    
    def fit(self, X, epochs=50, batch_size=512):
        cb = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=cb, verbose=2)
        return self
    
    def score(self, X):
        Xp = self.model.predict(X, batch_size=1024, verbose=0)
        return ((X - Xp)**2).mean(axis=1)
    
    def save(self, path): 
        self.model.save(path)
    
    @classmethod
    def load(cls, path): 
        obj = cls(1)
        obj.model = tf.keras.models.load_model(path)
        return obj