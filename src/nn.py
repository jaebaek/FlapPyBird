import tensorflow as tf
import numpy as np

def create_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(5,)),
        tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(units=2)
    ])
    return model

def compile_model(model: tf.keras.Sequential):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.mse)

def train(model: tf.keras.Sequential, x, y):
    x_ = np.array(x, dtype=float).reshape(1, 5)
    y_ = np.array(y, dtype=float).reshape(1, 2)
    model.fit(x_, y_, epochs=1)

def predict(model: tf.keras.Sequential, state):
    x = np.array(state, dtype=float).reshape(1, 5)
    return model.predict(x, verbose=False)