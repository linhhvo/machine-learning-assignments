import pandas as pd
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
from utils_loaddata import load_dataset

X_train, y_train, X_test, y_test, classes = load_dataset()
print("The shape of X_train: " + str(X_train.shape))
print("The shape of y_train: ", y_train.shape)
y_train = y_train.T
y_test = y_test.T

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[64, 64, 3]))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(7, activation="relu"))
model.add(keras.layers.Dense(5, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

eta = 0.0075
iterations = 1000
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=eta),
    metrics=["accuracy"],
)
history = model.fit(X_train, y_train, epochs=iterations)

# pd.DataFrame(history.history).plot(figsize=(5, 5))
# plt.gca().set_ylim(0.6, 0.7)
# plt.show()