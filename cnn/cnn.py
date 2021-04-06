import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import backend
from utils_loaddata import load_dataset

# Load and preprocess data
X_train, y_train, X_test, y_test, classes = load_dataset()

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = y_train.T
y_test = y_test.T

print("Original X shape: ", X_train.shape)
print("X data type:", X_train.dtype)
print("Y shape: ", y_train.shape)

# Construct a CNN network
model = Sequential(
    [
        Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            activation="relu",
            padding="same",
            input_shape=(64, 64, 3),
        ),
        MaxPooling2D(2),
        Conv2D(filters=48, kernel_size=5, activation="relu", padding="same"),
        MaxPooling2D(2),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(84, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Get the summary of each layer
print(model.summary())

# Compile model
model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

# Train model for 100 epochs
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Performance plot
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0, 1]
plt.savefig("cost_curves.png")
plt.show()

print("Training accuracy:")
model.evaluate(X_train, y_train)
print("Testing accuracy:")
model.evaluate(X_test, y_test)