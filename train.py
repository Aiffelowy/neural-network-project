#!/usr/bin/env python3


import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

print(tf.config.list_physical_devices('GPU'))


IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 32


def plot_loss(history):
    plt.figure()
    plt.plot(history.history["loss"], "ro", label="Training Loss")
    plt.plot(history.history["val_loss"], "b", label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.legend()


def plot_accuracy(history):
    plt.figure()
    plt.plot(history.history["sparse_categorical_accuracy"], "ro", label="Training accuracy")
    plt.plot(history.history["val_sparse_categorical_accuracy"], "b", label="Validation accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.legend()


def normalize_img(img, label):
    img = tf.image.rgb_to_grayscale(img)
    return tf.cast(img, tf.float32), label


def normalize_old(image, label):
    return tf.cast(image, tf.float32) / 255., label


def resize_image(image, label):
    return tf.image.resize_with_pad(
            image, IMG_WIDTH, IMG_HEIGHT, tf.image.ResizeMethod.NEAREST_NEIGHBOR
            ), label


ds_train = tf.keras.utils.image_dataset_from_directory(
        "./dataset",
        validation_split=.2,
        seed=42069,
        subset="training",
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE
        )


ds_test = tf.keras.utils.image_dataset_from_directory(
        "./dataset",
        validation_split=.2,
        seed=42069,
        subset="validation",
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE
        )

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img)
ds_test = ds_test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


model = tf.keras.Sequential([
    layers.InputLayer(shape=(IMG_WIDTH, IMG_HEIGHT, 1)),

    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),

    layers.Conv2D(128, 3, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Activation("relu"),
    layers.SeparableConv2D(256, 3, padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(3, strides=2, padding="same"),

    layers.Activation("relu"),
    layers.SeparableConv2D(512, 3, padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(3, strides=2, padding="same"),

    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.25),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(2)
])

# keras.utils.plot_model(model, show_shapes=True)

# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
# ]


model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    ds_train,
    epochs=50,
    #    callbacks=callbacks,
    validation_data=ds_test,
)


# model.save("imposter.keras")
print(history.history.keys())
plot_loss(history)
plot_accuracy(history)
plt.show()
