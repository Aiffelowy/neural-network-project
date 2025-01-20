#!/usr/bin/env python3

import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf


IMG_WIDTH, IMG_HEIGHT = 64, 64

class_names = ["car", "plane"]


def plot_image(i, predictions_array, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    c = 'blue'
    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                    100*np.max(predictions_array)), color=c)


def plot_value_array(i, predictions_array):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')


model = load_model("working.keras")
model = keras.Sequential([model, keras.layers.Softmax()])

# img = cv2.imread(sys.argv[1])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = np.expand_dims(img, axis=0)

img_orig = keras.utils.load_img(sys.argv[1])
img = tf.image.resize_with_pad(img_orig, IMG_WIDTH, IMG_HEIGHT,
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                               antialias=True)

img = img.numpy()
img = keras.ops.expand_dims(img, 0)
# img = tf.cast(img, tf.float32)

img = tf.image.rgb_to_grayscale(img)


predictions = model.predict(img)

# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], img_orig)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i])
# plt.show()


predicted_label = np.argmax(predictions)
print(class_names[predicted_label])
