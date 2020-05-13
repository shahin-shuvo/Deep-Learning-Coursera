import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)


  if title:
    plt.title(title)
  plt.imshow(image)
