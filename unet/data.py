import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(path):
    path = path.decode()  # convert byte string to normal text string
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # read the image in colour
    x = cv2.resize(x, (256, 256))   # make images uniform , efficient, and AI friendly
    x = x/255.0  # normalize the pixel values easier for ML to understand
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # read an image in grayscale
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)  # turn a 2D array to 3D ,adding a channel
    return x


def tf_parse(x, y):
    def _parse(x, y):  # call read_image and read_mask to load them
        x = read_image(x)
        y = read_mask(y)
        return x, y

    # using our python function code inside a tensorflow graph based system
    # to convert the images into tensorflow tensors with type tf.float64
    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])

    # we explicitly tell tensorflow the shape of our data
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    # each x is paired with its corresponding y to form a tensorflow dataset containing paired items in a list
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # each image-label pair is passed through a tf_parse function
    dataset = dataset.map(tf_parse)
    # group the pairs into batches
    dataset = dataset.batch(batch)
    # this makes the dataset repeat forever
    dataset = dataset.repeat()
    return dataset

 """ below is a protective way to specify which code should only run when the file is executed as the main program,
  and not when it is imported into another module."""
if __name__ == "__main__":
    path = r"C:\Users\Administrator\PycharmProjects\tf_project\unet\CVC-ClinicDB"
    (x_train, y_train), (valid_x, valid_y), (x_test, y_test) = load_data(path)
    print(len(x_train), len(valid_x), len(x_test))
