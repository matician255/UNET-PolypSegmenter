import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision
from data import load_data, tf_dataset
from unet.model import build_model
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from train import iou
import tifffile

"""
So i had issues with loading the TIFF file RBG images using cv2 alone so i decided to use tifffile library 
which solved my problem   
"""


def read_image(path):
    x = tifffile.imread(path)  # Reads TIFF reliably
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if x.ndim == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x


def mask_parse(mask):
    mask = np.squeeze(mask)  # squeeze out unnecessary dimension from (256,256,1) to (256, 256)
    mask = np.stack([mask, mask, mask], axis=-1) # copying the mask into three channels (256, 256, 3)
    return mask
# The resulting mask is now a 3-channel image that can be treated like a regular RGB image when displaying or saving.


if __name__ == "__main__":
    # Seeding
    tf.random.set_seed(42)

    path = r"C:\Users\Administrator\PycharmProjects\tf_project\unet\CVC-ClinicDB"
    batch = 8
    (x_train, y_train), (valid_x, valid_y), (x_test, y_test) = load_data(path)
    print(len(x_train), len(valid_x), len(x_test))

    test_dataset = tf_dataset(x_test, y_test, batch=batch)
    test_steps = len(x_test) // batch
    if len(x_test) % batch != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("files/model.h5")

    model.evaluate(test_dataset, steps=test_steps)

    # tqdm for progress bar, total tells how many iteration there will be
    for i, (x, y) in tqdm(enumerate(zip(x_test, y_test)), total=len(x_test)):
        x = read_image(x)
        y = read_mask(y)

        # add a new axis so that the image has a batch dimension (making it shape (1, 256, 256, 3))
        y_pred = model.predict(np.expand_dims(x, axis=0))
        y_pred = y_pred[0] > 0.5
        h, w, _ = x.shape

        white_line = np.ones((h, 10, 3), dtype=np.uint8) * 255.0
        """creates a white image (all pixel values are 255) with the same height as the image, 10 pixels wide,
         and 3 color channels. This white line is used to separate images visually when concatenated"""

        # converting them to an 8 standard bit image
        x_uint8 = (x * 255.0).astype(np.uint8)
        y_mask = (mask_parse(y) * 255.0).astype(np.uint8)
        y_pred_mask = (mask_parse(y_pred) * 255.0).astype(np.uint8)

        # concatenating images
        all_images = [
            x_uint8, white_line,
            y_mask, white_line,
            y_pred_mask
        ]

        image = np.concatenate(all_images, axis=1)  # puts these images side by side horizontally
        cv2.imwrite(f"results/{i}.png", image)  # save the concatenated images in the results folder

