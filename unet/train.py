import keras.optimizers
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision
from data import load_data, tf_dataset
from unet.model import build_model


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        # Intersection are The pixels where both y_true and y_pred are 1 (or "active") contribute to the sum.
        intersection = (y_true * y_pred).sum()
        # take all pixels minus the intersection
        union = y_true.sum() + y_pred.sum() - intersection
        # The small constant 1e-15 is added to avoid division by zero, which is a safety check.
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    """This way, you can include this IoU calculation as part of your TensorFlow model’s operations
     while working with TensorFlow tensors."""
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


if __name__ == "__main__":
    # Seeding
    tf.random.set_seed(42)

    path = r"C:\Users\Administrator\PycharmProjects\tf_project\unet\CVC-ClinicDB"
    (x_train, y_train), (valid_x, valid_y), (x_test, y_test) = load_data(path)
    print(len(x_train), len(valid_x), len(x_test))

    # hyperparameters
    batch = 8
    learning_rate = 1e-4
    epochs = 40

    # dataset loading
    train_dataset = tf_dataset(x_train, y_train, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    model = build_model()

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = ['accuracy', Recall(), Precision(), iou]

    # Ensure folders exist
    os.makedirs("file", exist_ok=True)
    os.makedirs("files", exist_ok=True)  # This is also used for saving model.h5

    model.compile(
        loss="binary_crossentropy",
        optimizer = optimizer,
        metrics = metrics
    )

    callbacks = [
        # Saves the model to the file files/model.h5 at the end of each epoch (or based on a specified condition).
        ModelCheckpoint('files/model.h5'),
        # Monitors validation loss, if it doesn't improve for 3 epochs, it reduces the learning rate by a factor of 0.1.
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
        # Logs training metrics (like loss and accuracy) into a CSV file named file/data.csv after every epoch.
        CSVLogger('file/data.csv'),
        # tool that lets you visualize metrics such as loss curves, accuracy, and the model architecture during training
        TensorBoard(),
        # Monitors the val_loss, and if it does not improve for 10 consecutive epochs, it stops the training early.
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    # an integer division to determine how many full batches you have.
    train_steps = len(x_train) // batch
    valid_steps = len(valid_x) // batch

    # Ensures the extra training examples that dont form a complete batch are processed
    if len(x_train) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
        shuffle=False
    )

""" Callbacks are functions that run at certain points during training (for example, at the end of each epoch) 
and perform actions such as saving models, adjusting the learning rate, logging metrics, 
and stopping training when appropriate """








