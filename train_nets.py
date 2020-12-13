from pathlib import Path
from pickle import dump
from typing import Tuple

import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Flatten,
    Input,
    Permute,
    Reshape,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

import speck as sp


def cyclic_learning_rate(num_epochs: int, high_learning_rate: float, low_learning_rate: float):
    return lambda i: low_learning_rate + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (
        high_learning_rate - low_learning_rate
    )


def make_checkpoint(filepath: str) -> ModelCheckpoint:
    return ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True)


# make residual tower of convolutional blocks
def make_resnet(
    num_blocks=2,
    num_filters=32,
    num_outputs=1,
    d1=64,
    d2=64,
    word_size=16,
    ks=3,
    depth=5,
    reg_param=0.0001,
    final_activation="sigmoid",
):
    # Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
    # add a single residual layer that will expand the data to num_filters channels
    # this is a bit-sliced layer
    conv0 = Conv1D(
        num_filters, kernel_size=1, padding="same", kernel_regularizer=l2(reg_param)
    )(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation("relu")(conv0)
    # add residual blocks
    shortcut = conv0
    for _ in range(depth):
        conv1 = Conv1D(
            num_filters,
            kernel_size=ks,
            padding="same",
            kernel_regularizer=l2(reg_param),
        )(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)
        conv2 = Conv1D(
            num_filters,
            kernel_size=ks,
            padding="same",
            kernel_regularizer=l2(reg_param),
        )(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)
        shortcut = Add()([shortcut, conv2])
    # add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation("relu")(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation("relu")(dense2)
    out = Dense(
        num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param)
    )(dense2)
    model = Model(inputs=inp, outputs=out)
    return model


def train_speck_distinguisher(
    num_epochs: int,
    num_rounds: int = 7,
    depth: int = 1,
    batch_size: int = 5000,
    working_dir: str = "./freshly_trained_nets/",
) -> Tuple[Model, Model]:
    """
    Train a SPECK distinguisher.

    Parameters:
        num_epochs -- Number of epochs to train for.
        num_rounds -- Number of SPECK rounds.
        depth -- Residual network depth.
        batch_size -- Training batch size.
        working_dir -- Directory to save data.
    """

    # Create the network
    net = make_resnet(depth=depth, reg_param=10 ** -5)
    net.compile(optimizer="adam", loss="mse", metrics=["acc"])

    # Generate training and validation data
    train_x, train_y = sp.make_train_data(10 ** 7, num_rounds)
    validation_x, validation_y = sp.make_train_data(10 ** 6, num_rounds)

    # Make checkpoints
    checkpoints = make_checkpoint(f"{working_dir}best{num_rounds}depth{depth}.h5")

    # Create learning rate scheduler
    learning_rate_sched = LearningRateScheduler(cyclic_learning_rate(10, 0.002, 0.0001))

    # Train and evaluate
    h = net.fit(
        train_x,
        train_y,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(validation_x, validation_y),
        callbacks=[learning_rate_sched, checkpoints],
    )

    # Create working directory if it doesn't exist
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    # Save data to working directory
    np.save(
        f"{working_dir}h{num_rounds}r_depth{depth}.npy",
        h.history["val_acc"],
    )
    np.save(
        f"{working_dir}h{num_rounds}r_depth{depth}.npy",
        h.history["val_loss"],
    )
    dump(
        h.history,
        open(f"{working_dir}hist{num_rounds}r_depth{depth}.p", "wb"),
    )

    print(f"Best validation accuracy: {np.max(h.history['val_acc'])}")

    return net, h
