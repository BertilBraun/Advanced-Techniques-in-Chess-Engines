import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Reshape

from util import bitfield_to_nums


def gen_model():
    model = Sequential()
    model.add(Reshape((12, 64, 1), input_shape=(12 * 64,)))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same', input_shape=(12, 64, 1)))

    for _ in range(12):
        model.add(Conv2D(256, (3, 3), activation='relu',
                         padding='same'))
        model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def train_on_chunk(model, dataset):
    y = dataset[12].values
    X = dataset.drop(12, axis=1)

    def transform(row):
        return list(np.concatenate([bitfield_to_nums(e) for e in row]))
    X = X.apply(transform, axis=1, result_type='expand')

    # move into range of 0 - 1
    y = np.vectorize(lambda v: v / 250. + 0.5)(y)
    print(min(y), max(y))

    model.fit(
        X,
        y,
        epochs=10,
        batch_size=4,
        callbacks=[
            ModelCheckpoint('../training/weights{epoch:08d}.h5',
                            save_weights_only=True, save_freq=1)
        ]
    )


if __name__ == '__main__':

    model = gen_model()
    model.summary()

    for chunk in pd.read_csv("../dataset/nm_games.csv", header=None, chunksize=10000):
        train_on_chunk(gen_model(), chunk)

    model.save("../dataset/model.h5")
