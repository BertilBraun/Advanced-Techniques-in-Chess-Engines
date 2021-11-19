import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Reshape

from util import create_training_data


def gen_model():
    model = Sequential()
    model.add(Reshape((12, 64, 1), input_shape=(12 * 64,)))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same', input_shape=(12, 64, 1)))

    """ for _ in range(12):
        model.add(Conv2D(256, (3, 3), activation='relu',
                         padding='same'))
        model.add(BatchNormalization()) """

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
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def train(model: Sequential, X, y):
    model.fit(
        X,
        y,
        epochs=50,
        batch_size=32,
        validation_split=0.3,
        callbacks=[
            ModelCheckpoint('../training/weights{epoch:08d}.h5',
                            save_weights_only=True, save_freq='epoch')
        ]
    )


if __name__ == '__main__':

    model = gen_model()
    model.summary()

    for chunk in pd.read_csv("../dataset/nm_games.csv", header=None, chunksize=100000):
        X, y = create_training_data(chunk)
        train(gen_model(), X, y)

    model.save("../dataset/model.h5")
