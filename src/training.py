import pandas as pd

from time import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Reshape, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import History, Callback, TensorBoard

from util import create_training_data, load_last_training_weights_file, plot, plot_history


class Plotter(Callback):
    batch_loss = []  # loss at given batch

    def __init__(self, batches):
        super(Plotter, self).__init__()
        self.batches = batches
        self.current_batch = 0

    def on_train_batch_end(self, batch, logs=None):
        self.current_batch += 1

        Plotter.batch_loss.append(logs.get('loss'))

        if self.current_batch % self.batches == 0:
            plot(Plotter.batch_loss, Plotter.batch_loss, 'loss', '')


def gen_model() -> Sequential:
    model = Sequential()
    model.add(Dense(2048, input_shape=(12 * 8 * 8,), activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.001),
        # metrics=['accuracy', 'mse']
    )

    return model

    model = Sequential()
    model.add(Reshape((12, 8, 8, 1), input_shape=(12 * 64,)))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same', input_shape=(12, 8, 8, 1)))

    for _ in range(10):
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())

    """ model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization()) """
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    # model.add(Rescaling(scale=1 / 10., offset=0))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.01),
        # metrics=['accuracy', 'mse']
    )
    return model


def train(model: Sequential, X, y, index: int):
    model.optimizer.learning_rate = 0.001 / (index + 1)

    history: History = model.fit(
        X,
        y,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        callbacks=[
            ModelCheckpoint('../training/' + f'{index:03d}' + 'weights{epoch:08d}.h5',
                            save_weights_only=True, save_freq='epoch'),
            Plotter(batches=100),
            # access via tensorboard --logdir ../training/logs
            TensorBoard(log_dir=f'../training/logs/{time()}.log')
        ]
    )

    plot_history(history, index)


if __name__ == '__main__':

    model = gen_model()
    model.summary()
    # load_last_training_weights_file(model)

    for i, chunk in enumerate(pd.read_csv("../dataset/nm_games.csv", header=None, chunksize=100000)):
        X, y = create_training_data(chunk)
        train(gen_model(), X, y, i)

    model.save("../dataset/model.h5")
