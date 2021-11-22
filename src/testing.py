import os
import numpy as np
import pandas as pd


def test_model():
    from training import create_training_data, gen_model

    model = gen_model()

    # get the last filename in the sorted directory "../training"
    last_file = sorted([
        f for f in os.listdir("../training") if f.endswith('.h5')
    ])[-1]
    print("Loading weights from: ../training/" + last_file)

    model.load_weights("../training/" + last_file)

    # test the model

    for chunk in pd.read_csv("../dataset/nm_games.csv", header=None, chunksize=1000):
        X, y = create_training_data(chunk)

        predictions = model.predict(X)

        # print the results and evaluate the error
        for i, p in enumerate(predictions):
            print(f"{y[i]} {p} - {abs(y[i] - p)}")

        break


def plot_data():
    import matplotlib.pyplot as plt
    from training import create_training_data

    for chunk in pd.read_csv("../dataset/nm_games small.csv", header=None, chunksize=100000):
        _, y = create_training_data(chunk)
        x = np.arange(-1, 1.01, 0.01)
        ys = [0] * len(x)

        for v in y:
            ys[int(v * 100) + 100] += 1

        plt.plot(x, ys)
        plt.show()


if __name__ == "__main__":
    # test_model()
    plot_data()
