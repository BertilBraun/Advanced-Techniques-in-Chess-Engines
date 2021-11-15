import os
import numpy as np

import pandas as pd
from training import gen_model
from util import bitfield_to_nums

if __name__ == "__main__":
    model = gen_model()

    # get the last filename in the sorted directory "../training"
    last_file = sorted(os.listdir("../training"))[-1]
    print("Loading weights from: ../training/" + last_file)

    model.load_weights("../training/" + last_file)

    # test the model

    for chunk in pd.read_csv("../dataset/nm_games.csv", header=None, chunksize=1000):

        y = chunk[12].values
        X = chunk.drop(12, axis=1)

        def transform(row):
            return list(np.concatenate([bitfield_to_nums(e) for e in row]))
        X = X.apply(transform, axis=1, result_type='expand')

        predictions = model.predict_on_batch(X)

        # print the results and evaluate the error
        for i, p in enumerate(predictions):
            print(f"{y[i]} {p} - {abs(y[i] - p)}")
