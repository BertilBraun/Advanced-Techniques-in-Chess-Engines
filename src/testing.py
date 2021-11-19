import os
import pandas as pd
from training import create_training_data, gen_model

if __name__ == "__main__":
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
