git clone https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git

cd Advanced-Techniques-in-Chess-Engines/py

python3 -m pip install --upgrade pip
pip install -r requirements.txt

# run train.py for 2 minute to create the training_data directory and setup of the reference dataset
timeout 2m python3 train.py

# download the chess database
python3 -m src.games.chess.ChessDatabase 30 2000

# pretrain the model on the chess database
python3 -m src.eval.DatasetTrainer reference/chess_database/memory_*/*.hdf5

# copy the pretrained model to the training_data directory
cp reference/ChessGame/model_9.pt training_data/ChessGame/model_1.pt
cp reference/ChessGame/model_9.pt training_data/ChessGame/reference_model.pt

# run train.py based on the pretrained model
python3 train.py

# view the results in TensorBoard
tensorboard --logdir logs/