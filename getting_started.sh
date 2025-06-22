# NOTE: Only if on lambdalabs, run:
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

cd ~

git clone https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git

cd Advanced-Techniques-in-Chess-Engines/py

source setup.sh

compile

python3 -m test.inference_speed_test
python3 -m test.mcts_speed_test

# download the chess database
python3 -m src.games.chess.ChessDatabase 5 2000

mkdir -p training_data/chess
# initialize the training_data with some of the chess database to start training
mkdir -p training_data/chess/memory_0
mv reference/chess_database/memory_202409/* training_data/chess/memory_0/

# pretrain the model on the chess database
python3 -m src.eval.DatasetTrainer reference/chess_database/memory_*/*.hdf5

# copy the pretrained model to the training_data directory
cp reference/ChessGame/model_5.pt training_data/chess/model_1.pt
cp reference/ChessGame/model_5.jit.pt training_data/chess/model_1.jit.pt
cp reference/ChessGame/model_5.pt training_data/chess/reference_model.pt
cp reference/ChessGame/model_5.jit.pt training_data/chess/reference_model.jit.pt

# run train.py based on the pretrained model
python3 train.py

# Or run train.py in the background
start
# tail train_$(date +%Y%m%d_%H%M%S).log