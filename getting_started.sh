# NOTE: Only if on lambdalabs, run:
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

git clone https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git

cd Advanced-Techniques-in-Chess-Engines/py

source setup.sh

# # run train.py for 1 minute to setup of the reference dataset
# timeout 1m python3.11 train.py
# 
# # download the chess database
# python3.11 -m src.games.chess.ChessDatabase 30 2000
# 
# # pretrain the model on the chess database
# python3.11 -m src.eval.DatasetTrainer reference/chess_database/memory_*/*.hdf5
# 
# # copy the pretrained model to the training_data directory
# mkdir -p training_data/chess
# cp reference/ChessGame/model_5.pt training_data/chess/model_1.pt
# cp reference/ChessGame/model_5.pt training_data/chess/reference_model.pt
# 
# # run train.py based on the pretrained model
# python3.11 train.py

# Or run train.py in the background
nohup python3.11 train.py > train.log 2>&1 &
tail -f train.log
