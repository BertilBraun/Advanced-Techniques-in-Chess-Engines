# NOTE: Only if on lambdalabs, run:
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

git clone https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git

cd Advanced-Techniques-in-Chess-Engines/py

source setup.sh
    echo "alias ch='chmod +x *.sh'" >> ~/.bashrc
    echo "alias tb='tensorboard --port 6007 --logdir'" >> ~/.bashrc
    echo "alias gp='git pull'" >> ~/.bashrc
    echo "alias start='git pull && nohup python3 train.py > train.log 2>&1 &'" >> ~/.bashrc
    echo "alias stop='pkill -f train.py'" >> ~/.bashrc
    echo "alias log='tail -f train.log'" >> ~/.bashrc
        
    pip3 install -r requirements.txt

    start

    sleep 1

    log
else
    source setup.sh

    # download the chess database
    python3.11 -m src.games.chess.ChessDatabase 30 2000

    # pretrain the model on the chess database
    python3.11 -m src.eval.DatasetTrainer reference/chess_database/memory_*/*.hdf5

    # copy the pretrained model to the training_data directory
    mkdir -p training_data/chess
    cp reference/ChessGame/model_4.pt training_data/chess/model_15.pt
    cp reference/ChessGame/model_4.pt training_data/chess/reference_model.pt
    
    # run train.py based on the pretrained model
    # python3.11 train.py

    # Or run train.py in the background
    nohup python3.11 train.py > train.log 2>&1 &
    tail -f train.log
fi