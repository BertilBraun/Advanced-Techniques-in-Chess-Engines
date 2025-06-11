# NOTE: Only if on lambdalabs, run:
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

cd ~

git clone https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git

cd Advanced-Techniques-in-Chess-Engines/py

# if on aarch64
if [[ "$(uname -m)" == "aarch64" ]]; then
    echo "alias tail='tail -f -n 2000'" >> ~/.bashrc
    echo "alias ch='chmod +x *.sh'" >> ~/.bashrc
    echo "alias tb='tensorboard --port 6007 --logdir'" >> ~/.bashrc
    echo "alias gp='git pull'" >> ~/.bashrc
    echo "alias start='git pull && nohup python3 train.py > \"train_\$(date +%Y%m%d_%H%M%S).log\" 2>&1 &'" >> ~/.bashrc
    echo "alias stop='pkill -f train.py'" >> ~/.bashrc
    echo "alias log='tail -f train.log'" >> ~/.bashrc

    echo "cd ~/Advanced-Techniques-in-Chess-Engines/py" >> ~/.bashrc

    python3 -m venv .venv
    
    echo "source .venv/bin/activate" >> ~/.bashrc

    source ~/.bashrc
        
    pip3 install --upgrade pip
    pip3 install -r requirements.txt

    start

    sleep 1

    log
else
    source setup.sh

    compile

    # download the chess database
    python3 -m src.games.chess.ChessDatabase 30 2000

    mkdir -p training_data/chess
    # initialize the training_data with some of the chess database to start training
    mkdir -p training_data/chess/memory_0
    mv reference/chess_database/memory_202409/* training_data/chess/memory_0/

    # pretrain the model on the chess database
    python3 -m src.eval.DatasetTrainer reference/chess_database/memory_*/*.hdf5

    # copy the pretrained model to the training_data directory
    cp reference/ChessGame/model_4.pt training_data/chess/model_15.pt
    cp reference/ChessGame/model_4.pt training_data/chess/reference_model.pt
    cp reference/ChessGame/model_4.jit.pt training_data/chess/reference_model.jit.pt
    
    # run train.py based on the pretrained model
    python3 train.py

    # Or run train.py in the background
    start
    # tail train_$(date +%Y%m%d_%H%M%S).log
fi