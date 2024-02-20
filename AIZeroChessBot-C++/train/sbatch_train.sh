#!/bin/bash

cd ../build

cmake --build . --config Release

cp AIZeroChessBot ../train/AIZeroChessBot

cd ../train

sbatch train.sh

