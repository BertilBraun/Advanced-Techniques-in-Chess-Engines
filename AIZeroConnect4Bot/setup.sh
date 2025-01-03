#!/bin/bash

# if miniconda is not installed, install it and create a new conda environment
if [ -d "../../miniconda3" ]; then
  echo "Conda environment already exists."
else
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh

    # Initialize conda in your bash shell
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc

    conda create -n Chess python=3.11 -y
    conda activate Chess

    pip install -r requirements.txt
    pip install -r AIZeroConnect4Bot/requirements.txt

    # Add to .bashrc
    echo "module purge" >> ~/.bashrc
    echo "module load compiler/intel/2024.0_llvm" >> ~/.bashrc
    echo "module load devel/cuda/12.4" >> ~/.bashrc
    echo "export OMP_NUM_THREADS=8" >> ~/.bashrc
    echo "alias q='squeue --long'" >> ~/.bashrc
    echo "alias qs='squeue --start'" >> ~/.bashrc
    echo "alias qi='watch "squeue && ls"'" >> ~/.bashrc
    echo "alias b='sbatch'" >> ~/.bashrc
    echo "alias c='scancel'" >> ~/.bashrc
    echo "alias info='sinfo_t_idle'" >> ~/.bashrc
    echo "alias tail='tail -f -n 1000'" >> ~/.bashrc
    echo "conda activate Chess" >> ~/.bashrc
fi

source ~/.bashrc
