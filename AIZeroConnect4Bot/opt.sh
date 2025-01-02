#!/bin/bash

#SBATCH --job-name=opt_zero                # job name
#SBATCH --partition=dev_accelerated        # mby GPU queue for the resource allocation.
#SBATCH --time=01:00:00                    # wall-clock time limit
#SBATCH --mem=200000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=4                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:4                       # number of GPUs required
#SBATCH --output=opt_zero_%j.txt
#SBATCH --error=opt_zero_%j.txt

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

python3.11 -m src.hyperparameter_optimization.opt &

sleep 10

python3.11 -m src.hyperparameter_optimization.opt &
python3.11 -m src.hyperparameter_optimization.opt &
python3.11 -m src.hyperparameter_optimization.opt