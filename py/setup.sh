#!/bin/bash

#SBATCH --job-name=setup                 # job name
#SBATCH --partition=dev_accelerated      # mby GPU queue for the resource allocation.
#SBATCH --time=00:20:00                  # wall-clock time limit
#SBATCH --mem=10000                      # memory per node
#SBATCH --nodes=1                        # number of nodes to be used
#SBATCH --cpus-per-task=1                # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1              # maximum count of tasks per node
#SBATCH --mail-type=ALL                  # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=setup.txt
#SBATCH --error=setup.txt


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

    conda create -n Chess python=3.10 -y
    conda activate Chess
    conda install -c conda-forge libstdcxx-ng libgcc-ng -y
    conda install -c conda-forge 'cmake>=3.27' make gxx_linux-64 -y

    # 1) add / refresh NVIDIA’s CUDA repo for Ubuntu 20.04
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    sudo add-apt-repository \
      "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    sudo apt-get update

    # 2) pull in NVTX + cuDNN 9 for CUDA 12.x
    CUDA_REL=12-8              # 12-6, 12-7, … use whatever tool-kit minor you’re building against
    sudo apt-get install -y \
        cuda-nvtx-${CUDA_REL} \
        libcudnn9-cuda-12 \
        libcudnn9-dev-cuda-12

    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt update
    sudo apt install gcc-11 g++-11 -y


    sudo apt install stockfish nano htop -y
    echo "export PATH=\"\$PATH:/usr/games\"" >> ~/.bashrc

    echo "export CMAKE_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:\$CMAKE_INCLUDE_PATH" >> ~/.bashrc
    echo "export CMAKE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$CMAKE_LIBRARY_PATH" >> ~/.bashrc
    echo "export PATH=/usr/local/cuda-12/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

    pip3 install -r requirements.txt

    # Add to .bashrc
    echo "module purge" >> ~/.bashrc
    echo "module load compiler/intel/2024.0_llvm" >> ~/.bashrc
    echo "module load devel/cuda/12.4" >> ~/.bashrc
    
    echo "alias q='squeue --long'" >> ~/.bashrc
    echo "alias qs='watch \"squeue --start\"'" >> ~/.bashrc
    echo "alias qi='watch \"sinfo_t_idle && squeue && ls\"'" >> ~/.bashrc
    echo "alias b='sbatch'" >> ~/.bashrc
    echo "alias c='scancel'" >> ~/.bashrc
    echo "alias info='sinfo_t_idle'" >> ~/.bashrc
    echo "alias python='python3'" >> ~/.bashrc
    echo "alias pip='pip3'" >> ~/.bashrc
    echo "alias tail='tail -f -n 2000'" >> ~/.bashrc
    echo "alias ch='chmod +x *.sh'" >> ~/.bashrc
    echo "alias tb='ulimit -n 50000 && tensorboard --port 8080 --logdir'" >> ~/.bashrc
    echo "alias gp='git pull'" >> ~/.bashrc
    echo "alias start='git pull && nohup python3 -O train.py > \"train_\$(date +%Y%m%d_%H%M%S).log\" 2>&1 &'" >> ~/.bashrc
    echo "alias stop='pkill -f train.py && pkill -f \"python3 -O -c from multiprocessing.spawn import spawn_main\"'" >> ~/.bashrc
    echo "alias gpuH='salloc -p dev_accelerated-h100 --gres=gpu:1 -t 30'" >>  ~/.bashrc
    echo "alias gpuA='salloc -p dev_accelerated --gres=gpu:1 -t 30'" >> ~/.bashrc
    echo "alias gpuH2='salloc -p dev_accelerated-h100 --gres=gpu:2 -t 30'" >> ~/.bashrc
    echo "alias gpuA2='salloc -p dev_accelerated --gres=gpu:2 -t 30'" >>  ~/.bashrc
    echo "alias compile='cd ../cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=\$(which python3.10) && make -j && cd ../../py'" >> ~/.bashrc
    echo "alias compileDebug='cd ../cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE=\$(which python3.10) && make -j && cd ../../py'" >> ~/.bashrc
    echo "alias compileRel='cd ../cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_EXECUTABLE=\$(which python3.10) && make -j && cd ../../py'" >> ~/.bashrc
    
    

    echo "set -g mouse on #For tmux version 2.1 and up" >> ~/.tmux.conf
    echo "unbind C-b # Unbind the default prefix" >> ~/.tmux.conf
    echo "set -g prefix C-a # Set new prefix to Ctrl+a" >> ~/.tmux.conf
    echo "bind C-a send-prefix # Bind the new prefix key" >> ~/.tmux.conf

    tmux source-file ~/.tmux.conf


    echo "cd ~/Advanced-Techniques-in-Chess-Engines/py" >> ~/.bashrc

    echo "conda activate Chess" >> ~/.bashrc
fi

source ~/.bashrc
