#!/bin/bash
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:k80:4
#SBATCH --job-name=pp1953
#SBATCH --mail-user=pp1953p@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --reservation=chung 


. ~/.bashrc
module load anaconda3/5.3.1

conda activate PPUU
conda install -n PPUU nb_conda_kernels


cd /scratch/pp1953/cml/ass/a-PyTorch-Tutorial-to-Object-Detection/Object-Detection-Custom-Dataset-pytorch/
python modified_train.py --name="adam_pretrained" --checkpoint >> "output6.txt"

# srun --nodes=1 --cpus-per-task=28 --mem=250GB --gres=gpu:k80:4 -- time=1:00:00 --reservation=chung --pty /bin/bash


#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=100000
#SBATCH --gres=gpu:p100:1
#SBATCH --job-name=pp1953
#SBATCH --mail-user=pp1953p@nyu.edu
#SBATCH --output=slurm_%j.out
